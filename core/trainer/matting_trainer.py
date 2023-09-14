# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from core.module import CommonModel
from core.trainer import Trainer, register_trainer
from core.utils.omnimatte_utils import render_omnimatte_fg
from core.utils.trainer_utils import batch_to_frame, frame_to_batch
from third_party.omnimatte.third_party.models.networks_lnr import MaskLoss

logger = logging.getLogger(__name__)


@register_trainer("matting")
class MattingTrainer(Trainer):
    def __init__(
        self,
        img_batch_size: int,  # images in each batch
        ray_batch_size: int,  # rays to sample in each BG step (all from first image)
        output: str,
        train_bg: bool = False,
        train_fg: bool = False,
        render_fg: bool = True,
        pbar_update_frequency: int = 0,
        writer_path: Optional[str] = None,
        fg_batch_size_fg: int = 0,  # fg-region pixels from EACH IMAGE in each FG step
        fg_batch_size_bg: int = 0,  # bg-region pixels from EACH IMAGE in each FG step
        pretrain_bg_step: int = 0,  # pretrain BG for this number of steps
        bg_composite_grad_step: int = -1,  # if non-negative, enable BG training from composite RGB after this number of steps
        fg_flow_mode: str = "zeros",  # how to fill bg flow of fg region
        tqdm_at_rays: bool = False,
        log_raw_mask_loss: bool = False,
        use_writer: bool = True,
        depth_visualization_gamma: float = 1,
        train_full_image: bool = False,
        prerender_bg: bool = False,
        prerender_bg_path: str = None,
        fg_indexing_strategy: str | None = None,
        num_workers: int = 0,
    ):
        assert fg_flow_mode in {"gt", "zeros"}
        super().__init__(
            train_fg,
            train_bg,
            render_fg,
            output,
            pbar_update_frequency,
            writer_path,
            use_writer,
            depth_visualization_gamma,
            fg_indexing_strategy,
            num_workers,
        )

        self.tqdm_at_rays = tqdm_at_rays

        self.train_bg = train_bg
        self.train_fg = train_fg
        self.render_fg = render_fg
        self.img_batch_size = img_batch_size
        self.ray_batch_size = ray_batch_size
        self.fg_batch_size_fg = fg_batch_size_fg
        self.fg_batch_size_bg = fg_batch_size_bg
        self.pretrain_bg_step = pretrain_bg_step
        self.bg_composite_grad_step = bg_composite_grad_step
        self.fg_flow_mode = fg_flow_mode
        self.log_raw_mask_loss = log_raw_mask_loss

        if train_full_image:
            assert prerender_bg
        self.train_full_image = train_full_image
        self.prerender_bg = prerender_bg
        self.prerender_bg_path = prerender_bg_path
        self.rendered_bg: Dict[str, Tensor] = {}

        if log_raw_mask_loss:
            self.mask_loss = MaskLoss()

    def _pre_train(self):
        super()._pre_train()

        if self.prerender_bg:
            self.rendered_bg = self._get_prerender()

    def _get_prerender(self):
        """
        Returns dictionary of key -> (N, *) Tensors
        N is number of datapoints
        * is shape of each output from bg render
        """
        root = Path(self.prerender_bg_path)
        root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Prerender bg, file location: {root}")

        rendered_bg = defaultdict(list)
        for i in tqdm(range(len(self.dataset))):
            file = root / f"{i:04d}.pth"
            if os.path.isfile(file):
                output = torch.load(file)
            else:
                with torch.no_grad():
                    output = self._render_one_bg(self.dataset[i])
                output = {k: v.cpu() for k, v in output.items()}
                torch.save(output, file)

            for k, v in output.items():
                rendered_bg[k].append(v)

        rendered_bg = {k: torch.stack(v, dim=0) for k, v in rendered_bg.items()}
        return rendered_bg

    def _post_step(self, batch):
        self.fg.post_training_step(self.global_step)
        bg_updates = self.bg.post_training_step(self.global_step)
        if "optimizer" in bg_updates:
            self.optimizers["bg"] = bg_updates["optimizer"]
        if "scheduler" in bg_updates:
            self.schedulers["bg"] = bg_updates["scheduler"]

    def _render_one_bg(self, frame: Dict[str, Any]) -> Dict[str, Tensor]:
        batch_size = self.ray_batch_size

        output = defaultdict(list)

        iterator = range(0, len(self.coords), batch_size)
        if self.tqdm_at_rays:
            iterator = tqdm(iterator)
        for i in iterator:
            coords = self.coords[i: i + batch_size]
            ctx = self._get_context(coords, is_train=False)
            self.bg.forward(frame, ctx)

            for key in ["bg_rgb", "bg_depths"]:
                output[key].append(ctx.output[key])

        output = {k: torch.cat(v, dim=0) for k, v in output.items()}
        return output

    def _render_one_image(self, frame: Dict[str, Any]) -> Dict[str, Tensor]:
        if self.prerender_bg:
            output = {
                k: v[int(frame["data_idx"])].to(self.device)
                for k, v in self.rendered_bg.items()
            }
        else:
            output = self._render_one_bg(frame)
        bg_rgb = output["bg_rgb"]

        if self.render_fg:
            bg_rgba = torch.cat([bg_rgb, torch.ones_like(bg_rgb[:, :1])], dim=1)[None]
            output.update(
                batch_to_frame(
                    self._render_fg(bg_rgba, frame_to_batch(frame), None, False),
                    0,
                )
            )
        else:
            output["bg_layer"] = bg_rgb

        return output

    def _train_bg(self, frame: Dict[str, Tensor]) -> Dict[str, Tensor]:
        bg_model: CommonModel = self.bg
        H, W = frame["image"].shape[:2]
        device = self.device

        # this mask may be dynamic or trainable when FG is trained
        background_mask = frame["background_mask"].view(-1)

        # sample a batch of coords for training BG
        indices = np.arange(H * W)[background_mask]
        indices = np.random.choice(indices, [self.ray_batch_size], replace=False)
        coords = self.coords[indices]

        ctx = self._get_context(coords, is_train=True)
        ctx.output["bg_gt_rgb"] = frame["image"][..., :3].view(-1, 3)[indices].to(device)

        bg_model.forward(frame, ctx)

        if "rgba_masks" in frame:
            fg_rgb = frame["rgba_masks"][..., :3].view(-1, 3)[indices].to(device)
            fg_alpha = frame["rgba_masks"][..., 3:].view(-1, 1)[indices].to(device)
            bg_rgb = ctx.output["bg_rgb"]
            ctx.output["bg_rgb"] = fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)

        if "depth_matching" in self.criterions["bg"] or "robust_depth_matching" in self.criterions["bg"]:
            ctx.output["gt_depths"] = frame["depths"].view(-1)[indices].to(self.device)
            ctx.output["depth_mask"] = 1

        ctx.output["global_step"] = self.global_step
        return ctx.output

    def _train_fg(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        bg_model: CommonModel = self.bg
        B, L, H, W = batch["masks"].shape
        workspace = {}

        if self.train_full_image:
            indices = None
            b_indices = batch["data_idx"]
            bg_rgb = self.rendered_bg["bg_rgb"][b_indices]
            bg_rgb = bg_rgb.view(B, H*W, 3).to(self.device)
            bg_rgba = torch.cat([bg_rgb, torch.ones_like(bg_rgb[..., :1])], dim=-1)
        else:
            # sample points from BG and FG regions
            indices = self._get_fg_indices(
                batch["background_mask"],
                self.ray_batch_size,
                self.fg_batch_size_fg,
                self.fg_batch_size_bg,
            )
            coords = self.coords[indices].view(B, -1, 2)

            # get BG rendering (sparse)
            bg_rgb = []

            def render_bg_frame(i: int, is_train: bool):
                bg_ctx = self._get_context(coords[i], is_train=is_train)
                bg_model.forward(batch_to_frame(batch, i), bg_ctx)
                bg_rgb.append(bg_ctx.output["bg_rgb"])
                if is_train:
                    workspace.update(
                        {k: v for k, v in bg_ctx.output.items() if k != "bg_rgb"}
                    )

            # use some of the composite RGB pixels to supervise BG
            # not using all because of memory
            n_bg_grad_frames = 1 if self._optimize_bg_within_fg_step() else 0
            for i in range(n_bg_grad_frames):
                render_bg_frame(i, True)
            with torch.no_grad():
                for i in range(n_bg_grad_frames, B):
                    render_bg_frame(i, False)

            bg_rgb = torch.cat(bg_rgb, dim=0)  # [B*S, 3]
            bg_rgba = torch.cat([bg_rgb, torch.ones_like(bg_rgb[:, :1])], dim=1)
            bg_rgba = bg_rgba.view(B, -1, 4)  # [B, S, 4]

        # get FG rendering at sparse samples
        workspace.update(self._render_fg(bg_rgba, batch, indices, True))

        if "depth_matching" in self.criterions["fg"] or "robust_depth_matching" in self.criterions["fg"]:
            bg_indices = indices.view(B, -1)[0]
            workspace["gt_depths"] = batch["depths"][0, bg_indices].to(self.device)
            workspace["depth_mask"] = 1 - workspace["composite_alpha"].detach()[0]

        if self.log_raw_mask_loss:
            with torch.no_grad():
                mask_loss = self.mask_loss(
                    workspace["alpha_layers"], workspace["fg_gt_mask"]
                )
            self.writer.add_scalar(
                "fg_raw_mask_loss", float(mask_loss), global_step=self.global_step
            )
        return workspace

    def _render_fg(
        self,
        bg_rgba: Tensor,
        batch: Dict[str, Tensor],
        indices: Optional[Tensor],
        is_train: bool,
    ) -> Dict[str, Tensor]:
        """
        bg_rgba: [B, S, 4]
        """
        B = bg_rgba.shape[0]
        batch["bg_rgba"] = bg_rgba

        if self.fg_flow_mode == "gt":
            batch["bg_flow"] = batch["flow"].permute(0, 3, 1, 2).reshape(B, 2, -1)
        elif self.fg_flow_mode == "zeros":
            # for blending, set foreground region flow to zeros
            bg_flow = batch["flow"].clone()
            bg_mask = batch["background_mask"]
            bg_flow[~bg_mask] = 0
            batch["bg_flow"] = bg_flow.permute(0, 3, 1, 2).reshape(B, 2, -1)

        batch["flow_confidence"][batch["background_mask"]] = 0

        return render_omnimatte_fg(
            self.dataset,
            batch,
            indices,
            self.fg.forward,
            self._get_fg_context,
            self.coords,
            self.global_step,
            is_train,
        )

    def _should_step_fg(self):
        return self.train_fg and self.global_step >= self.pretrain_bg_step

    def _optimize_bg_within_fg_step(self):
        return (
            self.train_bg
            and self.train_fg
            and self.bg_composite_grad_step >= 0
            and self.global_step >= self.bg_composite_grad_step
        )
