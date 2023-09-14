# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from core.trainer import Trainer, register_trainer
from core.utils.omnimatte_utils import render_omnimatte_fg
from core.utils.trainer_utils import batch_to_frame, frame_to_batch
from third_party.omnimatte.data.omnimatte_dataset import get_background_flow

logger = logging.getLogger(__name__)


@register_trainer("omnimatte")
class OmnimatteTrainer(Trainer):
    def __init__(
        self,
        img_batch_size: int,  # images in each batch
        output: str,
        pbar_update_frequency: int = 0,
        writer_path: Optional[str] = None,
        ray_batch_size: int = 0,  # if positive, supervise sparse samples; otherwise supervise whole image
        fg_batch_size_fg: int = 0,  # fg-region pixels from EACH IMAGE in each FG step
        fg_batch_size_bg: int = 0,  # bg-region pixels from EACH IMAGE in each FG step
        use_writer: bool = True,
        fg_indexing_strategy: str | None = None,
        num_workers: int = 0,
    ):
        super().__init__(
            True,
            False,
            True,
            output,
            pbar_update_frequency,
            writer_path,
            use_writer,
            fg_indexing_strategy=fg_indexing_strategy,
            num_workers=num_workers,
        )

        self.img_batch_size = img_batch_size
        self.ray_batch_size = ray_batch_size
        self.fg_batch_size_fg = fg_batch_size_fg
        self.fg_batch_size_bg = fg_batch_size_bg

    def _post_step(self, batch):
        self.fg.post_training_step(self.global_step)

    def _render_one_image(self, frame: Dict[str, Any]) -> Dict[str, Tensor]:
        return batch_to_frame(
            self._render_fg(
                frame_to_batch(frame),
                None,
                False,
            ),
            0,
        )

    def _train_fg(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        B, L, H, W = batch["masks"].shape

        # sample points
        if self.ray_batch_size == 0:
            indices = None
        else:
            indices = self._get_fg_indices(
                batch["background_mask"],
                self.ray_batch_size,
                self.fg_batch_size_fg,
                self.fg_batch_size_bg,
            )

        # get FG rendering at sparse samples
        return self._render_fg(batch, indices, True)

    def _render_fg(
        self,
        batch: Dict[str, Tensor],
        indices: Optional[Tensor],
        is_train: bool,
    ):
        B, L, H, W = batch["masks"].shape
        dataset = self.dataset
        homography = dataset.global_data["homography"]
        homography_size = dataset.global_data["homography_size"]

        # bg flow from homography
        bg_flow = []
        for i in range(B):
            bg_flow.append(
                get_background_flow(
                    batch["homography"][i],
                    homography[int(batch["data_idx"][i]) + 1],
                    homography_size[1],
                    homography_size[0],
                    W,
                    H,
                ).view(2, -1)
            )
        batch["bg_flow"] = torch.stack(bg_flow, dim=0)  # [B, 2, H*W]

        batch["flow_confidence"][batch["background_mask"]] = 0

        return render_omnimatte_fg(
            dataset,
            batch,
            indices,
            self.fg.forward,
            self._get_fg_context,
            self.coords,
            self.global_step,
            is_train,
        )
