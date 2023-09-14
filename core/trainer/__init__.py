# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.model.matting_dataset import MattingDataset
from core.model.render_context import RenderContext
from core.module import CommonModel
from core.utils.trainer_utils import batch_to_frame
from lib.registry import create_registry, import_children
from lib.trainer import Trainer as BaseTrainer
from third_party.omnimatte.utils import flow_to_image
from utils.image_utils import save_image_np, visualize_array
from utils.io_utils import mkdir
from utils.render_utils import alpha_composite, detail_transfer, get_coords
from utils.string_utils import format_float

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_fg: bool,
        train_bg: bool,
        render_fg: bool,
        output: str,
        pbar_update_frequency: int,
        writer_path: Optional[str],
        use_writer: bool = True,
        depth_visualization_gamma: float = 2,
        fg_indexing_strategy: str | None = None,
        num_workers: int = 0,
    ):
        assert fg_indexing_strategy in {None, "x_neighbor"}

        super().__init__()
        self.output = mkdir(output)
        self.num_workers = num_workers

        if use_writer and writer_path is None:
            writer_path = mkdir(self.output / "tensorboard")
        if use_writer:
            self.writer = SummaryWriter(writer_path)
        else:
            self.writer = None

        self.depth_visualization_gamma = depth_visualization_gamma

        # dataset
        self.dataset: MattingDataset = None
        self.coords: Tensor = None

        # strategy to get training samples:
        #   if None, uniform random
        #   if not None, sample points as follows:
        #     - x_neighbor: first get N samples, then get N samples to the right of the first N
        self.fg_indexing_strategy = fg_indexing_strategy
        self.training_pixel_indices: Tensor = None

        # switches
        self.train_fg = train_fg
        self.train_bg = train_bg
        self.render_fg = render_fg

        # progress bar
        self.pbar = None
        self.start_step = 0
        self.total_step = 0
        self.pbar_update_frequency = pbar_update_frequency
        self.pbar_accumulated_loss = 0

        # type hints
        self.models: Dict[str, CommonModel] = self.models

    @property
    def fg(self):
        return self.models["fg"]

    @property
    def bg(self):
        return self.models["bg"]

    def set_dataset(self, dataset: MattingDataset):
        H, W = dataset.image_hw
        self.dataset = dataset
        self.coords = get_coords(H, W).view(-1, 2)

        if self.fg_indexing_strategy == "x_neighbor":
            # all pixels except the last column
            self.training_pixel_indices = torch.tensor([
                i for i in range(H * W) if i % W != (W - 1)
            ]).long()

    def _update_progress(self):
        if self.pbar_update_frequency <= 0:
            return

        pbar = self.pbar
        if self._check_step(self.pbar_update_frequency):
            inc_steps = self.global_step - self.start_step + 1 - pbar.n
            pbar.update(inc_steps)
            pbar.set_postfix_str(
                f"loss={format_float(self.pbar_accumulated_loss/inc_steps)}"
            )
            self.pbar_accumulated_loss = 0

    def _train_loader(self) -> Iterator:
        loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.img_batch_size,
            num_workers=self.num_workers,
        )
        while True:
            for batch in loader:
                yield batch

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True):
        super().load_state_dict(state_dict, strict)
        self.start_step = self.global_step

    def train(self, steps: int):
        if self.pbar_update_frequency > 0:
            self.pbar = tqdm(total=steps - self.global_step)

        if not self.train_fg:
            logger.info("Freeze FG model")
            self.fg.requires_grad_(False)
        if not self.train_bg:
            logger.info("Freeze BG model")
            self.bg.requires_grad_(False)
        super().train(steps)

    def _train_batch(self, batch: Dict[str, Tensor]):
        workspace = {}

        # train BG with first image in batch
        if self.train_bg:
            workspace.update(self._train_bg(batch_to_frame(batch, 0)))

        # train FG
        if self.train_fg:
            workspace.update(self._train_fg(batch))

        return workspace

    def init_test(self):
        self.set_model_eval()

    def test(self):
        self.init_test()
        self._test_full_sequence(self.output, self.test_dataset)

    def test_one_image(
        self,
        frame: Dict[str, Any],
        out_dir: Optional[Path] = None,
        name: Optional[str] = None,
        bg_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        render a full image, optionally save them to files

        frame: dataset data for one frame, i.e. obtained from dataset[i]
        out_dir, name: if specified, results will be saved to out_dir/key/name.(jpg|npy)

        returns: dictionary of images and data, all in numpy arrays
        """
        H, W = self.dataset.image_hw
        output = self._render_one_image(frame)
        render_fg = self.render_fg and not bg_only

        # gather visualizations and data
        result = {}

        def get_item(key):
            item = output[key].cpu().numpy()
            # reshape [H*W] axis to [H, W]
            # assuming that there is only one such axis
            shape = list(item.shape)
            if H * W in shape:
                hw_axis = shape.index(H * W)
                shape = shape[:hw_axis] + [H, W] + shape[hw_axis + 1:]
            return item.reshape(shape)

        def get_images(key):
            return np.clip(get_item(key), 0, 1)

        def save_image(key, image):
            result[key] = image
            if out_dir is None:
                return
            save_image_np(mkdir(out_dir / key) / f"{name}.png", result[key])

        # images
        bg_layer = get_item("bg_layer")[..., :3]

        if render_fg:
            br_scale = get_item("br_scale")
            save_image("br_scale", br_scale[..., 0])  # not a useful image, for dumping debug

            # since bg layer is trained with br_scale applied to composite rgb,
            # if we wish to evaluate it, it probably makes sense to apply it here as well
            # NOTE: should we not train apply br_scale to bg part completely?
            save_image("bg_layer", np.clip(bg_layer * br_scale, 0, 1))
            save_image("bg_layer_unscaled", np.clip(bg_layer, 0, 1))
        else:
            # when the model includes fg but we only render bg,
            # the key should have _unscaled
            unscaled_bg_layer_key = "bg_layer_unscaled" if self.render_fg else "bg_layer"
            br_scale = None
            save_image(unscaled_bg_layer_key, np.clip(bg_layer, 0, 1))

        if "bg_depths" in output:
            bg_depths = get_item("bg_depths")
            result["bg_depths_raw"] = bg_depths.copy()

            # per-image scale bg depths
            near, far = np.percentile(np.clip(bg_depths, self.bg.near, self.bg.far), [0.1, 99.9])
            bg_depths = np.clip((bg_depths - near) / (far - near), 0, 1)
            bg_depths = np.power(bg_depths, self.depth_visualization_gamma)
            bg_depths = visualize_array(bg_depths)
            save_image("bg_depths", bg_depths)

        # special handling: don't save full bg per-image
        if "bg_full" in output:
            result["bg_full"] = get_images("bg_full")

        if not render_fg:
            return result

        # composite_* are brightness-adjusted
        save_image("composite_rgb", get_images("composite_rgb"))

        # RGBA layers
        rgba_layers = np.concatenate(
            [
                get_images("rgb_layers") * br_scale,
                get_images("alpha_layers")[..., None],
            ],
            axis=-1,
        )

        L = len(rgba_layers)
        for i_layer in range(L):
            save_image(f"rgba_layer_{i_layer}", rgba_layers[i_layer])

        # detail transfer
        rgba_detail, fg_alpha = detail_transfer(
            frame["image"].numpy(), get_images("composite_rgb"), rgba_layers
        )
        save_image("fg_alpha", fg_alpha)
        for i_layer in range(L):
            save_image(f"detailed_layer_{i_layer}", rgba_detail[i_layer])

        # composite with detailed layer
        rgba_detail = torch.from_numpy(rgba_detail).view(1, L, H * W, 4)
        detailed_rgb = (
            alpha_composite(
                rgba_detail[..., 3:4],
                [rgba_detail[..., :3]],
                [torch.from_numpy(bg_layer * br_scale).view(1, H * W, 3)],
            )[0]
            .view(H, W, 3)
            .numpy()
        )
        save_image("detailed_rgb", detailed_rgb)

        # flow
        save_image("composite_flow", flow_to_image(get_item("composite_flow")))
        save_image("bg_flow", flow_to_image(get_item("bg_flow")))
        flow_layers = get_item("flow_layers")
        for i_layer in range(len(flow_layers)):
            save_image(f"flow_layer_{i_layer}",
                       flow_to_image(flow_layers[i_layer]))

        return result

    def _test_full_sequence(
        self, out_dir: Path, limit: int = 0
    ) -> Iterator[Dict[str, Any]]:
        dataset = self.dataset
        n_frames = min(len(dataset), limit) if limit > 0 else len(dataset)

        for i in tqdm(range(n_frames)):
            frame = dataset[i]
            with torch.no_grad():
                result = self.test_one_image(frame, out_dir, f"{i:08d}")
            if i == 0 and "bg_full" in result:
                save_image_np(out_dir / "bg_full.png", result["bg_full"])
            yield result, frame

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["model_kwargs_override"] = {
            name: model.get_kwargs_override() for name, model in self.models.items()
        }
        return state_dict

    def _step_batch(
        self,
        name: str,
        optimizers: List[torch.optim.Optimizer],
        forward: Callable,
        batch: Dict[str, Tensor],
    ):
        for optm in optimizers:
            optm.zero_grad()

        workspace = forward(batch)
        if self.global_step == self.start_step:
            self._log_workspace(name, workspace)

        loss, loss_dict = self.criterions[name](workspace)
        loss.backward()

        for optm in optimizers:
            optm.step()

        return workspace, loss_dict, float(loss)

    def _step_bg(self, batch: Dict[str, Tensor]):
        return self._step_batch(
            "bg", [self.optimizers["bg"]],
            self._train_bg, batch_to_frame(batch, 0)
        )

    def _step_fg(self, batch):
        optimizers = [self.optimizers["fg"]]
        if self._optimize_bg_within_fg_step():
            optimizers += [self.optimizers["bg"]]

        return self._step_batch("fg", optimizers, self._train_fg, batch)

    def _step(self, batch: Dict[str, Any]):
        # forward
        workspace = {}
        loss_dict = {}
        loss = 0

        def update_step(loss_prefix, step_workspace, step_loss_dict, step_loss):
            nonlocal loss
            workspace.update(step_workspace)
            loss_dict.update(
                {f"{loss_prefix}_{k}": v for k, v in step_loss_dict.items()}
            )
            loss += step_loss

        step_schedulers = []

        if self._should_step_bg():
            update_step("bg", *self._step_bg(batch))
            step_schedulers.append(self.schedulers["bg"])

        if self._should_step_fg():
            update_step("fg", *self._step_fg(batch))
            step_schedulers.append(self.schedulers["fg"])
            if self._optimize_bg_within_fg_step():
                step_schedulers.append(self.schedulers["bg"])

        # scheduler steps
        for scheduler in step_schedulers:
            scheduler.step()

        self._post_step(batch)

        # log things
        self._log_losses(loss_dict)
        self._log_learning_rates()

        # update progress bar
        self.pbar_accumulated_loss += float(loss)
        self._update_progress()

    @staticmethod
    def _log_workspace(name: str, workspace: Dict[str, Any]):
        workspace_info = []
        name_length = max((len(k) for k in workspace))
        for k, v in workspace.items():
            if isinstance(v, torch.Tensor):
                workspace_info += [
                    f"{k:<{name_length}} tensor {list(v.shape)}"]
            else:
                workspace_info += [f"{k:<{name_length}} ({type(v)}) {v}"]
        logger.info(f"Workspace [{name}]\n" + "\n".join(workspace_info))

    def _log_losses(self, loss_dict: Dict[str, Tensor]):
        for k, v in loss_dict.items():
            # filter out losses that are turned off
            if abs(float(v)) > 0:
                self.writer.add_scalar(
                    k, float(v), global_step=self.global_step)

    def _log_learning_rates(self):
        # log current learning rates
        for name in ["fg", "bg"]:
            optm = self.optimizers[name]
            for i in range(len(optm.param_groups)):
                self.writer.add_scalar(
                    f"{name}_lr_{i}", optm.param_groups[i]["lr"], self.global_step
                )

    def _get_context(self, coords: Tensor, is_train: bool) -> RenderContext:
        return RenderContext(
            coords, self.dataset, self.device, is_train, self.global_step
        )

    def _get_fg_context(self) -> RenderContext:
        """
        Our FG model does not use coords, and does not differentiate train / test
        """
        return self._get_context(None, True)

    def _get_fg_indices(
        self, bg_masks: Tensor, n_points: int, n_fg_points: int, n_bg_points: int
    ) -> Tensor:
        """
        Get indices for supervising FG

        bg_masks: [B, H, W] background region masks
        n_points: number of fg/bg points to sample by default
        n_fg_points, n_bg_points: if zero, use default (n_points);
            if either is negative, just sample 2*n_points random pixels (not considering fg/bg)
        """
        B, H, W = bg_masks.shape
        assert n_fg_points < 0 and n_bg_points < 0, "Not supported now"

        if self.fg_indexing_strategy is None:
            indices = np.concatenate(
                [
                    np.random.choice(H * W, [2 * n_points], replace=False)
                    for _ in range(B)
                ],
                axis=0,
            )
            return torch.from_numpy(indices)

        match self.fg_indexing_strategy:
            case "x_neighbor":
                pixel_indices = []
                for _ in range(B):
                    first_half = self.training_pixel_indices[np.random.choice(
                        len(self.training_pixel_indices), [n_points], replace=False)]
                    pixel_indices += [first_half, first_half + 1]
                return torch.cat(pixel_indices)

    # Subclass functions

    def _train_bg(self):
        pass

    def _train_fg(self):
        pass

    def _should_step_bg(self):
        return self.train_bg and not self._optimize_bg_within_fg_step()

    def _should_step_fg(self):
        return self.train_fg

    def _optimize_bg_within_fg_step(self):
        return False

    def _post_step(self):
        pass


_, register_trainer, build_trainer = create_registry("Trainer", Trainer)
import_children(__file__, __name__)
