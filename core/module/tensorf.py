# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from core.data import CameraDataSource
from core.model.render_context import RenderContext
from core.module import CommonModel, register_bg_model
from core.utils.tensorf_utils import MyTensorVMSplit
from third_party.TensoRF.models.tensoRF import TensorVMSplit
from third_party.TensoRF.utils import N_to_reso, cal_n_samples

logger = logging.getLogger(__name__)


class SafeTVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        if count_h == 0 or count_w == 0:
            return 0

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


@register_bg_model("tensorf")
class TensoRFModel(CommonModel):
    def __init__(
        self,
        aabb: Tensor,
        near: float,
        far: float,
        hwf: Tuple[int, int, float],
        contraction: str,
        device: str,
        N_voxel_init: int,
        N_voxel_final: int,
        upsamp_list: List[int],
        update_AlphaMask_list: List[int],
        density_n_comp: List[int],
        appearance_n_comp: List[int],
        app_dim: int,
        shadingMode: str,
        alphaMask_thres: float,
        density_shift: float,
        distance_scale: float,
        pos_pe: int,
        view_pe: int,
        fea_pe: int,
        featureC: int,
        step_ratio: float,
        fea2denseAct: str,
        nSamples: int,
        lr_upsample_reset: bool,
        viewpe_skip_steps: int,
        # continuing training
        prev_global_step: int = 0,
        global_step_offset: int = 0,
        # overriding kwargs
        gridSize: Optional[List[float]] = None,
        rayMarch_weight_thres: float = 0.0001,
    ):
        super().__init__()

        if gridSize is None:
            gridSize = N_to_reso(N_voxel_init, aabb)

        if contraction == "ndc":
            logger.info("NDC is enabled, setting near to 0 and far to 1.")
            near = 0
            far = 1
        elif contraction == "mipnerf":
            logger.info("Mip-NeRF ray contraction, setting near to 0 and far to 256")
            near = 0
            far = 256

        self.contraction = contraction
        self.hwf = hwf
        self.near = near
        self.far = far
        self.max_nSamples = nSamples
        self.lr_upsample_reset = lr_upsample_reset

        self.reso_cur = gridSize
        self.reso_mask = tuple(self.reso_cur)
        self.nSamples = min(nSamples, cal_n_samples(self.reso_cur, step_ratio))
        self.step_ratio = step_ratio
        self.tvloss = SafeTVLoss()

        aabb = aabb.to(device)

        if viewpe_skip_steps > 0:
            model_cls = partial(MyTensorVMSplit, viewpe_skip_steps)
        else:
            model_cls = TensorVMSplit

        self.model = model_cls(
            aabb=aabb,
            gridSize=gridSize,
            device=device,
            density_n_comp=density_n_comp,
            appearance_n_comp=appearance_n_comp,
            app_dim=app_dim,
            near_far=[near, far],
            shadingMode=shadingMode,
            alphaMask_thres=alphaMask_thres,
            density_shift=density_shift,
            distance_scale=distance_scale,
            pos_pe=pos_pe,
            view_pe=view_pe,
            fea_pe=fea_pe,
            featureC=featureC,
            step_ratio=step_ratio,
            fea2denseAct=fea2denseAct,
            rayMarch_weight_thres=rayMarch_weight_thres,
        )

        self.N_voxel_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(N_voxel_init),
                        np.log(N_voxel_final),
                        len(upsamp_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        self.update_AlphaMask_list = update_AlphaMask_list
        self.upsamp_list = upsamp_list

        # remove upsample / alpha steps that's already done
        if prev_global_step > 0:
            logger.info(f"Remove update lists before step {prev_global_step}")

            self.update_AlphaMask_list = [
                v for v in update_AlphaMask_list if v > prev_global_step
            ]
            self.N_voxel_list = [
                self.N_voxel_list[i]
                for i in range(len(self.N_voxel_list))
                if self.upsamp_list[i] > prev_global_step
            ]
            self.upsamp_list = [
                v for v in self.upsamp_list if v > prev_global_step]

            logger.info(
                f"update_AlphaMask_list = {self.update_AlphaMask_list}")
            logger.info(f"N_voxel_list = {self.N_voxel_list}")
            logger.info(f"upsamp_list = {self.upsamp_list}")

        self.global_step_offset = global_step_offset

        self.optimizer_params = None
        self.scheduler_params = None

    def forward(
        self,
        data: Dict[str, Tensor],
        ctx: RenderContext,
        additional_keys: Optional[List[str]] = None,
    ) -> None:
        device = ctx.device
        additional_keys = additional_keys or []
        self.model.global_step = ctx.global_step + self.global_step_offset

        camera: CameraDataSource = ctx.dataset.global_data["camera"]
        rays_o, rays_d = camera.get_rays(int(data["data_idx"]), ctx.coords)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        if ctx.ray_offset is not None:
            offset = torch.tensor(ctx.ray_offset).view(1, 3).to(device)
            rays_o = rays_o + offset

        rays = torch.cat((rays_o, rays_d), dim=1)
        net_output = self.model.forward(
            rays_chunk=rays,
            is_train=ctx.is_train,
            ray_contraction=self.contraction,
            N_samples=self.nSamples,
            return_everything_dict=True,
        )

        bg_depths = net_output["depth_map"]
        if self.contraction == "mipnerf":
            bg_depths = 1 / torch.clamp_min(bg_depths, 1e-6)
        elif self.contraction == "ndc":
            bg_depths = -1 * bg_depths + 1

        ctx.output.update(
            {
                "bg_rgb": net_output["rgb_map"],
                "bg_depths": bg_depths,
                **{key: net_output[key] for key in additional_keys},
            }
        )

        if not ctx.is_train:
            return

        ctx.output.update(
            {
                "reg_tv_density": self.model.TV_loss_density(self.tvloss),
                "reg_tv_app": self.model.TV_loss_app(self.tvloss),
                "bg_weight": net_output["weight"],
                "bg_z_vals": net_output["z_vals"],
            }
        )

    def post_training_step(self, global_step: int) -> Dict[str, Any]:
        self._update_alpha_mask(global_step + self.global_step_offset)
        return self._update_grid(global_step + self.global_step_offset)

    def _update_alpha_mask(self, global_step: int) -> None:
        if global_step not in self.update_AlphaMask_list:
            return

        reso_cur = self.reso_cur
        if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3:
            self.reso_mask = tuple(reso_cur)

        new_aabb = self.model.updateAlphaMask(self.reso_mask)
        if global_step == self.update_AlphaMask_list[0]:
            self.model.shrink(new_aabb)
            # TODO: TensoRF updates L1_reg_weight here,
            # but it's zero by default so we skip it for now

        return

    def _update_grid(self, global_step: int) -> Dict[str, Any]:
        result = {}

        if global_step not in self.upsamp_list:
            return result

        n_voxels = self.N_voxel_list.pop(0)
        self.reso_cur = N_to_reso(n_voxels, self.model.aabb)
        self.nSamples = min(
            self.max_nSamples, cal_n_samples(self.reso_cur, self.step_ratio)
        )
        self.model.upsample_volume_grid(self.reso_cur)

        if self.lr_upsample_reset:
            result["optimizer"] = self.create_optimizer(self.optimizer_params)
            result["scheduler"] = self.create_scheduler(
                self.scheduler_params, result["optimizer"]
            )

        return result

    def create_optimizer(self, params: Dict[str, Any]) -> torch.optim.Optimizer:
        self.optimizer_params = params
        groups = self.model.get_optparam_groups(
            params["lr"], params["lr_basis"])
        return torch.optim.Adam(groups, betas=(0.9, 0.99))

    def create_scheduler(
        self, params: Dict[str, Any], optimizer: torch.optim.Optimizer
    ):
        self.scheduler_params = params
        return super().create_scheduler(params, optimizer)

    def get_kwargs_override(self) -> Dict[str, Any]:
        kwargs = self.model.get_kwargs()
        [near, far] = kwargs.pop("near_far")
        kwargs.update(
            {
                "near": near,
                "far": far,
            }
        )
        return kwargs

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        cp = {"state_dict": state_dict}

        for key in ["alphaMask.aabb", "alphaMask.mask", "alphaMask.shape"]:
            if key not in state_dict:
                continue

            cp[key] = state_dict.pop(key)

        self.model.load(cp)

    def state_dict(self):
        cp = self.model.get_ckpt()
        cp.pop("kwargs")

        state_dict = cp["state_dict"]
        for k, v in cp.items():
            if k != "state_dict":
                state_dict[k] = v

        return state_dict
