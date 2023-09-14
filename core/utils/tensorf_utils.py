# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Dict

import torch
from torch import Tensor

from third_party.TensoRF.models.tensorBase import (MLPRender_Fea,
                                                   positional_encoding)
from third_party.TensoRF.models.tensoRF import TensorVMSplit

logger = logging.getLogger(__name__)


class MLPRender_Fea_Switchable(MLPRender_Fea):
    def __init__(self, inChannel, viewpe_skip_steps, viewpe=6, feape=6, featureC=128):
        super().__init__(inChannel, viewpe, feape, featureC)
        self.global_step = 0
        self.viewpe_skip_steps = viewpe_skip_steps
        self.viewpe_ch = (
            0
            if viewpe <= 0
            else positional_encoding(torch.zeros(1, 3), self.viewpe).shape[-1]
        )
        self.log_once_keys = set()
        logger.info(f"View PE channels: {self.viewpe_ch}")

    def _log_once(self, key, message):
        if key in self.log_once_keys:
            return
        self.log_once_keys.add(key)
        logger.info(message)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.global_step < self.viewpe_skip_steps:
            self._log_once("zeros_viewdir", "Using zeros for viewdirs")
            indata += [torch.zeros_like(viewdirs)]
        else:
            self._log_once("actual_viewdir", "Start using actual viewdirs")
            indata += [viewdirs]

        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]

        if self.viewpe > 0:
            if self.global_step < self.viewpe_skip_steps:
                self._log_once("zeros_viewdir_pe",
                               "Using zeros for viewdir PE")
                indata += [
                    torch.zeros(len(features), self.viewpe_ch,
                                device=features.device)
                ]
            else:
                self._log_once("actual_viewdir_pe",
                               "Start using acutal viewdir PE")
                indata += [positional_encoding(viewdirs, self.viewpe)]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MyTensorVMSplit(TensorVMSplit):
    def __init__(self, viewpe_skip_steps: int, **kargs):
        self.global_step = 0
        self.viewpe_skip_steps = viewpe_skip_steps

        super().__init__(**kargs)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea_Switchable(
                self.app_dim, self.viewpe_skip_steps, view_pe, fea_pe, featureC
            ).to(device)
        else:
            raise NotImplementedError()

    def forward(
        self,
        rays_chunk: Tensor,
        is_train: bool,
        ray_contraction: str,
        N_samples: int,
        **kwargs,
    ) -> Dict[str, Tensor]:
        self.renderModule.global_step = self.global_step
        return super().forward(
            rays_chunk,
            white_bg=False,
            is_train=is_train,
            ray_contraction=ray_contraction,
            N_samples=N_samples,
            **kwargs,
        )

    def query_render_rgb(self, points, viewdirs):
        app_features = self.compute_appfeature(points)
        rgbs = self.renderModule(points, viewdirs, app_features)
        return rgbs
