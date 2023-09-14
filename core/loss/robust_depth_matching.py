# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional

import torch
from torch import Tensor

from core.loss import Loss, register_loss


@register_loss("robust_depth_matching")
class RobustDepthMatchingLoss(Loss):
    def __init__(self, alpha: float = 1, inputs: Optional[list[str]] = None, start_step: int = 0):
        super().__init__(alpha, inputs)
        self.start_step = start_step

    def forward(self, pred: Tensor, gt: Tensor, mask: Tensor, global_step: int):
        if global_step < self.start_step:
            return 0

        return self.alpha * compute_depth_loss(pred, gt, mask)


def compute_depth_loss(dyn_depth, gt_depth, mask):
    # https://github.com/gaochen315/DynamicNeRF/blob/c417fb207ef352f7e97521a786c66680218a13af/run_nerf_helpers.py#L483

    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return torch.mean((mask * (dyn_depth_norm - gt_depth_norm)) ** 2)
