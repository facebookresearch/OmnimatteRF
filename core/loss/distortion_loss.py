# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch import Tensor
from torch_efficient_distloss import eff_distloss_native

from core.loss import Loss, register_loss


@register_loss("distortion")
class DistortionLoss(Loss):
    def forward(self, weight: Tensor, z_vals: Tensor):
        interval = 1 / weight.shape[1]
        z_vals = z_vals.expand(len(weight), -1)
        return self.alpha * eff_distloss_native(weight, z_vals, interval)
