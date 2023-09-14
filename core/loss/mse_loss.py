# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.nn.functional import mse_loss

from core.loss import Loss, register_loss


@register_loss("mse")
class MseLoss(Loss):
    def forward(self, pred, target):
        return self.alpha * mse_loss(pred, target)
