# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.nn.functional import l1_loss

from core.loss import Loss, register_loss


@register_loss("l1")
class L1Loss(Loss):
    def forward(self, pred, target):
        return self.alpha * l1_loss(pred, target)
