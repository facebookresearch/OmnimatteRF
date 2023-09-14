# Copyright (c) Meta Platforms, Inc. and affiliates.

from core.loss import Loss, register_loss


@register_loss("zero_reg")
class ZeroRegLoss(Loss):
    def forward(self, pred):
        return self.alpha * pred
