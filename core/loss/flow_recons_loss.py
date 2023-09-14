# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.nn.functional import l1_loss

from core.loss import Loss, register_loss


@register_loss("flow_recons")
class FlowReconsLoss(Loss):
    def forward(self, pred, target, confidence):
        return self.alpha * l1_loss(pred * confidence, target * confidence)
