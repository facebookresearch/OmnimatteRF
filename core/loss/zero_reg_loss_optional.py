# Copyright (c) Meta Platforms, Inc. and affiliates.

from core.loss import Loss, register_loss


@register_loss("zero_reg_optional")
class ZeroRegLossOptional(Loss):
    def __init__(self, alpha: float, optional_input: str):
        super().__init__(alpha)
        self.optional_input = optional_input

    def forward(self, *args, **kwargs):
        pred = kwargs.get(self.optional_input)
        if pred is None:
            return 0
        return self.alpha * pred
