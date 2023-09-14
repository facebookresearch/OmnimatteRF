# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List

from core.loss import Loss, register_loss
from third_party.omnimatte.third_party.models.networks_lnr import cal_alpha_reg


@register_loss("alpha_reg")
class AlphaRegLoss(Loss):
    def __init__(
        self,
        alpha: float,
        inputs: List[str],
        lambda_alpha_l1: float,
        lambda_alpha_l0: float,
        l1_end_step: int,
    ):
        super().__init__(alpha, inputs=inputs)
        self.lambda_alpha_l1 = lambda_alpha_l1
        self.lambda_alpha_l0 = lambda_alpha_l0
        self.l1_end_step = l1_end_step

    def forward(self, pred, global_step):
        lambda_alpha_l1 = self.lambda_alpha_l1
        if global_step >= self.l1_end_step:
            lambda_alpha_l1 = 0

        return self.alpha * cal_alpha_reg(pred, lambda_alpha_l1, self.lambda_alpha_l0)
