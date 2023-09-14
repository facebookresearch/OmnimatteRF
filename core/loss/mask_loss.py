# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import List

import torch

from core.loss import Loss, register_loss
from third_party.omnimatte.third_party.models.networks_lnr import MaskLoss

logger = logging.getLogger(__name__)


@register_loss("mask_loss")
class MyMaskLoss(Loss):
    def __init__(
        self,
        alpha: float,
        inputs: List[str],
        reduce_threshold: float,
    ):
        """Compute the mask loss in Omnimatte.

        reduce_threshold: After the loss (pre alpha) is smaller than this threshold, alpha is reduced to 1/10 of original from the next step.
                        Further, this loss is disabled after another the same number of steps it took to reach the threshold.
        """
        super().__init__(alpha, inputs=inputs)
        self.loss = MaskLoss()

        # keep track of when the threshold is reached
        self.register_buffer("reduce_step", torch.zeros(1, dtype=torch.long))
        self.reduce_threshold = reduce_threshold

    def forward(self, pred, target, global_step):
        reduce_step = int(self.reduce_step)

        if reduce_step > 0:
            # disable this loss when the reduced weight has been used long enough
            if global_step > reduce_step * 2:
                return 0

            mult = 0.1
        else:
            mult = 1

        # our alpha is in [0, 1] while omnimatte is [-1, 1]
        pred = pred * 2 - 1
        loss = self.loss(pred, target)

        if reduce_step == 0 and float(loss) < self.reduce_threshold:
            self.reduce_step[0] = global_step
            logger.info(
                f"Start reducing mask loss (step {global_step}), raw loss is {float(loss)} < {self.reduce_threshold}"
            )

        return self.alpha * mult * self.loss(pred, target)
