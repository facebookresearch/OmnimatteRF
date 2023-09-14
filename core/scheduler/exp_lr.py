# Copyright (c) Meta Platforms, Inc. and affiliates.

from core.scheduler import Scheduler, register_scheduler


@register_scheduler("exp_lr")
class ExpLR(Scheduler):
    def __init__(
        self,
        optimizer,
        decay_start,
        decay_rate,
        decay_steps,
        min_rate,
        last_epoch=-1,
        verbose=False,
    ):
        self.decay_start = decay_start
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_rate = min_rate
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.decay_start:
            return list(self.base_lrs)
        rate = max(
            self.min_rate,
            (
                self.decay_rate
                ** ((self.last_epoch - self.decay_start) / self.decay_steps)
            ),
        )
        return [rate * base_lr for base_lr in self.base_lrs]
