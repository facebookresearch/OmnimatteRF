# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional
from lib.trainer import Trainer, TrainerEvents
import logging


logger = logging.getLogger(__name__)


class Hook:
    def __init__(self, step_size: Optional[int] = None):
        self.step_size = step_size
        self._last_execution_step = -1

    def __call__(self, event: TrainerEvents, trainer: Trainer):
        if not self._valid_step(event, trainer.global_step):
            return
        logger.info(
            f"Executing {self.__class__.__name__} "
            f"(Current Step: {trainer.global_step} "
            f"Last Step {self._last_execution_step})"
        )
        self._last_execution_step = trainer.global_step
        self.execute(trainer)

    def _valid_step(self, event: TrainerEvents, global_step: int):
        if self._last_execution_step == global_step:
            return False
        if event != TrainerEvents.POST_STEP:
            return True
        return self.step_size is None or global_step % self.step_size == 0

    def execute(self, trainer: Trainer):
        raise NotImplementedError
