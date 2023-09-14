# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path
from typing import Optional

import torch

from lib.hook import Hook
from lib.trainer import Trainer

logger = logging.getLogger(__name__)


class SaveCheckpointHook(Hook):
    def __init__(
        self,
        folder: str,
        step_size: Optional[int] = None,
        min_step: Optional[int] = None,
    ):
        super().__init__(step_size)
        self.folder = Path(folder)
        self.min_step = 0 if min_step is None else min_step

    def execute(self, trainer: Trainer):
        if trainer.global_step > 0 and trainer.global_step < self.min_step:
            logger.info(f"Checkpoint saving skipped by min_step ({self.min_step})")
            return
        torch.save(trainer.state_dict(), self.folder / f"checkpoint_{trainer.global_step}.pth")
