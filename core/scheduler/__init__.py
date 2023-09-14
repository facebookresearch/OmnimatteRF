# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.optim.lr_scheduler import _LRScheduler

from lib.registry import create_registry, import_children


class Scheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)


_, register_scheduler, build_scheduler = create_registry("Scheduler", Scheduler)
import_children(__file__, __name__)
