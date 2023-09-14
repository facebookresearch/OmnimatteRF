# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from lib.loss import ComposedLoss

logger = logging.getLogger(__name__)


class TrainerEvents(Enum):
    POST_STEP = "post_step"
    POST_TRAIN = "post_train"
    PRE_TRAIN = "pre_train"
    POST_VALIDATION = "post_validation"


HookType = Callable[[TrainerEvents, "Trainer"], None]


class Trainer:
    def __init__(self):
        self.models: Dict[str, Module] = {}
        self.criterions: Dict[str, ComposedLoss] = {}
        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, _LRScheduler] = {}

        self.global_step = 0
        self.device = "cpu"
        self.hooks: Dict[TrainerEvents, List[HookType]] = defaultdict(list)

    def set_device(self, device: str):
        logger.info(f"Setting device: {device}")
        self.device = device

        for name in self.models:
            self.models[name].to(self.device)
        for name in self.criterions:
            self.criterions[name].to(self.device)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            "global_step": self.global_step,
            "rng": {
                "random": random.getstate(),
                "torch": torch.get_rng_state(),
                "torch.cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
            }
        }

        def save_dict(key: str, d: Dict[str, Module]):
            state_dict[key] = {}
            for name, module in d.items():
                state_dict[key][name] = module.state_dict()

        save_dict("models", self.models)
        save_dict("losses", self.criterions)
        save_dict("schedulers", self.schedulers)
        save_dict("optimizers", self.optimizers)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True):
        self.global_step = state_dict.get("global_step", 0)
        if "rng" in state_dict:
            random.setstate(state_dict["rng"]["random"])
            torch.set_rng_state(state_dict["rng"]["torch"])
            torch.cuda.set_rng_state_all(state_dict["rng"]["torch.cuda"])
            np.random.set_state(state_dict["rng"]["numpy"])

        def load_dict(key: str, d: Dict[str, Module], has_strict=True):
            kwargs = {}
            if has_strict:
                kwargs["strict"] = strict

            for name, module in d.items():
                module.load_state_dict(state_dict[key][name], **kwargs)

        load_dict("models", self.models)
        load_dict("losses", self.criterions)
        load_dict("schedulers", self.schedulers, False)
        load_dict("optimizers", self.optimizers, False)

    def set_model_train(self):
        for model in self.models.values():
            model.train()

    def set_model_eval(self):
        for model in self.models.values():
            model.eval()

    def train(self, steps: int):
        loader = self._train_loader()
        self.set_model_train()

        self._pre_train()
        self._execute_hooks(TrainerEvents.PRE_TRAIN)

        for _ in range(self.global_step, steps):
            batch = next(loader)
            self._step(batch)
            self.global_step += 1
            self._execute_hooks(TrainerEvents.POST_STEP)

        self._execute_hooks(TrainerEvents.POST_TRAIN)

    def register_hook(self, events: Union[List[TrainerEvents], TrainerEvents], hook: HookType):
        if not isinstance(events, list):
            events = [events]
        for event in events:
            logger.info(f"Adding {hook.__class__.__name__} to {event}")
            self.hooks[event].append(hook)

    def _execute_hooks(self, event: TrainerEvents):
        for hook in self.hooks[event]:
            hook(event, self)

    def _check_step(self, step_size: int, first_always: bool = True) -> bool:
        if step_size <= 0:
            return False
        return (self.global_step == 0 and first_always) or (
            self.global_step + 1
        ) % step_size == 0

    def _train_loader(self) -> DataLoader:
        raise NotImplementedError

    def _pre_train(self):
        pass

    def _step(self, batch: Dict[str, Any]):
        raise NotImplementedError
