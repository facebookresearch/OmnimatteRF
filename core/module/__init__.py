# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module

from core.model.render_context import RenderContext
from core.scheduler import build_scheduler
from lib.registry import create_registry, import_children


class CommonModel(Module):
    def forward(self, data: Dict[str, Tensor], ctx: RenderContext) -> None:
        raise NotImplementedError()

    def post_training_step(self, global_step: int) -> Dict[str, Any]:
        return {}

    def create_optimizer(self, params: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), **params)

    def create_scheduler(
        self, params: Dict[str, Any], optimizer: torch.optim.Optimizer
    ):
        name = params["name"]
        config = params["config"]
        config["optimizer"] = optimizer
        return build_scheduler(name, config)

    def get_kwargs_override(self) -> Dict[str, Any]:
        return {}


_, register_fg_model, build_fg_model = create_registry("FgModel", CommonModel)
_, register_bg_model, build_bg_model = create_registry("BgModel", CommonModel)
import_children(__file__, __name__)
