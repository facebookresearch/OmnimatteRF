# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from core.model.render_context import RenderContext
from core.module import CommonModel, register_bg_model, register_fg_model


@register_fg_model("dummy")
@register_bg_model("dummy")
class DummyModel(CommonModel):
    """A model that does not produce any output"""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, data: Dict[str, Tensor], ctx: RenderContext) -> None:
        return
