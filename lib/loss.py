# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Loss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        inputs: Optional[List[str]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.inputs = inputs

    def __call__(self, workspace: Dict[str, Any]):
        if self.inputs is not None:
            missing = [k for k in self.inputs if k not in workspace]
            if len(missing) > 0:
                raise ValueError(f"Missing keys from workspace: {missing}")

            return super().__call__(*(workspace[k] for k in self.inputs))
        else:
            return super().__call__(**workspace)


class ComposedLoss(nn.ModuleDict):
    def validate(self):
        invalid = [name for name, loss in self.items()
                   if not isinstance(loss, Loss)]
        if len(invalid) > 0:
            raise ValueError(
                f"Found loss that are not subclasses of Loss: {invalid}")

    def forward(self, workspace: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = {}
        for name, loss in self.items():
            ret = loss(workspace)
            if ret is None:
                continue

            loss_dict[name] = ret

        total_loss = sum(loss_dict.values())
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
