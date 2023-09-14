# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any

from core.config.object_config import ObjectConfig


@dataclass
class ModelConfig(ObjectConfig):
    train: bool
    """Whether this model is trained"""

    optim: dict[str, Any]
    """Adam optimizer constructor kwargs"""
