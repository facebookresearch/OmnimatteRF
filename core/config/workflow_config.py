# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any, Literal
from core.config.dataset_config import DatasetConfig

from core.config.object_config import ObjectConfig
from core.config.model_config import ModelConfig

@dataclass
class WorkflowConfig:
    device: str

    data_sources: dict[str, dict[str, Any]]
    dataset: DatasetConfig
    fg_model: ModelConfig
    bg_model: ModelConfig
    trainer: ObjectConfig

    contraction: Literal["none", "ndc", "mipnerf"]
    """Ray contraction scheme to use"""
