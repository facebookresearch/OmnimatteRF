# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any

from core.config.checkpoint_config import CheckpointConfig
from core.config.workflow_config import WorkflowConfig
from core.config.object_config import ObjectConfig


@dataclass
class SchedulerConfig:
    fg: ObjectConfig
    bg: ObjectConfig


@dataclass
class ValidationConfig:
    config: dict[str, Any]
    """ValidationHook constructor kwargs"""
    pretrain: bool
    """Whether to run this validation before training"""


@dataclass
class TrainConfig(WorkflowConfig):
    output: str
    checkpoint: str | None
    seed: int
    debug: bool

    n_steps: int
    validation: dict[str, ValidationConfig]

    # Optimization
    fg_losses: dict[str, ObjectConfig]
    bg_losses: dict[str, ObjectConfig]
    scheduler: SchedulerConfig

    # Checkpoint saving
    save_checkpoint: CheckpointConfig
    save_pretrain_checkpoint: bool
    save_final_checkpoint: bool

    # Checkpoint loading
    load_fg: bool
    load_bg: bool

    reset_global_step: bool
    """When loading checkpoint, set beginning global step to zero."""
    reset_bg_optimization: bool
    """When loading checkpoint, reset bg model optimizer and scheduler states."""
