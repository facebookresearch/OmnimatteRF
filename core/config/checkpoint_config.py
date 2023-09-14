# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    step_size: int
    min_step: int
    folder: str
