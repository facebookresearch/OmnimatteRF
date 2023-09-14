# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Dict, Optional

from torch import Tensor

from core.model.matting_dataset import MattingDataset


@dataclass
class RenderContext:
    coords: Tensor

    dataset: MattingDataset
    device: str
    is_train: bool
    global_step: int

    ray_offset: Optional[list] = None

    output: Dict[str, Tensor] = field(default_factory=dict)
