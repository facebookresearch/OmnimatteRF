# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any

from core.config.object_config import ObjectConfig


@dataclass
class DatasetConfig:
    path: str
    image_subpath: str
    scale: float
    source_configs: list[ObjectConfig]
    sources_injection: dict[str, Any]
