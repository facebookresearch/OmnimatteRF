# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any


@dataclass
class ObjectConfig:
    name: str
    config: dict[str, Any]
