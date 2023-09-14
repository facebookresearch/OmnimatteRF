# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict

from torch import Tensor


def frame_to_batch(frame: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Convert frame data to batch data by prepending a [1] dimension, e.g.
    (H, W, C) tensor -> (1, H, W, C) tensor
    """
    return {k: v[None] for k, v in frame.items()}


def batch_to_frame(batch: Dict[str, Tensor], idx: int) -> Dict[str, Tensor]:
    """
    Convert batch data to frame data by taking the idx item, e.g.
    (B, H, W, C) tensor -> (H, W, C) tensor
    """
    return {k: v[idx] for k, v in batch.items()}
