# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass

from core.config.workflow_config import WorkflowConfig


@dataclass
class EvalConfig(WorkflowConfig):
    output: str

    checkpoint: str
    train_config_file: str
    data_root: str
    dataset_name: str | None
    experiment: str | None
    step: str | None
    write_videos: bool

    alpha_threshold: float
    """Alpha threshold when generating pred_fg masks"""
    eval_bg_layer: bool
    """Evaluate background layer against input image at input background mask"""

    debug_count: int
    """Debug: only render this number of frames"""
    raw_data_keys: list[str]
    """Debug: save raw data of these keys as npy"""
    raw_data_indices: list[int]
    """Debug: indices of frames to save raw data"""
