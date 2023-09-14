# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path

import numpy as np

from core.data import CameraDataSource, register_data_source
from utils.array_utils import log_stats

logger = logging.getLogger(__name__)


@register_data_source("llff_camera")
class LlffCameraDataSource(CameraDataSource):
    def __init__(
        self,
        root: Path,
        image_hw: tuple[int, int],
        subpath: str,
        near_p: float,
        far_p: float,
        near_min: float,
        far_max: float,
        contraction: str,
        compute_ndc_aabb: bool = False,
        scene_scale: float = 1,
        process_poses: bool = True,
    ):
        poses_bounds = np.load(root / subpath / "poses_bounds.npy")  # [N, 17]
        bounds = poses_bounds[:, 15:]  # [N, 2]
        poses = poses_bounds[:, :15].reshape([-1, 3, 5])  # [N, 3, 5]

        # Scale focal
        Hpose = poses[0, 0, 4]
        focal = poses[0, 2, 4] * (image_hw[0] / Hpose)

        # Correct poses
        poses = np.concatenate(
            [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1
        )  # (N, 3, 4)

        # Clip bounds
        bounds = self.clip_bounds(bounds, near_p, far_p, near_min, far_max)
        super().__init__(
            root,
            image_hw,
            contraction,
            focal,
            poses,
            bounds,
            compute_ndc_aabb,
            scene_scale,
            process_poses,
        )

    @staticmethod
    def clip_bounds(
        bounds: np.ndarray, near_p: float, far_p: float, near_min: float, far_max: float
    ) -> np.ndarray:
        # Limit range of bounds to specified percentiles of values within range
        near_bounds = bounds[:, 0]
        far_bounds = bounds[:, 1]
        log_stats(logger.info, "near bounds", near_bounds)
        log_stats(logger.info, "far bounds", far_bounds)

        near_bounds_in_range = near_bounds[near_bounds > near_min]
        far_bounds_in_range = far_bounds[far_bounds < far_max]
        log_stats(logger.info, "near bounds (in range)", near_bounds_in_range)
        log_stats(logger.info, "far bounds (in range)", far_bounds_in_range)

        bounds[:, 0] = np.clip(
            bounds[:, 0], a_min=np.percentile(near_bounds_in_range, near_p), a_max=None
        )
        bounds[:, 1] = np.clip(
            bounds[:, 1], a_min=None, a_max=np.percentile(far_bounds_in_range, far_p)
        )

        return bounds
