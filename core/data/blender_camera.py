# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
from pathlib import Path

import numpy as np

from core.data import CameraDataSource, register_data_source

logger = logging.getLogger(__name__)


@register_data_source("blender_camera")
class BlenderCameraDataSource(CameraDataSource):
    def __init__(
        self,
        root: Path,
        image_hw: tuple[int, int],
        subpath: str,
        near_default: float,
        far_default: float,
        contraction: str,
        compute_ndc_aabb: bool = False,
        scene_scale: float = 1,
        process_poses: bool = True,
    ):
        with open(Path(root / subpath), "r") as f:
            meta = json.load(f)

        # read data
        frames = meta["frames"]
        poses = []
        bounds = []
        for i in range(len(frames)):
            frame = frames[i]

            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
            bounds.append(
                [
                    frame.get("near", float(near_default)),
                    frame.get("far", float(far_default)),
                ]
            )

        poses = np.stack(poses, axis=0)
        bounds = np.array(bounds, dtype=np.float32)

        # intrinsics
        fov = float(meta["camera_angle_x"])
        focal = 0.5 * image_hw[1] / np.tan(0.5 * fov)

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
