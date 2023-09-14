# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from lib.registry import create_registry, import_children
from third_party.TensoRF.dataLoader.llff import center_poses
from third_party.TensoRF.dataLoader.ray_utils import ndc_rays_blender

logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self, root: Path, image_hw: Tuple[int, int]):
        self.root = root
        self.image_hw = image_hw

    def __len__(self) -> int:
        return 1e10

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {}

    def get_keys(self) -> List[str]:
        """Get list of data keys provided by this source"""
        return []

    def get_global_data(self) -> Dict[str, Any]:
        """Get global (not per-frame) data from this source"""
        return {}

    def _scale_to_image_size(
        self, name: Optional[str], data: Tensor
    ) -> Tuple[Tensor, float]:
        """
        Scale a [N, C, H, W] data to image size (match H, W)
        Note: assumes that data and image has same aspect ratio; checks height only

        returns: scaled data, scale
        """
        H, W = data.shape[-2:]
        Himg, Wimg = self.image_hw

        if H == Himg:
            return data, 1

        scale = Himg / H
        data = F.interpolate(
            data, size=[Himg, Wimg], mode="bilinear", align_corners=True
        )

        if name is not None:
            logger.info(
                f"Scaled {name} from ({H}, {W}) to ({Himg}, {Wimg}), factor = {scale}"
            )

        return data, scale


class CameraDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: Tuple[int, int],
        contraction: str,
        focal: float,
        poses: np.ndarray,
        bounds: np.ndarray,
        compute_ndc_aabb: bool,
        scene_scale: float = 1,
        process_poses: bool = True,
    ):
        super().__init__(root, image_hw)
        self.focal = focal
        self.contraction = contraction

        # process poses
        if process_poses:
            logger.info("Centering poses and scaling scene")
            poses, _ = center_poses(poses[:, :3], np.eye(4))
            match contraction:
                case "ndc":
                    # correct scale so that the nearest depth is at a little more than 1.0
                    near_original = float(bounds.min())
                    scale_factor = near_original * 0.75 / scene_scale # 0.75 is the default parameter
                    # the nearest depth is at 1/0.75=1.33
                    bounds /= scale_factor
                    poses[..., 3] /= scale_factor
                case "mipnerf":
                    scale_factor = np.linalg.norm(poses[..., 3], axis=-1).max() / 0.8 / scene_scale
                    bounds /= scale_factor
                    poses[..., 3] /= scale_factor

                    poses[:, 2, 3] += (1 - poses[:, 2, 3].max())
        else:
            logger.info("Pose processing is disabled")

        self.poses = poses = torch.from_numpy(poses.astype(np.float32))
        self.bounds = bounds = torch.from_numpy(bounds.astype(np.float32))

        # precompute all rays
        logger.info(f"Compute rays of {len(poses)} views")
        self.rays_o, self.rays_d = self.compute_all_euclidean_rays()

        # compute AABB
        match contraction:
            case "ndc":
                H, W = image_hw
                self.rays_o, self.rays_d = ndc_rays_blender(H, W, focal, 1.0, self.rays_o, self.rays_d)
                ndc_bounds = bounds.clone()
                ndc_bounds[..., 0] = 0
                ndc_bounds[..., 1] = 1

                if compute_ndc_aabb:
                    self.aabb = self.compute_aabb(self.rays_o, self.rays_d, ndc_bounds)
                else:
                    self.aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]], dtype=torch.float32)
            case "mipnerf":
                self.aabb = torch.tensor([[-2, -2, -2], [2, 2, 2]], dtype=torch.float32)
            case "none":
                self.aabb = self.compute_aabb(self.rays_o, self.rays_d, bounds)

        logger.info(f"Dataset AABB ({contraction}):\n{self.aabb}")

    def __len__(self):
        return len(self.poses)

    def get_global_data(self) -> Dict[str, Any]:
        return {
            "camera": self,
        }

    def get_rays(self, idx: int, coords: Tensor) -> tuple[Tensor, Tensor]:
        x = coords[..., 0]
        y = coords[..., 1]
        return (
            self.rays_o[idx][y, x],
            self.rays_d[idx][y, x],
        )

    @staticmethod
    def compute_aabb(
            rays_o: Tensor,
            rays_d: Tensor,
            bounds: Tensor,
    ) -> Tensor:
        points = []
        N, H, W, _ = rays_o.shape

        for i, x, y in itertools.product(range(N), [0, W-1], [0, H-1]):
            ray_o = rays_o[i, y, x]
            ray_d = rays_d[i, y, x]

            points.append(ray_o + ray_d * bounds[i, 0])
            points.append(ray_o + ray_d * bounds[i, 1])
        points = torch.stack(points, dim=0)

        aabb = torch.stack(
            [
                points.min(dim=0)[0],
                points.max(dim=0)[0],
            ]
        )

        return aabb

    def compute_euclidean_rays(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get all rays of a views in world space

        Returns: [rays_o, rays_d], both (H, W, 3)
        """
        rays_d = self.compute_camera_rays(idx)
        rays_d = rays_d @ self.poses[idx, :3, :3].T

        rays_o = self.poses[idx, :3, 3].expand(rays_d.shape)
        return rays_o, rays_d

    def compute_all_euclidean_rays(self) -> tuple[Tensor, Tensor]:
        """Get all rays of all views

        Returns: [rays_o, rays_d], both (N, H, W, 3)
        """
        rays = [list(self.compute_euclidean_rays(i)) for i in range(len(self))]
        return (
            torch.stack([p[0] for p in rays]),
            torch.stack([p[1] for p in rays]),
        )

    def compute_camera_rays(self, idx: int) -> Tensor:
        """Get all camera rays (directions) in camera space

        Returns: rays_d, (H, W, 3)
        """
        H, W = self.image_hw
        focal = self.focal
        x, y = torch.stack(
            torch.meshgrid(
                torch.arange(W),
                torch.arange(H),
                indexing="xy",
            ),
            dim=0,
        )
        return torch.stack(
            [
                (x - W / 2) / focal,
                -(y - H / 2) / focal,
                -torch.ones_like(x),
            ],
            dim=-1,
        )


_, register_data_source, build_data_source = create_registry("DataSource", DataSource)
import_children(__file__, __name__)
