# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from core.data import DataSource, register_data_source
from utils.image_utils import read_image_np
from utils.io_utils import multi_glob_sorted

logger = logging.getLogger(__name__)


@register_data_source("mask")
class MaskDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: Tuple[int, int],
        subpath: str,
        trimap_width: int,
        bg_mask_erode_width: int,
        n_images: int,
        blank_layer_only: bool,
        extra_layers: int,
        indices: List[int],
    ):
        super().__init__(root, image_hw)

        if blank_layer_only:
            logger.info("Using only a blank layer as mask")
            self.masks = torch.zeros([n_images, 1, *image_hw], dtype=torch.bool)
            self.background_mask = torch.ones([n_images, *image_hw], dtype=torch.bool)
            self.trimasks = -1 * torch.ones([n_images, 1, *image_hw])
            self.extra_layers = 0
            return

        self.masks, self.background_mask = self.load_masks(
            root / subpath, bg_mask_erode_width, indices, extra_layers
        )
        self.extra_layers = extra_layers
        self.trimasks = self.create_trimasks(trimap_width)
        logger.info("\n".join([
            "Shapes:",
            f"masks: {self.masks.shape}",
            f"background_mask: {self.background_mask.shape}",
            f"trimasks: {self.trimasks.shape}",
        ]))

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "masks": self.masks[idx],
            "trimasks": self.trimasks[idx],
            "background_mask": self.background_mask[idx],
        }

    def get_keys(self) -> List[str]:
        return ["masks", "background_mask", "trimasks"]

    def get_global_data(self):
        return {
            "mask_extra_layer": self.extra_layers,
        }

    def load_masks(self, path: Path, bg_mask_erode_width: int, layer_indices: List[int], extra_layers: int) -> Tensor:
        path_content = sorted(os.listdir(path))
        path_content = [p for p in path_content if p[0] != "."]

        if all([os.path.isdir(path / d) for d in path_content]):
            logger.info("Mask subpath has folders of layer masks")
            dir_names = path_content
        elif all([f.endswith(".png") for f in path_content]):
            logger.info("Mask subpath has mask files (single layer)")
            dir_names = [""]
        else:
            logger.info(f"Mask subpath content: {path_content}")
            raise ValueError("Cannot determine path content type")

        if len(layer_indices) > 0:
            dir_names = [dir_names[i] for i in layer_indices]

        objects: List[List[Tensor]] = []  # [n_frames, n_objects, H, W]
        backgrounds: List[Tensor] = []
        n_masks = 0
        empty_mask_template = None
        for dir_name in dir_names + [None] * extra_layers:
            if dir_name is not None:
                mask_dir = path / dir_name
                if not os.path.isdir(mask_dir):
                    continue
                files = multi_glob_sorted(mask_dir, ["*.png"])
                n_masks = len(files)
            else:
                assert n_masks > 0, "Must have at least one real mask"
                mask_dir = None
                files = [None] * n_masks

            for i, file in enumerate(files):
                if file is not None:
                    mask = torch.from_numpy(read_image_np(file))
                    mask, _ = self._scale_to_image_size(
                        "masks" if i == 0 else None, mask[None, None]
                    )
                    mask = mask[0, 0]

                    # conver to binary (boolean) mask
                    mask = mask > 0.5
                    empty_mask_template = torch.zeros_like(mask)
                else:
                    assert empty_mask_template is not None
                    mask = empty_mask_template

                if len(backgrounds) <= i:
                    backgrounds.append(torch.ones_like(mask))
                if len(objects) <= i:
                    objects.append([])

                bg = backgrounds[i]

                # occluded by previous layers
                mask = mask & bg
                objects[i].append(mask)

                # cumulated occlusion
                backgrounds[i] = bg & ~mask

        if bg_mask_erode_width > 0:
            backgrounds = [
                self.erode(mask.float(), bg_mask_erode_width) > 0
                for mask in backgrounds
            ]

        return (
            torch.stack([torch.stack(lst, dim=0) for lst in objects], dim=0),
            torch.stack(backgrounds, dim=0),
        )

    def create_trimasks(self, trimap_width: int) -> Tensor:
        masks = self.masks
        N, L, H, W = masks.shape
        masks = masks.view(N * L, H, W)

        fg_masks = (masks > 0).float()
        bg_masks = (masks <= 0).float()

        if trimap_width > 0:
            bg_masks = torch.stack(
                [self.erode(mask[0], trimap_width)
                 for mask in torch.split(bg_masks, 1)]
            )
        masks = fg_masks - bg_masks

        return masks.view(N, L, H, W)

    @staticmethod
    def erode(t: Tensor, width) -> Tensor:
        """Erode an [H, W] tensor"""
        return torch.from_numpy(
            cv2.erode(
                t.numpy(),
                kernel=np.ones((width, width)),
                iterations=1,
            )
        )
