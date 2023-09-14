# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from core.data import DataSource, register_data_source
from utils.io_utils import multi_glob_sorted

logger = logging.getLogger(__name__)


@register_data_source("depths")
class DepthsDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: tuple[int, int],
        subpath: str,
    ):
        super().__init__(root, image_hw)

        files = multi_glob_sorted(root / subpath, "*.npy")

        # load depths and scale them to (0, 1) per image
        # we only care about smoothness so this makes sense?
        # note that MiDaS depths are in disparity space
        depths = []
        for file in files:
            d = np.load(file)
            d = (d - d.min()) / (d.max() - d.min())
            depths.append(d)

        depths = torch.from_numpy(np.stack(depths))  # (N, H, W)

        logger.info(f"Loaded depths: {depths.shape}")
        depths, _ = self._scale_to_image_size("depths", depths[:, None])
        self.depths = depths[:, 0].view(len(depths), -1)

    def __len__(self) -> int:
        return len(self.depths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "depths": self.depths[idx]
        }

    def get_keys(self) -> list[str]:
        """Get list of data keys provided by this source"""
        return ["depths"]
