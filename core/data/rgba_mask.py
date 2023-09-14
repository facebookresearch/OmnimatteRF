# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from core.data import DataSource, register_data_source
from utils.image_utils import read_image_np
from utils.io_utils import multi_glob_sorted

logger = logging.getLogger(__name__)


@register_data_source("rgba_mask")
class RGBAMaskDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: Tuple[int, int],
        subpath: str,
    ):
        super().__init__(root, image_hw)
        self.masks = self.load_masks(root / subpath)

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "rgba_masks": self.masks[idx],
        }

    def get_keys(self) -> List[str]:
        return ["rgba_masks"]

    def load_masks(self, path: Path) -> Tensor:
        files = multi_glob_sorted(path, "*.png")
        logger.info(f"Load {len(files)} RGBA masks")

        result = []
        for file in files:
            img = torch.from_numpy(read_image_np(file))
            assert len(img.shape) == 3 and img.shape[2] == 4
            result.append(img)
        result = torch.stack(result, dim=0)

        result = torch.permute(result, (0, 3, 1, 2))  # to NCHW
        result, _ = self._scale_to_image_size("rgba_masks", result)
        result = torch.permute(result, (0, 2, 3, 1))  # to NHWC

        return result
