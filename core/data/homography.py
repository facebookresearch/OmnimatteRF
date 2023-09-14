# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from core.data import DataSource, register_data_source


@register_data_source("homography")
class HomographyDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: Tuple[int, int],
        subpath: str,
    ):
        super().__init__(root, image_hw)
        (
            self.homography,
            self.homography_size,
            self.homography_bounds,
        ) = self.load_homography(root / subpath)

    def __len__(self) -> int:
        return len(self.homography)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {"homography": self.homography[idx]}

    def get_keys(self) -> List[str]:
        return ["homography"]

    def get_global_data(self) -> Dict[str, Any]:
        return {
            "homography": self.homography,
            "homography_size": self.homography_size,
            "homography_bounds": self.homography_bounds,
        }

    def load_homography(self, path: Path) -> Tensor:
        homographies = np.load(path / "homographies-first-frame.npy")
        H_homo, W_homo = np.load(path / "size.npy")

        with open(path / "homographies-first-frame.txt", "r") as f:
            _ = f.readline()
            bounds = f.readline()

        bounds = [float(v) for v in bounds.rstrip().split(" ")[1:]]
        assert (
            len(bounds) == 4
        ), f"Failed to parse bounds (length is not 4 but {len(bounds)})"

        return torch.from_numpy(homographies).float(), [H_homo, W_homo], bounds
