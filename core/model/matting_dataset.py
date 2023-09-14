# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.data import build_data_source
from utils.dict_utils import inject_dict
from utils.image_utils import read_image_np
from utils.io_utils import multi_glob_sorted

logger = logging.getLogger(__name__)


class MattingDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_subpath: str,
        scale: float,
        source_configs: List[Dict[str, Any]],
        sources_injection: Dict[str, Any],
    ):
        path = Path(path)
        self.images = self.load_images(
            path / image_subpath, scale)  # [N, H, W, 3]
        self.length = len(self.images)
        self.image_hw = list(self.images.shape[1:3])

        self.sources = []
        self.data_keys = set()
        self.global_data = {"n_frames": self.length, "image_hw": self.image_hw}
        for name, config in source_configs.items():
            logger.info(f"Create data source: {name}")
            config["root"] = path
            config["image_hw"] = self.image_hw
            if "n_images" in config:
                config["n_images"] = len(self.images)
            inject_dict(config, sources_injection)

            source = build_data_source(name, config)
            self.sources.append(source)
            self.length = min(self.length, len(source))
            for key in source.get_keys():
                if key in self.data_keys:
                    raise ValueError(
                        f"Duplicated data key {key} provided by data sources"
                    )
                self.data_keys.add(key)
            for key, value in source.get_global_data().items():
                if key in self.global_data:
                    raise ValueError(
                        f"Duplicated global data {key} provided by data sources"
                    )
                self.global_data[key] = value

        logger.info(f"Data keys: {', '.join(list(self.data_keys))}")
        logger.info(f"Global data: {', '.join(list(self.global_data.keys()))}")

    def __len__(self) -> int:
        return self.length - 1

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        result = {
            "image": self.images[idx],
            "data_idx": torch.tensor([idx], dtype=torch.long),
        }

        for source in self.sources:
            result.update(source[idx])

        return result

    def load_images(self, path: Path, scale: float) -> Tensor:
        files = multi_glob_sorted(path, ["*.jpg", "*.png", "*.JPG", "*.PNG"])
        assert len(files) > 0, "No image file is found"

        logger.info(f"Loading {len(files)} images")
        return torch.from_numpy(
            np.stack(
                [read_image_np(f, scale)[..., :3] for f in files],
                axis=0,
            )
        )
