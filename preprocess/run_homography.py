# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import cv2
import hydra
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from numpy import ndarray
from tqdm import tqdm

from utils.io_utils import mkdir


@dataclass
class HomographyFinderConfig:
    # one of ("loftr")
    matching: str
    # one of ("outdoor", "indoor")
    loftr_pretrained: str
    # ransacReprojThreshold in cv2.findHomography
    ransac_threshold: float


@dataclass
class RunHomographyConfig:
    input: str
    output: str
    scale: float
    device: str
    finder: HomographyFinderConfig


ConfigStore.instance().store(name="run_homography_schema", node=RunHomographyConfig)


class HomographyFinder:
    def __init__(self, cfg: HomographyFinderConfig, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.find_matches = self.init_matches()

    def find_homography(self, i1: ndarray, i2: ndarray) -> ndarray:
        """Find homography of two images.

        i1, i2: [H, W, 3] images in BGR format (from cv2.imread)
        returns: [3, 3] homography,
        """
        p1, p2, _ = self.find_matches(i1, i2)
        M, _ = cv2.findHomography(p1, p2, cv2.RANSAC, self.cfg.ransac_threshold)
        return M

    def init_matches(self) -> Callable[[ndarray, ndarray], Tuple[ndarray, ...]]:
        matching = self.cfg.matching
        if matching == "loftr":
            return self.init_matches_loftr()
        else:
            raise ValueError(matching)

    def init_matches_loftr(self):
        """Initialize LoFTR match finder"""
        pretrained = self.cfg.loftr_pretrained
        self.loftr = KF.LoFTR(pretrained=pretrained).eval().to(self.device)
        return self.find_matches_loftr

    def find_matches_loftr(
        self, i1: ndarray, i2: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Find matches with LoFTR

        returns: p1, p2, confidence
        """
        def make_input(img: ndarray) -> torch.Tensor:
            return K.color.bgr_to_grayscale(K.image_to_tensor(img).float() / 255)[
                None
            ].to(self.device)

        with torch.no_grad():
            result = self.loftr(
                {
                    "image0": make_input(i1),
                    "image1": make_input(i2),
                }
            )

        return [
            result[k].cpu().numpy() for k in ["keypoints0", "keypoints1", "confidence"]
        ]


@hydra.main(version_base="1.2", config_path="config", config_name="run_homography")
def main(cfg: RunHomographyConfig):
    in_root = Path(cfg.input)
    files = sorted(os.listdir(in_root))
    logging.info(f"Process {len(files)} files")

    def read_image(idx: int) -> ndarray:
        img = cv2.imread(str(in_root / files[idx]))
        scale = cfg.scale
        if scale > 0:
            img = cv2.resize(
                img,
                (
                    int(np.round(scale * img.shape[1])),
                    int(np.round(scale * img.shape[0])),
                ),
                cv2.INTER_LINEAR,
            )
        return img

    finder = HomographyFinder(cfg.finder, cfg.device)

    result = np.zeros([len(files), 3, 3], dtype=np.float32)
    result[0] = np.eye(3)
    i1 = read_image(0)
    H, W = i1.shape[:2]
    for i in tqdm(range(len(files) - 1)):
        i2 = read_image(i + 1)
        result[i + 1] = finder.find_homography(i1, i2)
        i1 = i2

    out_root = mkdir(cfg.output)
    np.save(out_root / "homographies.npy", result)
    np.save(out_root / "size.npy", np.array([H, W]))
    with open(out_root / "size.txt", "w") as f:
        f.write(f"--width {W} --height {H}")

    # save homographies relative to first frame, for Omnimatte
    for i in range(len(files) - 1):
        result[i + 1] = result[i + 1] @ result[i]
    np.save(out_root / "homographies-first-frame.npy", result)


if __name__ == "__main__":
    main()
