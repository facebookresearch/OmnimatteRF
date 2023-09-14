# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from utils.colmap.colmap_utils import gen_poses
from utils.io_utils import mkdir, multi_glob_sorted

logger = logging.getLogger(__name__)


@dataclass
class RunColmapConfig:
    images: str
    masks: Optional[str]
    output: str
    colmap_binary: str
    colmap_options: Dict[str, Dict[str, Any]]


ConfigStore.instance().store(name="run_colmap_schema", node=RunColmapConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="run_colmap")
def main(cfg: RunColmapConfig):
    image_dir = Path(cfg.images)
    mask_dir = Path(cfg.masks) if cfg.masks else None

    image_files = multi_glob_sorted(image_dir, ["*.png", "*.jpg"])
    assert len(image_files) > 0, "No image is found!"

    colmap_cfg = OmegaConf.to_container(cfg.colmap_options)
    if mask_dir is not None:
        # check mask files before calling COLMAP
        for file in image_files:
            name = file.name
            assert os.path.isfile(mask_dir / f"{name}.png"), f"Mask image for {name} is not found!"

        colmap_cfg["feature_extractor"]["ImageReader.mask_path"] = str(mask_dir)

    out_root = mkdir(cfg.output)
    colmap_db_path = out_root / "database.db"
    colmap_out_path = out_root / "sparse"

    def run_colmap(cmd: List[str]) -> None:
        cmd = [str(v) for v in cmd]
        action = cmd[0]

        # apply additional configs
        options = colmap_cfg.get(action, {})
        for key, value in options.items():
            cmd += [f"--{key}", str(value)]

        logger.info("Run: colmap %s", " ".join(cmd))
        log_dir = mkdir(out_root / "logs")

        stdout_file = open(log_dir / f"{action}.stdout.txt", "w", encoding="utf-8")
        stderr_file = open(log_dir / f"{action}.stderr.txt", "w", encoding="utf-8")
        try:
            subprocess.run(
                [cfg.colmap_binary, *cmd], stdout=stdout_file, stderr=stderr_file, check=True
            )
        finally:
            stdout_file.close()
            stderr_file.close()

    try:
        run_colmap(
            [
                "feature_extractor",
                "--database_path",
                colmap_db_path,
                "--image_path",
                image_dir,
            ],
        )

        run_colmap(
            [
                "exhaustive_matcher",
                "--database_path",
                colmap_db_path,
            ],
        )

        colmap_out_path.mkdir(parents=True, exist_ok=True)
        run_colmap(
            [
                "mapper",
                "--database_path",
                colmap_db_path,
                "--image_path",
                image_dir,
                "--output_path",
                colmap_out_path,
            ]
        )

        gen_poses(str(out_root))

        poses_file = out_root / "poses_bounds.npy"
        poses = np.load(poses_file)
        if len(poses) < len(image_files):
            logger.error(f"Colmap only recovered {len(poses)} for {len(image_files)} images")
            poses_file.unlink()
    except subprocess.CalledProcessError:
        logger.error(f"Colmap has failed, aborting")


if __name__ == "__main__":
    main()
