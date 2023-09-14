# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from third_party.MiDaS.midas.model_loader import load_model
from utils.image_utils import read_image_np, save_image_np, visualize_array
from utils.io_utils import mkdir, multi_glob_sorted

logger = logging.getLogger(__name__)


@dataclass
class RunDepthConfig:
    type: str
    model: str
    device: str

    scale: float

    input: str
    output: str


ConfigStore.instance().store(name="run_depth_schema", node=RunDepthConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="run_depth")
def main(cfg: RunDepthConfig = None):
    device = cfg.device

    # scan input
    files = multi_glob_sorted(Path(cfg.input), ["*.png", "*.jpg"])
    if len(files) < 1:
        raise ValueError("No image to process")
    logger.info(f"Process {len(files)} files")

    # output dir
    out_root = mkdir(cfg.output)
    depth_folder = mkdir(out_root / "depth")
    vis_folder = mkdir(out_root / "visualization")

    model, transform, net_w, net_h = load_model(device, cfg.model, cfg.type, optimize=False)
    logger.info(f"net_w, net_h = {net_w}, {net_h}")

    for file in tqdm(files):
        img = read_image_np(file, cfg.scale)[..., :3]
        img = transform({"image": img})["image"]
        img = torch.from_numpy(img).to(device)[None]

        with torch.no_grad():
            prediction = model.forward(img)
        prediction = prediction[0].cpu().numpy()

        # save raw output, users should resize per use case
        np.save(depth_folder / f"{file.stem}.npy", prediction)

        # midas output is in disparity space, see
        # https://github.com/isl-org/MiDaS/issues/42#issuecomment-680801589
        vis = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        vis = visualize_array(vis, cv2.COLORMAP_TURBO)
        save_image_np(vis_folder / f"{file.stem}.png", vis)


if __name__ == "__main__":
    main()
