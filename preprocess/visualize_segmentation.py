# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils.io_utils import mkdir


@dataclass
class VisualizeSegmentationConfig:
    instances: str
    output: str
    rgb: str
    draw_masks: bool
    draw_mask_indices: Optional[List[int]]


ConfigStore.instance().store(name="visualize_segmentation_schema", node=VisualizeSegmentationConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="visualize_segmentation")
def main(cfg: VisualizeSegmentationConfig):
    in_root = Path(cfg.instances)
    rgb_root = Path(cfg.rgb)
    files = sorted(os.listdir(in_root))
    out_root = mkdir(cfg.output)

    mask_frames = cfg.draw_mask_indices or []

    for i_file, file in tqdm(list(enumerate(files))):
        name = os.path.splitext(file)[0]

        img = load_rgb(rgb_root, name)
        out_img = img.copy()
        draw = ImageDraw.Draw(out_img)
        w_rgb = 0.4

        boxes, masks = load_instance(in_root / file)
        for i in range(len(boxes)):
            x, y = boxes[i][:2]
            draw.rectangle(boxes[i], outline=(255, 0, 0))
            draw.text((x + 2, y + 2), f"{i}", fill=(255, 0, 0))

            if not cfg.draw_masks:
                continue
            if len(mask_frames) > 0 and i_file not in mask_frames:
                continue

            mask = masks[i]
            mask_img = np.array(img, dtype=np.float32) / 255
            mask_img[mask] = w_rgb * mask_img[mask] + (1 - w_rgb)
            mask_img = Image.fromarray((mask_img * 255).astype(np.uint8))
            mask_img.save(mkdir(out_root / "masked" / name) / f"{i}.jpg", quality=90)

        out_img.save(mkdir(out_root / "boxes") / f"{name}.jpg", quality=90)


def load_rgb(rgb_root: str, name: str) -> Image.Image:
    root = Path(rgb_root)
    for ext in [".png", ".jpg"]:
        file = root / f"{name}{ext}"
        if not os.path.isfile(file):
            continue

        return Image.open(file)

    raise RuntimeError(f"RGB file {name} not found")


def load_instance(path: str) -> Tuple[np.ndarray, ...]:
    cp = torch.load(path, map_location="cpu")
    cp = {k: v.numpy() for k, v in cp.items()}
    return [cp[k] for k in ["pred_boxes", "pred_masks"]]


if __name__ == "__main__":
    main()
