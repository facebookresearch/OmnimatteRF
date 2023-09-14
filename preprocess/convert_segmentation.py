# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from PIL import Image
from tqdm import tqdm

from utils.io_utils import mkdir

logger = logging.getLogger(__name__)


@dataclass
class ConvertSegmentationConfig:
    instances: str
    output: str
    indices: List[int]
    width: int
    height: int


ConfigStore.instance().store(name="convert_segmentation_schema", node=ConvertSegmentationConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="convert_segmentation")
def main(cfg: ConvertSegmentationConfig):
    in_root = Path(cfg.instances)
    files = sorted(os.listdir(in_root))
    out_root = mkdir(cfg.output)
    resize_width = cfg.width
    resize_height = cfg.height

    names = [os.path.splitext(f)[0] for f in files]

    targets = np.array(cfg.indices)
    n_targets = len(targets)
    if n_targets == 0:
        logger.error("No target box is specified")
        return

    ref_boxes = np.zeros([n_targets, 4], dtype=np.float32)
    H, W = 0, 0  # set by first frame
    for i_frame in tqdm(range(len(files))):
        boxes, masks = load_instance(in_root / files[i_frame])
        if len(masks) > 0:
            H, W = masks[0].shape

        for i_target in range(n_targets):
            target = targets[i_target]
            img = np.zeros([H, W], dtype=np.uint8)

            # for first frame, box is boxes[target],
            # for later frames, find index of maximum overlap
            if i_frame == 0:
                i_box = target
            else:
                if len(masks) > 0:
                    i_box = find_max_overlap_box_index(boxes, ref_boxes[i_target])
                else:
                    i_box = -1

            if i_box >= 0:
                ref_boxes[i_target] = boxes[i_box]
                img[masks[i_box]] = 255
            else:
                logger.warning(f"target {target} is not found in frame {i_frame}")

            img = Image.fromarray(img)

            # resize the mask image if specified
            if resize_width > 0 or resize_height > 0:
                # compute width / height with aspect ratio if not specified
                resize_width = (
                    int(np.round(img.width * resize_height / img.height)),
                    resize_width,
                )[resize_width > 0]
                resize_height = (
                    int(np.round(img.height * resize_width / img.width)),
                    resize_height,
                )[resize_height > 0]
                img = img.resize((resize_width, resize_height), Image.BILINEAR)
                img = Image.fromarray((np.array(img) >= 128).astype(np.uint8) * 255)

            img.save(
                mkdir(out_root / "mask" / f"{i_target:02d}") / f"{names[i_frame]}.png"
            )


def find_max_overlap_box_index(boxes: np.ndarray, ref_box: np.ndarray) -> int:
    """Find the index of box in boxes which has maximum overlap ratio with ref_box.
    Overlap ratio is defined as (overlapping area) / max(box area, ref_box area)
    """
    max_areas = [max(box_area(b), box_area(ref_box)) for b in boxes]
    ratios = [box_overlap(boxes[i], ref_box) / max_areas[i] for i in range(len(boxes))]

    return np.array(ratios).argmax()


def box_area(b: np.ndarray) -> float:
    """Compute area of a box"""
    x1, y1, x2, y2 = b
    return abs(x1 - x2) * abs(y1 - y2)


def box_overlap(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute overlapping area of two boxes"""
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    dx = min(max(x11, x12), max(x21, x22)) - max(min(x11, x12), min(x21, x22))
    dy = min(max(y11, y12), max(y21, y22)) - max(min(y11, y12), min(y21, y22))

    if dx > 0 and dy > 0:
        return dx * dy
    return 0


def load_instance(path: str) -> Tuple[np.ndarray, ...]:
    cp = torch.load(path, map_location="cpu")
    cp = {k: v.numpy() for k, v in cp.items()}
    return [cp[k] for k in ["pred_boxes", "pred_masks"]]


if __name__ == "__main__":
    main()
