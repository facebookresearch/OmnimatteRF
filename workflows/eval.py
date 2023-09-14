# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from core.config.eval_config import EvalConfig
from core.config.train_config import TrainConfig
from third_party.omnimatte.utils import flow_to_image
from utils.eval_utils import (EvalData, EvalFileRecord, EvalMetricRecord,
                              compute_frame_metrics)
from utils.image_utils import save_image_np, visualize_array
from utils.io_utils import mkdir
from utils.json_utils import write_data_json
from workflows.common import create_for_test

logger = logging.getLogger(__name__)

mask_colors = [
    np.array([[0.5, 0.8, 0.5]], dtype=np.float32),
    np.array([[0.8, 0.5, 0.5]], dtype=np.float32),
    np.array([[0.5, 0.5, 0.8]], dtype=np.float32),
]


def make_masked(img: np.ndarray, masks: np.ndarray):
    img = img.copy()
    for i, mask in enumerate(masks):
        color = mask_colors[i % len(mask_colors)]
        img[mask] = img[mask] * (1 - color) + color
    return img


@torch.no_grad()
def evaluate(config: EvalConfig):
    train_config_file = config.train_config_file or Path(config.checkpoint).parent.parent / "config.yaml"
    train_config: TrainConfig = OmegaConf.load(train_config_file)

    trainer, dataset, global_step, config = create_for_test(config, train_config, config.data_root)

    # dataset name: folder name
    dataset_name = config.dataset_name or Path(config.dataset.path).name
    logger.info(f"Dataset: {dataset_name}")

    # we store ckpts in //<experiment name>/checkpoints/step_xxx.ckpt
    experiment = config.experiment or Path(config.checkpoint).parent.parent.name
    logger.info(f"Experiment name: {experiment}")

    global_step = config.step or global_step
    logger.info(f"Global step: {global_step}")

    out_root = Path(config.output)

    # save eval config
    with open(out_root / "eval_config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config))

    # copy train config
    if os.path.isfile(train_config_file):
        shutil.copy2(train_config_file, out_root / "train_config.yaml")

    eval_data = EvalData(dataset_name, experiment, global_step)

    # run test and write results
    trainer.init_test()
    written_image_keys = set()
    total_count = len(dataset)
    if config.debug_count > 0:
        total_count = min(total_count, config.debug_count)
    raw_data_keys: List[str] = config.raw_data_keys
    raw_data_indices: List[int] = config.raw_data_indices
    all_bg_depths: List[np.ndarray] = []

    def write_image_tracked(key: str, result: Dict[str, np.ndarray], i_frame: int, datapoint: str):
        if key not in result:
            if i_frame == 0:
                logger.warning(f"Image key not found: {key}")
            return None
        written_image_keys.add(key)
        save_image_np(mkdir(out_root / key) / f"{datapoint}.png", result[key])
        eval_data.images.append(
            EvalFileRecord(datapoint, key, f"{key}/{datapoint}.png")
        )
        return result[key]

    for i_frame in tqdm(range(total_count)):
        frame = dataset[i_frame]
        datapoint = f"{i_frame:08d}"
        L = len(frame["masks"])

        def write_image(key: str):
            return write_image_tracked(key, result, i_frame, datapoint)

        result = trainer.test_one_image(frame)

        # store gt images with input masks
        result["input_rgb"] = frame["image"].numpy()
        result["input_mask"] = make_masked(frame["image"].numpy(), frame["masks"].numpy())
        result["background_mask"] = make_masked(frame["image"].numpy(), ~frame["background_mask"].numpy()[None])
        write_image("input_rgb")
        write_image("input_mask")
        for i in range(L):
            result[f"input_mask_{i}"] = frame["masks"][i].numpy()
            write_image(f"input_mask_{i}")

        if "bg_depths_raw" in result:
            all_bg_depths.append(result["bg_depths_raw"])

        # debug - save raw data
        if len(raw_data_indices) == 0 or i_frame in raw_data_indices:
            for key in raw_data_keys:
                if key not in result:
                    if i_frame == 0 or \
                            len(raw_data_indices) > 0 and i_frame == raw_data_indices[0]:
                        logger.warning(f"Raw data key not found: {key}")
                    continue
                np.save(mkdir(out_root / "raw_data" / key) / f"{datapoint}.npy", result[key])

        write_image("bg_layer")
        write_image("bg_layer_unscaled")
        write_image("bg_depths")
        write_image("bg_depths_abs")
        write_image("composite_rgb")
        write_image("detailed_rgb")
        write_image("composite_flow")
        write_image("bg_flow")
        write_image("background_mask")
        for i in range(L):
            write_image(f"flow_layer_{i}")
        if "flow" in frame:
            result["flow"] = flow_to_image(frame["flow"].numpy())
            write_image("flow")

        # create binary mask if available
        fg_alpha = write_image("fg_alpha")
        if fg_alpha is not None:
            result["fg_alpha_colorized"] = visualize_array(result["fg_alpha"], cv2.COLORMAP_JET)
            write_image("fg_alpha_colorized")

            if i_frame == 0:
                logger.info("Add pred_fg and pred_bg to result")

            if "pred_fg" not in result:
                result["pred_fg"] = fg_alpha > config.alpha_threshold
                write_image("pred_fg")

                result["pred_bg"] = ~result["pred_fg"]
                write_image("pred_bg")

                frame["pred_fg"] = result["pred_fg"]
                frame["pred_bg"] = result["pred_bg"]

            # create masked images
            result["combined_mask"] = make_masked(result["input_rgb"], [result["pred_fg"]])
            write_image("combined_mask")

        for i_layer in range(L):
            write_image(f"rgba_layer_{i_layer}")
            write_image(f"detailed_layer_{i_layer}")

        def write_metrics(pred_key: str, gt_key: str, mask_key: Optional[str] = None):
            metrics = compute_frame_metrics(
                result, frame, pred_key, gt_key, mask_key, log_missing=i_frame == 0
            )
            for metric_key, value in metrics.items():
                key = f"{pred_key}"
                if mask_key is not None:
                    key += f"-{mask_key}"
                key += f"-{metric_key}"

                if np.isnan(value):
                    value = "NaN"
                eval_data.metrics.append(EvalMetricRecord(datapoint, key, value))

        write_metrics("composite_rgb", "image")
        write_metrics("composite_rgb", "image", "pred_fg")
        write_metrics("composite_rgb", "image", "pred_bg")

        write_metrics("detailed_rgb", "image")
        write_metrics("detailed_rgb", "image", "pred_fg")
        write_metrics("detailed_rgb", "image", "pred_bg")

        write_metrics("bg_layer", "image", "pred_bg")

        if config.eval_bg_layer:
            write_metrics("bg_layer", "image", "background_mask")

    if len(all_bg_depths) > 0:
        all_bg_depths = np.stack(all_bg_depths, axis=0)
        near, far = np.percentile(all_bg_depths, [0.1, 99.9])
        logger.info(f"Visualize globally scaled bg_depths, near={near}, far={far}")
        for i_frame in tqdm(range(total_count)):
            bg_depths = np.clip(all_bg_depths[i_frame], near, far)
            bg_depths = (bg_depths - near) / (far - near)
            bg_depths = visualize_array(bg_depths)
            write_image_tracked("bg_depths_global", {"bg_depths_global": bg_depths}, i_frame, f"{i_frame:08d}")

    write_data_json(out_root / "eval_data.json", eval_data, EvalData)

    for key in written_image_keys:
        if not config.write_videos:
            break

        cmd = [
            "python",
            "tools/make_video.py",
            "-o",
            out_root / f"{key}.mp4",
            out_root / key,
        ]
        cmd = [str(v) for v in cmd]

        logger.info(" ".join(cmd))
        run = subprocess.run(cmd, capture_output=True)
        if run.returncode != 0:
            logger.error(run.stderr.decode("utf-8"))


@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def main(config: EvalConfig = None):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    logger.info(OmegaConf.to_yaml(config))

    evaluate(config)


if __name__ == "__main__":
    main()
