# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore

from utils.io_utils import multi_glob_sorted
from utils.image_utils import read_image_np, save_image_np

logger = logging.getLogger(__name__)


@dataclass
class VideoToImagesConfig:
    input: str
    output: str
    step: int
    limit: int
    skip: Optional[str]
    mask: bool


ConfigStore.instance().store(name="video_to_images_schema", node=VideoToImagesConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="video_to_images")
def main(cfg: VideoToImagesConfig):
    out_dir = Path(cfg.output)
    if len(multi_glob_sorted(out_dir, ["*.png", "*.jpg"])) > 0:
        raise ValueError("Output folder is not empty")

    proc = subprocess.run([
        "ffprobe",
        "-v",
        "quiet",
        "-of",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        cfg.input,
    ], capture_output=True, check=True)
    data = json.loads(proc.stdout.decode("utf-8"))["streams"][0]
    width = data["width"]
    height = data["height"]
    n_frames = data["nb_frames"]

    logger.info(f"Video frame is {width}x{height}, total {n_frames} frames")
    need_scale = None
    if width > height:
        if height > 1080:
            need_scale = "h"
    else:
        if width > 1080:
            need_scale = "w"

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Extract frames with ffmpeg")

        ss = []
        if cfg.skip is not None:
            ss = ["-ss", cfg.skip]

        vf = []
        if cfg.step > 1:
            vf.append(f"select='not(mod(n\\,{cfg.step}))'")
        if need_scale == "w":
            vf.append(f"scale=w=1080:h=-1")
        elif need_scale == "h":
            vf.append(f"scale=w=-1:h=1080")

        if len(vf) > 0:
            vf = ["-vf", ",".join(vf)]

        args = [
            "ffmpeg",
            "-vsync",
            "drop",
            *ss,
            "-i",
            cfg.input,
            *vf,
            "-frames:v",
            str(cfg.limit),
            f"{tmpdir}/%05d.png"
        ]

        logger.info(" ".join(args))
        subprocess.run(args, check=True)

        tmpdir = Path(tmpdir)
        files = multi_glob_sorted(tmpdir, "*.png")
        logger.info(f"Output has {len(files)} frames")

        out_dir.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(files):
            if not cfg.mask:
                shutil.move(file, out_dir / f"{i:05d}.png")
                continue

            # convert mask image
            img = read_image_np(file)
            if len(img.shape) == 3:
                img = img[..., 0]
            save_image_np(out_dir / f"{i:05d}.png", img)


if __name__ == "__main__":
    main()
