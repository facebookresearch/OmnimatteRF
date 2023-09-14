# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.image_utils import checkerboard, read_image_np, save_image_np


def load_image(path, scale=1):
    img = read_image_np(path, scale)

    # convert alpha image to RGBA
    if len(img.shape) == 2:
        img = np.concatenate([
            np.ones([*img.shape, 3], img.dtype),
            img.reshape([*img.shape, 1]),
        ], axis=-1)

    return img


parser = ArgumentParser()
parser.add_argument("-o", "--output")
parser.add_argument("-c", "--columns", type=int, default=1)
parser.add_argument("--ref", type=int, default=0)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--image_folder", default=None)
parser.add_argument("--image_format", default=None)
parser.add_argument("--add_diff", action="store_true")
parser.add_argument("--gap", type=int, default=0)
parser.add_argument("--no_check_count", dest="check_count", action="store_false")
parser.add_argument("input", nargs="+")
args = parser.parse_args()

all_input = args.input
all_files = []
for in_dir in all_input:
    in_dir = Path(in_dir)
    all_files.append(sorted(
        list(in_dir.glob("*.png")) +
        list(in_dir.glob("*.jpg")) +
        list(in_dir.glob("*.webp"))
    ))

n_videos = len(all_input)
n_frames = len(all_files[args.ref])
for i in range(n_videos):
    if len(all_files[i]) == 0:
        raise ValueError(f"No file is found in {all_files[0]}")

    if len(all_files[i]) < n_frames:
        if args.check_count:
            raise ValueError(
                f"{all_input[i]} has {len(all_files[i])} files, less than {n_frames}")
        else:
            n_frames = len(all_files[i])


scales = []
ref_img = load_image(all_files[args.ref][0], args.scale)
H, W = ref_img.shape[:2]
for i in range(n_videos):
    img = load_image(all_files[i][0])
    scales.append(H / img.shape[0])

print(f"Scales: {scales}")

output = args.output
if output is None:
    output = Path(all_input[0]).name + ".mp4"
else:
    assert output.endswith(".mp4")
output = Path(output)
output.parent.mkdir(parents=True, exist_ok=True)

image_format = args.image_format or "png"
if args.image_folder is None:
    tmp_folder = output.parent / "tmp_video_folder"
else:
    tmp_folder = Path(args.image_folder)

if os.path.isdir(tmp_folder):
    shutil.rmtree(tmp_folder)
tmp_folder.mkdir(parents=True)

if args.add_diff:
    assert n_videos == 2
    n_videos += 1

n_cols = args.columns
n_rows = n_videos // n_cols + int(n_videos % n_cols > 0)
gap = args.gap

start = args.start
end = args.end
if end <= 0:
    end = n_frames

for i_frame in tqdm(range(start, end)):
    result = np.zeros([H * n_rows + (n_rows - 1) * gap, W * n_cols + (n_cols - 1) * gap, 3])
    if args.add_diff:
        f_images = []

    for i_video in range(n_videos):
        y = (i_video // n_cols) * (H + gap)
        x = (i_video % n_cols) * (W + gap)

        if args.add_diff and i_video == n_videos - 1:
            img = np.power(np.abs(f_images[0] - f_images[1]), 0.5)
        else:
            img = load_image(all_files[i_video][i_frame], scales[i_video])

            if args.add_diff:
                f_images.append(img)

        if img.shape[-1] == 4:
            bg = checkerboard(H, W)
            alpha = img[..., 3:]
            img = img[..., :3] * alpha + bg * (1 - alpha)

        result[y:y+H, x:x+W] = img
    save_image_np(tmp_folder / f"{i_frame-start:05d}.{image_format}", result)

ffmpeg_args = [
    "ffmpeg",
    "-y",
    "-r",
    args.fps,
    "-i",
    tmp_folder / f"%05d.{image_format}",
    "-vf",
    "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
    "-crf",
    17,
    "-preset",
    "veryslow",
    "-r",
    30,
    output
]
ffmpeg_args = [str(v) for v in ffmpeg_args]

print(" ".join(ffmpeg_args))
subprocess.call(ffmpeg_args)

if args.image_folder is None:
    shutil.rmtree(tmp_folder)
