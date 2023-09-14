# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("input")
parser.add_argument("-o", "--output")
parser.add_argument("-s", "--scale", type=float, default=0)
args = parser.parse_args()

folder = Path(args.input)
if not os.path.isdir(folder):
    print(f"Not found: {folder}")
    exit(1)

in_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
if len(in_files) == 0:
    print("No input file exists")
    exit(1)

file = in_files[0]
prefix_idx = file.name.find("0")
prefix = "" if prefix_idx <= 0 else file.name[:prefix_idx]
start_number = int(file.stem[len(prefix):])

name_length = len(file.stem) - len(prefix)
name_ext = os.path.splitext(file.name)[1]

output = args.output
if output is None:
    output = folder.name + ".mp4"
output = Path(output)
assert output.suffix in { ".mp4", ".webm", ".mov" }

filters = ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
if args.scale > 0:
    filters = [f"scale=iw*{args.scale}:-2"] + filters
if output.suffix == ".mp4":
    filters += ["format=yuv420p"]
    encoder_args = ["-crf", 17, "-preset", "veryslow"]
elif output.suffix == ".webm":
    encoder_args = ["-c:v", "libvpx-vp9", "-crf", 17, "-b:v", 0]
elif output.suffix == ".mov":
    encoder_args = ["-c:v", "prores_ks", "-profile:v", 4, "-pix_fmt", "yuva444p10le"]

ffmpeg_args = [
    "ffmpeg",
    "-y",
    "-r",
    10,
    "-start_number",
    start_number,
    "-i",
    folder / f"{prefix}%0{name_length}d{name_ext}",
    "-vf",
    ",".join(filters),
    *encoder_args,
    "-r",
    30,
    output
]
ffmpeg_args = [str(v) for v in ffmpeg_args]

print(" ".join(ffmpeg_args))
subprocess.call(ffmpeg_args)
