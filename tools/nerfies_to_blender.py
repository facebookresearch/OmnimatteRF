# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np

from utils.json_utils import write_json


def load(path):
    with open(path, "r") as f:
        return json.load(f)


parser = ArgumentParser()
parser.add_argument("input", help="Path to nerfies video folder")
parser.add_argument("output", help="Path to matting video folder")
args = parser.parse_args()

src_root = Path(args.input)
if src_root.name == "freeze-test":
    scene = load(src_root / ".." / "scene_gt.json")
else:
    scene = load(src_root / "scene.json")

scale = scene["scale"]
near = scene["near"] / scale
far = scene["far"] / scale

src_cam_folder = src_root / "camera"
if not os.path.isdir(src_cam_folder):
    src_cam_folder = src_root / "camera-gt"

src_cam_files = sorted(src_cam_folder.glob("*.json"))
src_cams = [load(f) for f in src_cam_files]

cam = src_cams[0]
fov = 2 * math.atan2(1, 2 * cam["focal_length"] / cam["image_size"][0])
print(f"fov = {fov}")

image_files = sorted((Path(args.output) / "rgb_1x").glob("*.png"))

data = {
    "frames": [],
    "camera_angle_x": fov,
}
frames = data["frames"]


for i, cam in enumerate(src_cams):
    mat = np.eye(4)
    mat[:3, :3] = np.array(cam["orientation"]).T
    mat[:3, 1:3] *= -1
    mat[:3, 3:] = np.array(cam["position"]).reshape(3, 1)

    frames.append({
        "file_path": f"rgb_1x/{image_files[i].stem}",
        "transform_matrix": mat.tolist(),
        "near": near,
        "far": far,
    })

write_json(Path(args.output) / "poses.json", data)
