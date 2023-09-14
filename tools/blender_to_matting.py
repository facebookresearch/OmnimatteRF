# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
import shutil
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("folder", help="Folder containing Blender render outputs")

    args = parser.parse_args()
    root = Path(args.folder)

    folders = sorted(root.glob("frame_*"))
    N = len(folders)
    assert N > 0, "No frame is found!"
    print(f"Found {N} frames")

    rgb_folder = root / "rgb_1x"
    rgb_folder.mkdir(parents=True, exist_ok=True)

    # move image files
    for i in range(N):
        src = folders[i] / "camera_0000.png"
        dst = rgb_folder / f"{i:04d}.png"
        if dst.exists():
            continue
        shutil.move(src, dst)
    print("Moved all frames")

    frames = []
    # load data
    for i in range(N):
        with open(folders[i] / "camera_0000.json", "r", encoding="utf-8") as f:
            cam = json.load(f)
        fov = 2 * math.atan2(1, 2 * cam["normalized_focal_length_x"])
        frames.append({
            "file_path": f"rgb_1x/{i:04d}",
            "transform_matrix": cam["camera_to_world"],
            "world_to_camera": cam["world_to_camera"],
            "near": cam["near_clip"],
            "far": cam["far_clip"],
        })
    data = {
        "camera_angle_x": fov,
        "frames": frames,
    }
    with open(root / "poses.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote data, fov = {fov}")

    ans = input("Delete source folders? (y,N) ")
    if ans.lower() == "y":
        for i in range(N):
            shutil.rmtree(folders[i])
        print("Deleted source folders")


if __name__ == "__main__":
    main()
