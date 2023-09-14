# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from core.data import CameraDataSource
from core.data.blender_camera import BlenderCameraDataSource
from core.data.llff_camera import LlffCameraDataSource
from utils.image_utils import read_image, read_image_np
from utils.io_utils import multi_glob_sorted
from utils.json_utils import write_json


def get_cam_data(H, W, focal):
    return {
        "focal_length": focal,
        "principal_point": [W / 2, H / 2],
        "skew": 0,
        "pixel_aspect_ratio": 1.0,
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "image_size": [W, H],
    }


def get_scene_data(cam: CameraDataSource):
    return {
        "scale": 1,
        "center": [0, 0, 0],
        "bbox": cam.aabb.tolist(),
        "near": float(cam.bounds.min()),
        "far": float(cam.bounds.max()),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="Matting format video folder")
    parser.add_argument("output", help="Nerfies format video folder")
    parser.add_argument("-c", "--camera", help="Subpath (folder or .json file)")
    parser.add_argument("--overwrite_images", action="store_true")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    img_folder = in_root / "rgb_1x"
    img_files = multi_glob_sorted(img_folder, ["*.jpg", "*.png"])
    if len(img_files) == 0:
        raise ValueError("No image file found!")

    img = read_image_np(img_files[0])
    H, W = img.shape[:2]
    scales = [1]
    if min(H, W) > 300:
        scales += [2]
    if min(H, W) > 500:
        scales += [4]
    print(f"Target image scales: {scales}")

    # dataset info
    print(f"Save dataset.json ({len(img_files)} ids)")
    write_json(out_root / "dataset.json", {
        "count": len(img_files),
        "num_exemplars": len(img_files),
        "ids": [f.stem for f in img_files],
        "train_ids": [f.stem for f in img_files],
        "val_ids": [],
    })
    write_json(out_root / "metadata.json", {
        img_files[i].stem: {
            "warp_id": i,
            "appearance_id": i,
            "camera_id": 0,
        }
        for i in range(len(img_files))
    })

    # image files
    for scale in scales:
        print(f"Process image at scale {scale}")
        out_img_folder = out_root / "rgb" / f"{scale}x"
        out_img_folder.mkdir(parents=True, exist_ok=True)
        for file in tqdm(img_files):
            dst = out_img_folder / f"{file.stem}.png"
            if not args.overwrite_images and os.path.isfile(dst):
                continue

            if file.suffix == ".png" and scale == 1:
                shutil.copy2(file, dst)
                continue

            img = read_image(file, 1 / scale if scale > 1 else 1)
            img.save(dst, optimize=True)

    # camera poses
    print("Convert camera")
    cam_path = args.camera
    if cam_path is None:
        for folder in ["colmap_rodynrf_yl", "colmap_rodynrf", "colmap"]:
            if os.path.isfile(in_root / folder / "poses_bounds.npy"):
                print(f"Found camera: {folder}")
                cam_path = folder
                break
    if cam_path is None and os.path.isfile(in_root / "poses.json"):
        print("Found Blender camera (poses.json)")
        cam_path = "poses.json"
    if cam_path is None:
        raise ValueError("Can't find camera, please specify with -c")

    if cam_path.endswith(".json"):
        cam = BlenderCameraDataSource(in_root, [H, W], cam_path, 2, 6, "none", True)
    else:
        cam = LlffCameraDataSource(in_root, [H, W], cam_path, 0, 100, 0, 1000, "none")

    write_json(out_root / "scene.json", get_scene_data(cam))
    data = get_cam_data(H, W, cam.focal)
    for i in range(len(img_files)):
        rot = cam.poses[i, :3, :3].clone()
        rot[:3, 1:3] *= -1
        data["orientation"] = rot.T.tolist()
        data["position"] = cam.poses[i, :3, 3].tolist()
        write_json(out_root / "camera" / f"{img_files[i].stem}.json", data)


if __name__ == "__main__":
    main()
