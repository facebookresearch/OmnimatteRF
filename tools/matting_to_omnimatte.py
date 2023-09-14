# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from utils.image_utils import read_image
from utils.io_utils import multi_glob_sorted


def copy_folder(src: Path, dst: Path, glob_patterns: List[str], overwrite: bool):
    files = multi_glob_sorted(src, glob_patterns)
    if len(files) == 0:
        raise ValueError(f"No file found in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files):
        if os.path.isfile(dst) and not overwrite:
            continue
        shutil.copy2(file, dst / file.name)


def prepare_homography(src: Path, recompute: bool):
    hom_file = src / "homography/homographies-first-frame.txt"
    if os.path.isfile(hom_file) and not recompute:
        print("Use existing homography file")
        return hom_file

    npy_file = src / "homography/homographies-first-frame.npy"
    if not os.path.isfile(npy_file) or recompute:
        subprocess.run([
            "python",
            "preprocess/run_homography.py",
            f"input={src}/rgb_1x",
            f"output={src}/homography",
        ], check=True)

    with open(src / "homography/size.txt", "r") as f:
        sz_args = f.readline().split(" ")

    subprocess.run([
        "python",
        "third_party/omnimatte/datasets/homography.py",
        "--homography_path",
        str(npy_file),
        *sz_args,
    ], check=True)
    return hom_file


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="Matting format video folder")
    parser.add_argument("output", help="Omnimatte format video folder")
    parser.add_argument("--overwrite_images", action="store_true")
    parser.add_argument("--overwrite_mask", action="store_true")
    parser.add_argument("--overwrite_flow", action="store_true")
    parser.add_argument("--overwrite_confidence", action="store_true")
    parser.add_argument("--recompute_homography", action="store_true")
    parser.add_argument("--overwrite_homography", action="store_true")
    parser.add_argument("--no_resize", dest="resize", action="store_false")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    img_folder = in_root / "rgb_1x"
    img_files = multi_glob_sorted(img_folder, ["*.jpg", "*.png"])
    if len(img_files) == 0:
        raise ValueError("No image file found!")

    print("Copy images")
    out_img_folder = out_root / "rgb"
    out_img_folder.mkdir(parents=True, exist_ok=True)
    for file in tqdm(img_files):
        dst = out_img_folder / f"{file.stem}.png"
        if not args.overwrite_images and os.path.isfile(dst):
            continue

        img = read_image(file)
        if args.resize:
            img = img.resize((448, 256), resample=Image.Resampling.LANCZOS)
        img.save(dst, optimize=True)

    print("Copy masks")
    out_mask_folder = out_root / "mask"
    if os.path.isdir(out_mask_folder) and args.overwrite_mask:
        shutil.rmtree(out_mask_folder)
    if not os.path.isdir(out_mask_folder):
        shutil.copytree(in_root / "masks" / "mask", out_mask_folder)

    print("Copy flow")
    copy_folder(in_root / "flow/flow", out_root / "flow", ["*.flo"], args.overwrite_flow)

    print("Copy confidence")
    copy_folder(in_root / "flow/confidence", out_root / "confidence", ["*.png"], args.overwrite_confidence)

    print("Copy homography")
    out_hom_file = out_root / "homographies.txt"
    if not os.path.isfile(out_hom_file) or args.overwrite_homography:
        hom_file = prepare_homography(in_root, args.recompute_homography)
        shutil.copy2(hom_file, out_hom_file)


if __name__ == "__main__":
    main()
