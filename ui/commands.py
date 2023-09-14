# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from os.path import isdir, isfile
from pathlib import Path
from typing import Any, Dict, List
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm

from ui.data_manager import DataManager
from ui.data_model import Dataset
from utils.image_utils import read_image, read_image_np, save_image_np
from utils.io_utils import filter_dirs, multi_glob_sorted

all_cli_commands: Dict[str, Any] = {}
key_re = re.compile(r"^([a-zA-Z\d_-]+)/([a-zA-Z\d_-]+)$")
logger = logging.getLogger(__file__)


def cli_command(func):
    all_cli_commands[func.__name__] = func
    return func


def run(args: List[Any], cwd=None, capture_output=False):
    args = [str(v) for v in args]
    logger.info("\n======Run======\n" + " ".join(args) + "\n===============")
    return subprocess.run(args, cwd=cwd, capture_output=capture_output)


def print_dataset(dm: DataManager, key: str):
    ds = dm.datasets[key]
    exps = dm.experiments.get(key, [])
    print(key)
    print("  local:", ds.local)
    print(" remote:", ds.remote)
    print("   exps:")
    for exp in exps:
        print(f"    {exp.method}/{exp.name}")
        print(f"       local:", exp.local)
        print(f"      remote:", exp.remote)
        print(f"       notes:", exp.notes)


def validate_key(key: str):
    m = key_re.match(key)
    if m is None:
        raise ValueError(
            f"Invalid key format: {key}, must be [category]/[video], "
            "names may contain letters, numbers, dash, underscore."
        )
    return m.group(0), m.group(1), m.group(2)


@cli_command
def check_key(key: str):
    print(validate_key(key))


@cli_command
def list_data(dm: DataManager):
    for key in sorted(dm.datasets.keys()):
        print(key)


@cli_command
def print_data(dm: DataManager, key: str):
    if key not in dm.datasets:
        print(f"Dataset ({key}) does not exist.")
        return
    print_dataset(dm, key)


@cli_command
def run_colmap(
    dm: DataManager,
    key: str,
    mask: str = "mask",
    bg_erosion: int = 10,
    scale: int = 1,
    overwrite: bool = False,
):
    ds = dm.datasets[key]
    if "colmap" in ds.local.poses and not overwrite:
        logger.info(f"colmap poses already exist for {key}")
        return
    if not ds.local.images:
        raise ValueError("no images available")

    ds_root = dm.get_local_data_path(key)
    rgb_dir = ds_root / "rgb_1x"
    colmap_dir = ds_root / f"run_colmap/{scale}x"
    colmap_working_dir = colmap_dir / "working"
    colmap_mask_folder = colmap_dir / "mask"

    colmap_working_dir.mkdir(parents=True, exist_ok=True)

    def copy_output():
        if not isfile(colmap_working_dir / "poses_bounds.npy"):
            raise ValueError(f"No poses_bounds.npy file found in {colmap_working_dir}")

        pose_dir = ds_root / "colmap"
        pose_dir.mkdir(exist_ok=True)
        shutil.copy2(colmap_working_dir / "poses_bounds.npy", pose_dir / "poses_bounds.npy")
        ds.local.poses.append("colmap")

    if isfile(colmap_working_dir / "poses_bounds.npy"):
        logger.info(f"Found poses_bounds.npy from previous COLMAP runs in {colmap_working_dir}")
        copy_output()
        return

    # combine masks and use as motion mask

    if isdir(colmap_mask_folder):
        logger.info(f"COLMAP masks in {colmap_mask_folder}, using them")
    else:
        rgb_names = multi_glob_sorted(rgb_dir, ["*.png", "*.jpg"])

        mask_root = ds_root / "masks" / mask
        mask_folders = filter_dirs(sorted(mask_root.glob("*")))
        if len(mask_folders) == 0:
            raise ValueError(f"No mask folder found in {mask_folders}")

        logger.info(f"Combine masks from {mask_folders}")

        from core.data.mask import MaskDataSource
        img = read_image(next(mask_folders[0].glob("*.png")))
        data = MaskDataSource(ds_root, [img.height // scale, img.width // scale],
                              f"masks/{mask}", 0, bg_erosion, 0, False, 0, [])
        colmap_mask_folder.mkdir()
        print(len(data))
        for i in range(len(data)):
            save_image_np(
                colmap_mask_folder / f"{rgb_names[i].name}.png",  # colmap expects xxx.png.png
                data.background_mask[i].float().numpy(),
            )

    args = [
        "python",
        "preprocess/run_colmap.py",
        f"images={rgb_dir}",
        f"output={colmap_working_dir}",
        f"masks={colmap_mask_folder}",
    ]

    run(args)
    copy_output()


@cli_command
def v2i(dm: DataManager, key: str, video: str, step: int = 1, limit: int = 200, v2i_skip: str = None, dst: str = "rgb_1x", is_mask: bool = False):
    key, ds_cat, ds_name = validate_key(key)

    out_dir = dm.get_local_data_path(key) / dst
    if isdir(out_dir):
        logger.info(f"output folder already exists: {out_dir}")
        return

    args = [
        "python",
        "preprocess/video_to_images.py",
        f"input={video}",
        f"output={out_dir}",
        f"step={step}",
        f"limit={limit}",
        f"mask={is_mask}",
    ]
    if v2i_skip is not None:
        args.append(f"skip={v2i_skip}")
    run(args)

    img_files = multi_glob_sorted(out_dir, "*.png")
    if len(img_files) == 0:
        raise ValueError("No image is found")

    logger.info(f"Found {len(img_files)} files")
    if key not in dm.datasets:
        dm.datasets[key] = Dataset(ds_cat, ds_name)


@cli_command
def run_flow(dm: DataManager, key: str, keep_backward: bool = False):
    ds = dm.datasets[key]

    if not ds.local.images:
        raise ValueError("no images available")

    root = dm.get_local_data_path(key)
    rgb_dir = root / "rgb_1x"
    flow_dir = root / "flow"

    need_forward = not isdir(flow_dir / "flow")
    need_confidence = not isdir(flow_dir / "confidence")
    need_backward = (need_confidence or keep_backward) and not isdir(flow_dir / "flow_backward")
    if not need_forward and not need_backward and not need_confidence:
        logger.info(f"flow already exists for {key}")
        return

    if need_forward or need_backward:
        proc = run([
            "python",
            "preprocess/run_flow.py",
            f"input={rgb_dir}",
            f"output={flow_dir}",
            f"model={dm.local_data_root / 'pretrained/raft/raft-things.pth'}",
            f"forward={need_forward}",
            f"backward={need_backward}",
        ])
        if proc.returncode != 0:
            raise ValueError("flow script failed")

    if need_confidence:
        proc = run([
            "python",
            "third_party/omnimatte/datasets/confidence.py",
            flow_dir,
            "--rgb",
            rgb_dir,
        ])
        if proc.returncode != 0:
            raise ValueError("confidence script failed")

    if not keep_backward:
        shutil.rmtree(flow_dir / "flow_backward")


@cli_command
def run_depth(dm: DataManager, key: str):
    ds = dm.datasets[key]
    if ds.local.depth:
        logger.info(f"depth already exists for {key}")
        return
    if not ds.local.images:
        raise ValueError("no images available")

    root = dm.get_local_data_path(key)
    rgb_dir = root / "rgb_1x"
    depth_dir = root / "depth"

    proc = run([
        "python",
        "preprocess/run_depth.py",
        f"input={rgb_dir}",
        f"output={depth_dir}",
        f"model={dm.local_data_root / 'pretrained/midas/dpt_beit_large_512.pt'}",
    ])
    if proc.returncode != 0:
        raise ValueError("segmentation script failed")


@cli_command
def run_segmentation(dm: DataManager, key: str):
    ds = dm.datasets[key]
    if ds.local.segmentation:
        logger.info(f"segmentation already exists for {key}")
        return
    if not ds.local.images:
        raise ValueError("no images available")

    root = dm.get_local_data_path(key)
    rgb_dir = root / "rgb_1x"
    seg_file = root / "segmentation.zip"

    proc = run([
        "python",
        "preprocess/run_segmentation.py",
        f"input={rgb_dir}",
        f"output={seg_file}",
    ])
    if proc.returncode != 0:
        raise ValueError("segmentation script failed")


@cli_command
def run_homogrpahy(dm: DataManager, key: str, scale: int = -1):
    assert scale in {-1, 1, 2}

    ds = dm.datasets[key]
    if ds.local.homography:
        logger.info(f"homography already exists for {key}")
        return
    if not ds.local.images:
        raise ValueError("no images available")

    root = dm.get_local_data_path(key)
    rgb_dir = root / "rgb_1x"
    homography_dir = root / "homography"

    if scale == -1:
        scale = 1
        img = read_image(multi_glob_sorted(rgb_dir, ["*.png", "*.jpg"])[0])
        if img.width * img.height > 1280 * 720:
            scale = 2
        logger.info(f"Homography auto-scale: image resolution {img.width}x{img.height}, scale {scale}")

    proc = run([
        "python",
        "preprocess/run_homography.py",
        f"input={rgb_dir}",
        f"output={homography_dir}",
        f"scale={1/scale}",
    ])
    if proc.returncode != 0:
        raise ValueError("homography script failed")

    with open(homography_dir / "size.txt", "r") as f:
        sz_args = f.readline().strip().split(" ")

    proc = run([
        "python",
        "third_party/omnimatte/datasets/homography.py",
        "--homography_path",
        homography_dir / "homographies-first-frame.npy",
        *sz_args,
    ])
    if proc.returncode != 0:
        raise ValueError("homography conversion failed")


def copy_image_folder(src: Path, dst: Path):
    files = multi_glob_sorted(src, ["**/*.png", "**/*.jpg"])
    if len(files) == 0:
        raise ValueError(f"No image file found in {src}")

    logger.info(f"Copy {len(files)} images from {src} to {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(files):
        shutil.copy2(file, dst / f"{i:05d}{file.suffix}")


def copy_images(dm: DataManager, key: str, dst: str, src: str, v2i_step: int, v2i_limit: int, v2i_skip: str, is_mask: bool):
    out_dir = dm.get_local_data_path(key) / dst

    if isdir(src):
        # assume image folder
        copy_image_folder(Path(src), out_dir)
    elif isfile(src):
        if src.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                with ZipFile(src, "r") as zf:
                    zf.extractall(tmp)
                copy_image_folder(Path(tmp), out_dir)
        else:
            # assume video file
            v2i(dm, key, src, v2i_step, v2i_limit, v2i_skip, dst, is_mask)


@cli_command
def add_data(
    dm: DataManager,
    key: str,
    video: str = None,
    masks: str = None,
    mask_name: str = None,
    v2i_step: int = 1,
    v2i_limit: int = 200,
    v2i_skip: str = None,
    run_extra: bool = True,
):
    key, ds_cat, ds_name = validate_key(key)
    if key not in dm.datasets:
        dm.datasets[key] = Dataset(ds_cat, ds_name)

    avail = dm.datasets[key].local

    if not avail.images:
        assert video is not None, "Images not found, provide with --video, or manually copy from elsewhere"
        copy_images(dm, key, "rgb_1x", video, v2i_step, v2i_limit, v2i_skip, False)
    elif video is not None:
        logger.warning("Image folder exists, provided video is ignored")

    if masks is None:
        masks = []
    else:
        masks = masks.split(",")
    if len(masks) > 0:
        mask_name = mask_name or "mask"
        if mask_name not in avail.masks:
            for i, src in enumerate(masks):
                copy_images(dm, key, f"masks/{mask_name}/{i:02d}", src, v2i_step, v2i_limit, v2i_skip, True)
        else:
            logger.warning(f"Mask folder ({mask_name}) exists, provided masks are ignored")

    dm.load_local_dataset(dm.get_local_data_path(key))

    if run_extra:
        run_flow(dm, key)
        run_segmentation(dm, key)
        run_homogrpahy(dm, key)


@cli_command
def run_rodynrf(
    dm: DataManager,
    key: str,
    n_steps: int = 30000,
    mask: str = "mask",
    bg_erosion: int = 20,
    config: str = None,
    fov_init: float = None,
    davis_mode: bool = False,
    overwrite: bool = False,
    extra_args: List[str] = [],
):
    is_davis = davis_mode or "davis/" in key
    if config is None:
        config = "DAVIS_CAM" if is_davis else "REALWORLD_CAM"
        logger.info(f"Automatically set config to {config}")

    if fov_init is None:
        fov_init = 30 if is_davis else 72
        logger.info(f"Automatically set fov_init to {fov_init}")

    assert config in {
        "DAVIS_CAM",
        "REALWORLD_CAM",
        "REALWORLD_CAM_NDC",
    }

    ds = dm.datasets[key]
    avail = ds.local
    if "rodynrf" in avail.poses and not overwrite:
        logger.info(f"rodynrf already exists for {key}")
        return

    if not avail.images:
        raise ValueError("no image available")
    if mask not in avail.masks:
        raise ValueError(f"masks({mask}) not found")

    code_root = Path(__file__).parent.parent / "third_party/RoDynRF"
    ds_root = dm.get_local_data_path(key)
    train_dir = ds_root / "train_rodynrf"
    train_dir.mkdir(exist_ok=True)

    train_out_dir = train_dir / ("train-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # combine masks and use as motion mask
    mask_root = ds_root / "masks" / mask
    mask_folders = filter_dirs(sorted(mask_root.glob("*")))
    training_mask_folder = train_dir / "mask"

    if isdir(training_mask_folder):
        logger.info(f"Training masks for RoDynRF exist in {training_mask_folder}, using them")
    else:
        logger.info(f"Combine masks from {mask_folders}")

        from core.data.mask import MaskDataSource
        img = read_image(next(mask_folders[0].glob("*.png")))
        data = MaskDataSource(ds_root, [img.height, img.width], f"masks/{mask}", 0, bg_erosion, 0, False, 0, [])
        training_mask_folder.mkdir()
        for i in range(len(data)):
            save_image_np(
                training_mask_folder / f"{i:05d}.png",
                (~data.background_mask[i]).float().numpy(),
            )

    # compute depths
    training_depths_folder = train_dir / "disp"
    if isdir(training_depths_folder):
        logger.info(f"Training depths for RoDynRF exist in {training_depths_folder}, using them")
    else:
        run([
            "python",
            "scripts/generate_depth.py",
            "--dataset_path",
            ds_root,
            "--model",
            dm.local_data_root / "pretrained/midas/midas_v21-f6b98070.pt",
        ], cwd=code_root)

    # compute flow
    training_flow_folder = train_dir / "flow"
    if isdir(training_flow_folder):
        logger.info(f"Training flow for RoDynRF exist in {training_flow_folder}, using them")
    else:
        run([
            "python",
            "scripts/generate_flow.py",
            "--dataset_path",
            ds_root,
            "--model",
            dm.local_data_root / "pretrained/raft/raft-things.pth",
        ], cwd=code_root)

    proc = run([
        "python",
        "train.py",
        "--config",
        f"configs/{config}.txt",
        "--expname",
        train_out_dir.name,
        "--basedir",
        train_dir,
        "--datadir",
        ds_root,
        "--use_foreground_mask",
        training_mask_folder,
        "--n_iters",
        n_steps,
        "--fov_init",
        fov_init,
        "--vis_train_sancheck",
        *extra_args,
    ], cwd=code_root)

    if proc.returncode != 0:
        raise ValueError("RoDynRF training script failed")

    (ds_root / "rodynrf").mkdir(exist_ok=True)
    shutil.copy2(train_out_dir / f"render_test/{n_steps}/poses_bounds_ours.npy", ds_root / "rodynrf/poses_bounds.npy")


@cli_command
def list_rodynrf(
    dm: DataManager,
    key: str,
):
    """List existing RoDynRF trainings"""
    ds_root = dm.get_local_data_path(key)
    train_dir = ds_root / "train_rodynrf"
    if not isdir(train_dir):
        logger.error(f"RoDynRF training folder does not exist: {train_dir}")

    train_out_dirs = sorted(train_dir.glob("train-*"))
    logger.info(f"RoDynRF training in {train_dir}:\n" + "\n".join([d.name for d in train_out_dirs]))


@cli_command
def test_rodynrf(
    dm: DataManager,
    key: str,
    name: str = None,
    step: int = None,
):
    """Run RoDynRF render_only"""
    code_root = Path(__file__).parent.parent / "third_party/RoDynRF"
    ds_root = dm.get_local_data_path(key)
    train_dir = ds_root / "train_rodynrf"
    training_mask_folder = train_dir / "mask"

    if name is None:
        train_out_dirs = sorted(train_dir.glob("train-*"))
        if len(train_out_dirs) == 0:
            raise ValueError("No RoDynRF training is found")
        train_out_dir = train_out_dirs[-1]
    else:
        train_out_dir = train_dir / name
        if not isdir(train_out_dir):
            raise ValueError(f"Specified training not found: {train_out_dir}")

    ckpt_dir = train_out_dir / "checkpoints"
    ckpt_steps = [int(f.stem) for f in multi_glob_sorted(ckpt_dir, "*.th") if "_static" not in f.name]
    if len(ckpt_steps) == 0:
        raise ValueError(f"No checkpoint is found in {ckpt_dir}")
    if step is None:
        step = ckpt_steps[-1]
    elif step not in ckpt_steps:
        raise ValueError(f"Specified step ({step}) not found, available: {ckpt_steps}")

    run([
        "python",
        "train.py",
        "--render_only",
        "--config",
        "configs/DAVIS_CAM.txt",
        "--expname",
        train_out_dir.name,
        "--basedir",
        train_dir,
        "--datadir",
        ds_root,
        "--use_foreground_mask",
        training_mask_folder,
        "--downsample_train",
        4,
        "--ckpt",
        ckpt_dir / f"{step}.th",
    ], cwd=code_root)


@cli_command
def eval_ours(
    dm: DataManager,
    name: str,
    step: int = -1,
    overwrite: bool = False,
    method: str = "matting",
    extra_args: List[str] = [],
):
    exp = dm.experiment_names[name]
    key = f"{exp.category}/{exp.video}"

    if step < 0:
        ckpts = exp.local.checkpoints
    else:
        ckpts = [c for c in exp.local.checkpoints if c == f"checkpoint_{step}.pth"]
    if len(ckpts) == 0:
        raise ValueError("No checkpoint to run eval")

    out_root = dm.get_local_experiment_path(key, method, name.split("/")[-1])
    for ckpt in ckpts:
        ckpt_file = out_root / f"checkpoints/{ckpt}"
        step = int(ckpt_file.stem.split("_")[1])
        eval_root = out_root / f"eval/{step}"
        if isdir(eval_root) and not overwrite:
            logger.info(f"Eval output for step {step} already exists: {eval_root}")
            continue

        args = [
            "python",
            "workflows/eval.py",
            f"output={eval_root}",
            f"checkpoint={ckpt_file}",
            f"data_root={dm.local_data_root / 'matting'}",
            *extra_args,
        ]

        if method == "omnimatte":
            args += ["trainer=omnimatte"]

        proc = run(args)
        if proc.returncode != 0:
            raise ValueError("Eval script has failed")


def get_scale(scale: int, ds_root: Path):
    if scale > 0:
        return scale

    scale = 1
    img = read_image(multi_glob_sorted(ds_root / "rgb_1x", ["*.png", "*.jpg"])[0])
    img_size = min(img.width, img.height)
    if img_size >= 800:
        scale = 4
    elif img_size >= 400:
        scale = 2
    logger.info(f"Training auto-scale: image resolution {img.width}x{img.height}, scale {scale}")
    return scale


@cli_command
def train_ours(
    dm: DataManager,
    key: str,
    name: str = "",
    mask: str = "mask",
    pose: str = None,
    scale: int = -1,
    contraction: str = None,
    config: str = None,
    run_eval: bool = True,
    time_prefix: bool = True,
    use_depths: bool = False,
    eval_step: int = -1,
    extra_args: List[str] = [],
):
    assert scale in {-1, 1, 2, 4}
    if key not in dm.datasets:
        raise ValueError(f"Dataset {key} does not exist")

    ds = dm.datasets[key]
    avail = ds.local
    ds_root = dm.get_local_data_path(key)

    # check dataset
    missing = []
    if not avail.images:
        missing += ["images"]
    if not avail.flow:
        missing += ["flow"]
    if mask not in avail.masks and not isdir(mask):
        missing += [f"masks({mask})"]
    if len(avail.poses) == 0:
        missing += ["pose"]
    if use_depths and not avail.depth:
        missing += ["depth"]
    if len(missing) > 0:
        raise ValueError(f"Missing data required for training: {', '.join(missing)}")

    # determine pose
    if pose is not None:
        assert pose in {"rodynrf", "blender", "colmap"}
    else:
        for pose_name in ["rodynrf", "colmap", "blender"]:
            if pose_name in avail.poses:
                pose = pose_name
                break

        if pose is None:
            raise ValueError(f"No supported pose is found in: {avail.poses}")

    if contraction is None:
        contraction = "ndc"
        if pose == "rodynrf":
            contraction = "mipnerf"
        logger.info(f"For {pose} pose, set contraction to {contraction}")

    if config is None:
        config = "train_both_davis" if key.startswith("davis/") else "train_both"

    scale = get_scale(scale, ds_root)

    # set output name
    out_name_parts = []
    if time_prefix:
        out_name_parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    if len(name) > 0:
        out_name_parts.append(name)
    if len(out_name_parts) == 0:
        raise ValueError("If not using time prefix, must set name")

    # set output folder
    out_name = "-".join(out_name_parts)
    out_root = dm.get_local_experiment_path(key, "matting", out_name)
    if out_root.exists() and not any((s.startswith("checkpoint=") for s in extra_args)):
        raise ValueError(f"Output root {out_root} exists and no checkpoint is being loaded")

    # set required data
    if config not in {"train_tf", "train_bg"}:
        sources = f"flow,mask,{pose}"
    else:
        sources = f"mask,{pose}"
    if use_depths:
        sources += ",depths"

    if mask in avail.masks:
        mask = f"masks/{mask}"

    args = [
        "python",
        "workflows/train.py",
        "--config-name",
        config,
        f"output={out_root}",
        f"dataset.path={ds_root}",
        f"dataset.scale={1/scale}",
        f"data_sources=[{sources}]",
        f"data_sources.mask.subpath={mask}",
        f"contraction={contraction}",
        *extra_args,
    ]

    proc = run(args)
    if proc.returncode != 0:
        raise ValueError("Training script has failed")

    dm.load_local_experiment(out_root)
    if run_eval:
        eval_ours(dm, f"{key}/matting/{out_name}", step=eval_step)


@cli_command
def train_omnimatte(
    dm: DataManager,
    key: str,
    name: str = "",
    mask: str = "mask",
    scale: int = -1,
    run_eval: bool = True,
    extra_args: List[str] = [],
):
    assert scale in {-1, 1, 2, 4}
    if key not in dm.datasets:
        raise ValueError(f"Dataset {key} does not exist")

    ds = dm.datasets[key]
    avail = ds.local
    ds_root = dm.get_local_data_path(key)

    # check dataset
    missing = []
    if not avail.images:
        missing += ["images"]
    if not avail.flow:
        missing += ["flow"]
    if mask not in avail.masks:
        missing += [f"masks({mask})"]
    if not avail.homography:
        missing += ["homography"]

    if len(missing) > 0:
        raise ValueError(f"Missing data required for training: {', '.join(missing)}")

    scale = get_scale(scale, ds_root)

    out_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(name) > 0:
        out_name += f"-{name}"

    out_root = dm.get_local_experiment_path(key, "omnimatte", out_name)
    args = [
        "python",
        "workflows/train.py",
        "--config-name",
        "train_om",
        f"output={out_root}",
        f"dataset.path={ds_root}",
        f"dataset.scale={1/scale}",
        *extra_args,
    ]

    proc = run(args)
    if proc.returncode != 0:
        raise ValueError("Training script has failed")

    dm.load_local_experiment(out_root)
    if run_eval:
        eval_ours(dm, f"{key}/omnimatte/{out_name}", method="omnimatte")


@cli_command
def purge(
    dm: DataManager,
    name: str,
):
    """Purge an experiment by moving it to the purge folder"""
    exp = dm.experiment_names[name]
    key = f"{exp.category}/{exp.video}"

    src = dm.get_local_experiment_path(key, exp.method, exp.name)
    dst = dm.get_purged_experiment_path(key, exp.method, exp.name)
    # handle conflicts
    counter = 1
    while dst.exists():
        dst = dm.get_purged_experiment_path(key, exp.method, f"{exp.name}-{counter}")
        counter += 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Move {src}")
    shutil.move(src, dst)


@cli_command
def autoremove(
    dm: DataManager,
    dry_run: bool = False,
):
    """Move all experiments without a checkpoint to the failed folder"""
    exps = [e for e in dm.experiment_names.values() if len(e.local.checkpoints) == 0]

    logger.info(f"Found {len(exps)} experiment(s) without any checkpoint.")
    used_dsts = set()
    for exp in exps:
        key = f"{exp.category}/{exp.video}"
        src = dm.get_local_experiment_path(key, exp.method, exp.name)
        dst = dm.get_failed_experiment_path(key, exp.method, exp.name)

        # handle conflicts
        counter = 1
        while dst.exists() or str(dst) in used_dsts:
            dst = dm.get_failed_experiment_path(key, exp.method, f"{exp.name}-{counter}")
            counter += 1
        used_dsts.add(str(dst))

        dst.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Move {src} -> {dst}")
        if not dry_run:
            shutil.move(src, dst)


@cli_command
def list_experiments(
    dm: DataManager,
    dataset: str = "",
    method: str = "",
    prefix: str = "",
    suffix: str = "",
    missing_evals: bool = False,
):
    """List all experiments with filters"""
    result = []
    for key in [k for k in sorted(dm.datasets.keys()) if k.startswith(dataset)]:
        print(key)
        for exp in dm.experiments[key]:
            if not exp.name.startswith(prefix) or not exp.name.endswith(suffix):
                continue

            if method != "" and exp.method != method:
                continue

            ckpts = [Path(c).stem.split("_")[1] if "_" in c else c for c in exp.local.checkpoints]
            evals = exp.local.evals
            available = [c for c in ckpts if c in evals]
            missing = [c for c in ckpts if c not in evals]
            if missing_evals and len(missing) == 0:
                continue
            if missing_evals and exp.method == "lna":
                continue

            print("  " + f"{exp.method}/{exp.name}")
            print(f"     avail: {' '.join(available)}")
            print(f"   missing: {' '.join(missing)}")
            result.append(f"{key}/{exp.method}/{exp.name}")

    print()
    print("\n".join(result))


@cli_command
def trim_video(
    dm: DataManager,
    key: str,
    start: int = 0,
    end: int = 0,
    overwrite: bool = False,
):
    ds = dm.datasets[key]
    root = dm.get_local_data_path(key)
    image_dir = root / "rgb_1x"
    end = end if end > 0 else len(multi_glob_sorted(image_dir, ["*.png", "*.jpg"]))

    def copy_files_trimmed(src: Path, dst: Path):
        files = multi_glob_sorted(src, "*.*")

        files = files[start:end]
        dst.mkdir(parents=True, exist_ok=True)
        for file in files:
            if isfile(dst / file.name) and not overwrite:
                continue
            shutil.copy2(file, dst / file.name)

    out_root = root.parent / f"{ds.video}_{start}_{end}"

    logger.info("Copy images")
    copy_files_trimmed(image_dir, out_root / "rgb_1x")

    logger.info("Copy flow")
    copy_files_trimmed(root / "flow/confidence", out_root / "flow/confidence")
    copy_files_trimmed(root / "flow/flow", out_root / "flow/flow")

    for layer_dir in filter_dirs(multi_glob_sorted(root / "masks", "*/*")):
        subpath = Path("masks") / layer_dir.parent.name / layer_dir.name
        logger.info(f"Copy masks: {subpath}")
        copy_files_trimmed(layer_dir, out_root / subpath)

    for pose_folder in ["colmap", "rodynrf"]:
        if isdir(root / pose_folder):
            (out_root / pose_folder).mkdir(exist_ok=True)
            poses = np.load(root / pose_folder / "poses_bounds.npy")[start:end]
            logger.info(f"Save {pose_folder} poses ({poses.shape})")
            np.save(out_root / pose_folder / "poses_bounds.npy", poses)


@cli_command
def compare_exp(
    dm: DataManager,
    exp1_name: str,
    exp2_name: str,
    image_names: list[str] = ["detailed_layer_0", "bg_layer"],
):
    exp1 = dm.experiment_names[exp1_name]
    exp2 = dm.experiment_names[exp2_name]

    assert exp1.category == exp2.category and exp1.video == exp2.video
    step = 10000 if exp1.category == "davis" else 15000

    for image_name in image_names:
        print(image_name)

        key = f"{exp1.category}/{exp1.video}"
        folder1 = dm.get_local_experiment_path(key, "matting", exp1.name) / f"eval/{step}/{image_name}"
        folder2 = dm.get_local_experiment_path(key, "matting", exp2.name) / f"eval/{step}/{image_name}"

        out_folder = Path("/output/matting/compare_exp") / f"{key}/{exp1.name},{exp2.name},{step}/{image_name}"
        out_folder.mkdir(parents=True, exist_ok=True)

        args = [
            "python",
            "tools/make_video.py",
            "-o",
            f"{out_folder}.mp4",
            "--image_folder",
            out_folder,
            "-c",
            3,
            "--add_diff",
            folder1,
            folder2,
        ]

        run(args)


@cli_command
def s1_mask(
    dm: DataManager,
    exp_name: str,
    step: str = None,
    threshold: float = 0.5,
):
    exp = dm.experiment_names[exp_name]
    root = dm.get_local_experiment_path(f"{exp.category}/{exp.video}", exp.method, exp.name)
    mask_color = np.array([[0.5, 0.8, 0.5]], dtype=np.float32)

    if step is None:
        step = exp.local.evals[-1]
    folder = root / "eval" / step
    assert folder.exists(), str(folder)

    alpha_folder = folder / "fg_alpha"
    if not alpha_folder.exists():
        raise ValueError("alpha folder not found!")

    mask_folder = folder / "s1_mask"
    mask_folder.mkdir(parents=True, exist_ok=True)

    vis_folder = folder / "s1_visualization"
    vis_folder.mkdir(parents=True, exist_ok=True)

    for file in tqdm(multi_glob_sorted(alpha_folder, "*.png")):
        mask = read_image_np(file)
        mask = (mask >= threshold).astype(np.float32)
        save_image_np(mask_folder / file.name, mask)

        img = read_image_np(folder / "input_rgb" / file.name)
        mask = mask > 0.5
        img[mask] = img[mask] * (1 - mask_color) + mask_color
        save_image_np(vis_folder / file.name.replace(".png", ".jpg"), img)
    run([
        "python",
        "tools/simple_video.py",
        str(vis_folder),
        "--output",
        str(vis_folder.parent / f"{vis_folder.name}.mp4"),
    ], capture_output=True)
