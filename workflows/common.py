# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf

from core.config.eval_config import EvalConfig
from core.config.model_config import ModelConfig
from core.config.train_config import TrainConfig
from core.config.workflow_config import WorkflowConfig
from core.data import CameraDataSource
from core.model.matting_dataset import MattingDataset
from core.module import CommonModel, build_bg_model, build_fg_model
from core.trainer import Trainer, build_trainer
from utils.dict_utils import inject_dict
from utils.io_utils import mkdir

logger = logging.getLogger(__name__)


def build_model(
    builder,
    config: ModelConfig,
    injection: Dict[str, Any],
    overrides: Dict[str, Any],
):
    cfg = inject_dict(OmegaConf.to_container(config.config), injection)
    cfg.update(overrides)

    logger.info(f"Build model {config.name}, params: {cfg}")
    model = builder(config.name, cfg)
    if not config.train:
        model.requires_grad_(False)
        model.eval()
    return model


def create_dataset_and_model(
    config: WorkflowConfig,
    model_overrides: Dict[str, Any],
    fg_model_injection: Dict[str, Any],
    bg_model_injection: Dict[str, Any],
    sources_injection: Dict[str, Any],
) -> Tuple[MattingDataset, CommonModel, CommonModel]:
    # dataset
    dataset = MattingDataset(
        **OmegaConf.to_container(config.dataset),
        source_configs=OmegaConf.to_container(config.data_sources),
        sources_injection=sources_injection,
    )

    logger.info(
        f"Created dataset: {len(dataset.images)} images, {len(dataset)} datapoints"
    )

    # "dependency injection"

    common_injection = {
        "device": config.device,
        **dataset.global_data,
    }

    camera: CameraDataSource = dataset.global_data.get("camera")
    if camera is not None:
        common_injection.update({
            "near": float(camera.bounds.min()),
            "far": float(camera.bounds.max()),
            "hwf": [*camera.image_hw, camera.focal],
            "aabb": camera.aabb,
        })

    fg_model_injection.update(common_injection)
    bg_model_injection.update(common_injection)

    # model
    fg_model = build_model(
        build_fg_model,
        config.fg_model,
        fg_model_injection,
        model_overrides.get("fg") or {},
    )
    logger.info(f"FG model: {fg_model}")

    bg_model = build_model(
        build_bg_model,
        config.bg_model,
        bg_model_injection,
        model_overrides.get("bg") or {},
    )
    logger.info(f"BG model: {bg_model}")

    return dataset, fg_model, bg_model


def merge_config(config: EvalConfig, train_config: TrainConfig, data_root: str) -> EvalConfig:
    config.data_sources = OmegaConf.merge(config.data_sources, train_config.data_sources)

    config = OmegaConf.to_container(config)
    train_config = OmegaConf.to_container(train_config)

    for key in ["trainer", "fg_model", "bg_model"]:
        config[key] = train_config[key]
    inject_dict(config["dataset"], train_config["dataset"])

    # convert dataset path to local one, assuming format root/[category]/[name]
    ds_path = config["dataset"]["path"]
    ds_path = Path(ds_path)
    ds_path = Path(data_root) / ds_path.parent.name / ds_path.name
    config["dataset"]["path"] = str(ds_path)
    print(f"Dataset path: {ds_path}")

    sources = config["data_sources"]
    missing_sources = [k for k in sources if k not in train_config["data_sources"]]
    for key in missing_sources:
        sources.pop(key)

    # migration for old experiments
    bg_config = config["bg_model"]["config"]
    if "ndc" in bg_config:
        contraction = "ndc" if bg_config["ndc"] else "none"
        config["contraction"] = contraction
        bg_config["contraction"] = None
        bg_config.pop("ndc")
    else:
        assert "contraction" in train_config
        config["contraction"] = train_config["contraction"]
    if "blender_coord_system" in bg_config:
        bg_config.pop("blender_coord_system")

    fg_config = config["fg_model"]["config"]
    if "feature_cache_device" in fg_config:
        fg_config["feature_cache_device"] = config["device"]

    if "llff_camera" in sources:
        cam = sources["llff_camera"]
        if "ndc" in cam:
            cam.pop("ndc")
        if sources["llff_camera"]["subpath"].startswith("colmap_rodynrf"):
            sources["llff_camera"]["subpath"] = "rodynrf"
    if "blender_camera" in sources:
        cam = sources["blender_camera"]
        cam.pop("center_and_scale", None)
        cam.pop("use_new_scale", None)
        cam.pop("mipnerf_scale_factor", None)

    return OmegaConf.create(config)


def create_for_test(config: EvalConfig, train_config: TrainConfig, data_root: str) -> Tuple[Trainer, MattingDataset]:
    output = mkdir(config.output)
    device = config.device

    ckpt = torch.load(config.checkpoint, map_location="cpu")

    config = merge_config(config, train_config, data_root)
    logger.info("Merged config:\n" + OmegaConf.to_yaml(config))

    # dataset
    dataset, fg_model, bg_model = create_dataset_and_model(
        config,
        ckpt["model_kwargs_override"],
        fg_model_injection={
            "feature_cache_device": device,
        },
        bg_model_injection={
            "prev_global_step": 0,
            "global_step_offset": 0,
            "contraction": config.contraction,
        },
        sources_injection={
            "contraction": config.contraction,
        }
    )

    # trainer
    trainer_injection = {
        "output": str(output),
        "render_fg": config.fg_model.name != "dummy",
        "prerender_bg": False,
        "train_full_image": False,
    }
    trainer_cfg = inject_dict(
        OmegaConf.to_container(config.trainer.config), trainer_injection
    )
    trainer_cfg["use_writer"] = False
    logger.info(f"Trainer cfg: {trainer_cfg}")

    trainer: Trainer = build_trainer(config.trainer.name, trainer_cfg)
    logger.info(f"Created trainer: {trainer}")

    trainer.models = {"fg": fg_model, "bg": bg_model}
    trainer.set_dataset(dataset)
    trainer.set_device(device)

    # load checkpoint
    trainer.load_state_dict(ckpt, strict=True)

    return trainer, dataset, ckpt["global_step"], config
