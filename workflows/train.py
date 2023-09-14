# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from core.config.train_config import TrainConfig
from core.hook.checkpoint import SaveCheckpointHook
from core.hook.validation import ValidationHook
from core.loss import build_loss
from core.trainer import Trainer, build_trainer
from lib.loss import ComposedLoss
from lib.trainer import TrainerEvents
from utils.io_utils import mkdir
from workflows.common import create_dataset_and_model, inject_dict

logger = logging.getLogger(__name__)
to_dict = OmegaConf.to_container


def train(config: TrainConfig):  # noqa: C901
    output = mkdir(config.output)
    device = config.device
    ckpt_path = Path(config.checkpoint) if config.checkpoint else None

    # manual seed
    if config.seed >= 0:
        logger.info(f"Set random seed to {config.seed}")
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # save and upload experiment config
    with open(output / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config))

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_overrides = ckpt["model_kwargs_override"]

        prev_global_step = ckpt["global_step"]
        global_step_offset = 0
        if config.reset_global_step:
            ckpt["global_step"] = 0
            global_step_offset = prev_global_step
        prerender_bg_path = ckpt_path.parent / f"{ckpt_path.stem}_prerender_s{config.dataset.scale}"
    else:
        ckpt = None
        prerender_bg_path = None
        model_overrides = {}
        prev_global_step = 0
        global_step_offset = 0

    if not config.load_fg:
        model_overrides.pop("fg")
    if not config.load_bg:
        model_overrides.pop("bg")

    # dataset
    dataset, fg_model, bg_model = create_dataset_and_model(
        config,
        model_overrides,
        fg_model_injection={},
        bg_model_injection={
            "prev_global_step": prev_global_step,
            "global_step_offset": global_step_offset,
            "contraction": config.contraction,
        },
        sources_injection={
            "contraction": config.contraction,
        }
    )

    # loss
    def create_composed_loss(loss_configs: OmegaConf):
        return ComposedLoss(
            {
                k: build_loss(cfg["name"], to_dict(cfg["config"]))
                for k, cfg in loss_configs.items()
            }
        )

    criterions = {
        "fg": create_composed_loss(config.fg_losses),
        "bg": create_composed_loss(config.bg_losses),
    }
    criterions["fg"].validate()
    criterions["bg"].validate()
    logger.info(f"Created losses: {criterions}")

    # trainer
    trainer_injection = {
        "output": str(output),
        "train_fg": config.fg_model.train,
        "train_bg": config.bg_model.train,
        "render_fg": config.fg_model.name != "dummy",
        "prerender_bg_path": str(prerender_bg_path)
    }
    trainer_injection["writer_path"] = str(output / "tensorboard")

    trainer_cfg = inject_dict(
        to_dict(config.trainer.config), trainer_injection
    )
    logger.info(f"Trainer cfg: {trainer_cfg}")
    trainer = build_trainer(config.trainer.name, trainer_cfg)
    logger.info(f"Created trainer: {trainer}")

    trainer.models = {
        "fg": fg_model,
        "bg": bg_model,
    }
    trainer.optimizers = {
        "fg": fg_model.create_optimizer(to_dict(config.fg_model.optim)),
        "bg": bg_model.create_optimizer(to_dict(config.bg_model.optim))
    }
    trainer.schedulers = {
        "fg": fg_model.create_scheduler(to_dict(config.scheduler.fg), trainer.optimizers["fg"]),
        "bg": bg_model.create_scheduler(to_dict(config.scheduler.bg), trainer.optimizers["bg"]),
    }
    trainer.criterions = criterions
    trainer.set_dataset(dataset)
    trainer.set_device(device)

    if ckpt_path is not None:

        def exclude_from_ckpt(name):
            logger.info(f"Remove [{name}] from checkpoint loading")
            ckpt["models"][name] = trainer.models[name].state_dict()
            ckpt["losses"][name] = trainer.criterions[name].state_dict()
            ckpt["optimizers"][name] = trainer.optimizers[name].state_dict()
            ckpt["schedulers"][name] = trainer.schedulers[name].state_dict()

        if not config.load_fg:
            exclude_from_ckpt("fg")
        if not config.load_bg:
            exclude_from_ckpt("bg")
        if config.reset_bg_optimization:
            logger.info("Remove [bg] optimizer and scheduler states")
            ckpt["optimizers"]["bg"] = trainer.optimizers["bg"].state_dict()
            ckpt["schedulers"]["bg"] = trainer.schedulers["bg"].state_dict()

        trainer.load_state_dict(ckpt)

    # validation
    add_validation_hooks(trainer, config)

    # checkpointing
    if config.save_checkpoint.folder is None:
        ckpt_path = str(mkdir(output / "checkpoints"))
        logger.info(f"Set checkpoint folder to {ckpt_path}")
        config.save_checkpoint.folder = ckpt_path

    if config.save_checkpoint.step_size > 0:
        save_checkpoint_events = [TrainerEvents.POST_STEP]
        if config.save_pretrain_checkpoint:
            save_checkpoint_events += [TrainerEvents.PRE_TRAIN]
        if config.save_final_checkpoint:
            save_checkpoint_events += [TrainerEvents.POST_TRAIN]
        trainer.register_hook(
            save_checkpoint_events,
            SaveCheckpointHook(**to_dict(config.save_checkpoint)),
        )

    torch.autograd.set_detect_anomaly(config.debug)

    trainer.train(config.n_steps)


def add_validation_hooks(trainer: Trainer, config: TrainConfig):
    for val_config in config.validation.values():
        cfg_dict = to_dict(val_config.config)
        hook = ValidationHook(**cfg_dict)
        events = [TrainerEvents.POST_STEP]
        if val_config.pre_train:
            events += [TrainerEvents.PRE_TRAIN]

        trainer.register_hook(events, hook)


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config: TrainConfig = None):
    config = OmegaConf.create(to_dict(config, resolve=True))
    logger.info(OmegaConf.to_yaml(config))
    train(config)


if __name__ == "__main__":
    main()
