# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from collections import defaultdict
from typing import Callable, Optional

import torch

from core.trainer import Trainer
from lib.hook import Hook
from utils.eval_utils import compute_frame_metrics
from utils.io_utils import mkdir

logger = logging.getLogger(__name__)


class ValidationHook(Hook):
    def __init__(
        self,
        name: str,
        step_size: int,
        n_frames: int = 0,
        use_upload_callback: bool = False,
        upload_callback: Optional[Callable[[str, str], None]] = None,
        write_tensorboard: bool = False,
    ):
        """Hook for running validation steps in Matting project
        We want to have multiple validation hooks with different intervals, so we use name to distinguish them

        name: name of output folder and prefix of tensorboard tag
        first_step: whether to execute when global_step equals start_step
        n_frames: if positive, limit the number of frames to render
        write_tensorboard: if True, write the FIRST image to tensorboard after executing
        """
        super().__init__(step_size)
        self.name = name
        self.n_frames = n_frames
        self.use_upload_callback = use_upload_callback
        self.upload_callback = upload_callback
        self.write_tensorboard = write_tensorboard

    def execute(self, trainer: Trainer):
        step = trainer.global_step
        out_dir = mkdir(trainer.output / self.name / f"step_{step}")

        trainer.set_model_eval()

        # pred_key -> metric_key -> List[float]
        all_metrics = defaultdict(lambda: defaultdict(list))

        def add_metric(result, frame, pred_key, gt_key):
            metrics = compute_frame_metrics(result, frame, pred_key, gt_key)
            for key, value in metrics.items():
                all_metrics[pred_key][key].append(value)

        written_to_tb = False
        for result, frame in trainer._test_full_sequence(out_dir, self.n_frames):
            add_metric(result, frame, "composite_rgb", "image")
            add_metric(result, frame, "detailed_composite_rgb", "image")

            if written_to_tb or not self.write_tensorboard:
                continue

            for key, image in result.items():
                if len(image.shape) == 2:
                    image = image[..., None]
                trainer.writer.add_image(
                    f"{self.name}_{key}",
                    torch.from_numpy(image),
                    global_step=step,
                    dataformats="HWC",
                )
            written_to_tb = True

        # write metrics to tb
        for pred_k, pred_metrics in all_metrics.items():
            for metric_k, values in pred_metrics.items():
                mean = sum(values) / len(values)
                trainer.writer.add_scalar(
                    f"{self.name}_metrics_{pred_k}_{metric_k}",
                    mean,
                    global_step=step,
                )

        if self.use_upload_callback and self.upload_callback is not None:
            self.upload_callback(self.name, out_dir)

        trainer.set_model_train()
