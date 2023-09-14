# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from PIL import Image
from tqdm import tqdm

from third_party.RAFT.core.raft import RAFT
from third_party.RAFT.core.utils.frame_utils import writeFlow
from utils.io_utils import mkdir


@dataclass
class RunFlowConfig:
    model: str
    scale: float
    device: str
    input: str
    output: str
    forward: bool
    backward: bool


ConfigStore.instance().store(name="run_flow_schema", node=RunFlowConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="run_flow")
def main(cfg: RunFlowConfig=None):
    device = cfg.device
    scale = cfg.scale

    out_root = mkdir(cfg.output)
    forward_dir = mkdir(out_root / "flow")
    backward_dir = mkdir(out_root / "flow_backward")

    # scan input
    in_root = Path(cfg.input)
    files = sorted(os.listdir(in_root))
    logging.info(f"Process {len(files)} files")

    # load model
    model = create_raft_model(cfg.model, device)

    # read the first image and determine padding
    curr_image, pad_h, pad_w = load_image(in_root / files[0], scale, device)

    def save_flow(file, flow):
        """Save prediction from network with padding removed"""
        flow = flow.cpu().numpy()[0].transpose([1, 2, 0])
        H, W = flow.shape[:2]
        flow = flow[0 : H - pad_h, pad_w // 2 : (W - pad_w + pad_w // 2)]

        writeFlow(file, flow)

    for i in tqdm(range(len(files) - 1)):
        next_image, _, _ = load_image(in_root / files[i + 1], scale, device)
        name = os.path.splitext(files[i])[0] + ".flo"

        with torch.no_grad():
            if cfg.forward:
                _, forward = model(curr_image, next_image, iters=20, test_mode=True)
                save_flow(forward_dir / name, forward)

            if cfg.backward:
                _, backward = model(next_image, curr_image, iters=20, test_mode=True)
                save_flow(backward_dir / name, backward)

        curr_image = next_image


def create_raft_model(ckpt: str, device: str) -> RAFT:
    cp = torch.load(ckpt, map_location="cpu")

    # remove DataParallel prefix "module." from dictionary
    cp = {k[len("module."):]: cp[k] for k in cp}

    args = Namespace()
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    model = RAFT(args)
    model.load_state_dict(cp)
    model = model.to(device).eval()
    return model


def load_image(path: str, scale: float, device: str) -> torch.Tensor:
    """Read an image and convert to [1, 3, H, W] float tensor, keeping values in [0, 255].
    Also pad the sides of the image to multiples of 8.

    returns: image, pad_h, pad_w
    """
    img = Image.open(path)
    if scale != 1:
        img = img.resize(
            (int(np.round(scale * img.width)), int(np.round(scale * img.height))),
            Image.LANCZOS,
        )

    img = np.array(img, dtype=np.float32)[..., :3]  # drop alpha channel

    img = torch.from_numpy(img.transpose([2, 0, 1]))[None]

    H, W = img.shape[-2:]
    pH = (8 - H % 8) % 8
    pW = (8 - W % 8) % 8
    if pH != 0 or pW != 0:
        img = F.pad(img, [pW // 2, pW - pW // 2, 0, pH], mode="replicate")

    return img.to(device), pH, pW


if __name__ == "__main__":
    main()
