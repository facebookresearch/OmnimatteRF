# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import skimage.morphology
import torch
from PIL import Image
from tqdm import tqdm

# NOTE:
# Nvidia: step: [1], thres: 6.0
# DAVIS: step: [1, 2, 4], thres: 1.0

DEVICE = 'cuda'


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img[..., :3]).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_image(file, image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(file)


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z ** 2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(img, flow_new, None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    return res


def get_stats(X, norm=2):
    """
    :param X (N, C, H, W)
    :returns mean (1, C, 1, 1), scale (1)
    """
    mean = X.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=str, help='Dataset path')
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--epi_thres", type=float, default=64.0)
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.dataset_path)
    image_dir = data_dir / "rgb_1x"

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if len(images) == 0:
        raise ValueError("No images are found!")

    def mkdir(d):
        d.mkdir(parents=True, exist_ok=True)
        return d

    flow_dir = data_dir / "rodynrf/flow"
    if not os.path.isdir(flow_dir):
        raise ValueError("Need to run flow first!")

    our_mask_dir = mkdir(data_dir / "rodynrf/our_mask")
    epipolar_error_dir = mkdir(data_dir / "rodynrf/epipolar_error")

    img = load_image(images[0])
    H = img.shape[2]
    W = img.shape[3]

    uv = get_uv_grid(H, W, align_corners=False)
    x1 = uv.reshape(-1, 2)
    motion_mask_frames = []
    motion_mask_frames2 = []
    flow_for_bilateral = []
    epi_steps = [s for s in [1, 2, 4, 8, 16] if s <= args.max_step]
    for idx in tqdm(range(len(images))):
        motion_masks = []
        weights = []
        err_list = []
        normalized_flow = []
        this_flow = 0
        counter = 0
        for step in epi_steps:
            if idx - step >= 0:
                # backward flow and mask
                bwd_flow_path = os.path.join(flow_dir, f"{idx:03d}_bwd.npz")
                bwd_data = None
                with open(bwd_flow_path, "rb") as f:
                    bwd_data = np.load(f)
                    bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
                this_flow = np.copy(this_flow - bwd_flow)
                counter += 1
                bwd_flow = torch.from_numpy(bwd_flow)
                bwd_mask = np.float32(bwd_mask)
                bwd_mask = torch.from_numpy(bwd_mask)
                flow = torch.from_numpy(np.stack([2.0 * bwd_flow[..., 0] / (W - 1),
                                        2.0 * bwd_flow[..., 1] / (H - 1)], axis=-1))
                normalized_flow.append(flow)
                x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
                F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
                F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
                err = compute_sampson_error(x1, x2, F).reshape(H, W)
                fac = (H + W) / 2
                err = err * fac ** 2
                err_list.append(err)
                weights.append(bwd_mask.mean())

            if idx + step < len(images):
                # forward flow and mask
                fwd_flow_path = os.path.join(flow_dir, f"{idx:03d}_fwd.npz")
                fwd_data = None
                with open(fwd_flow_path, "rb") as f:
                    fwd_data = np.load(f)
                    fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
                this_flow = np.copy(this_flow + fwd_flow)
                counter += 1
                fwd_flow = torch.from_numpy(fwd_flow)
                fwd_mask = np.float32(fwd_mask)
                fwd_mask = torch.from_numpy(fwd_mask)
                flow = torch.from_numpy(np.stack([2.0 * fwd_flow[..., 0] / (W - 1),
                                        2.0 * fwd_flow[..., 1] / (H - 1)], axis=-1))
                normalized_flow.append(flow)
                x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
                F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
                F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
                err = compute_sampson_error(x1, x2, F).reshape(H, W)
                fac = (H + W) / 2
                err = err * fac ** 2
                err_list.append(err)
                weights.append(fwd_mask.mean())

        err = torch.amax(torch.stack(err_list, 0), 0)
        flow_for_bilateral.append(this_flow / counter)

        if args.save_raw:
            raw_dir = mkdir(data_dir / "rodynrf/raw_err")
            np.save(raw_dir / f"{idx:03d}.npy", err.cpu().numpy())

        thresh = torch.quantile(err, 0.8)
        err = torch.where(err <= thresh, torch.zeros_like(err), err)

        save_image(epipolar_error_dir / f"{idx:03d}.png", err/err.max())

        mask = skimage.morphology.binary_opening(
            err.numpy() > args.epi_thres, skimage.morphology.disk(1))  # 64.0 for nvidia, 1.0 for DAVIS

        mask = skimage.morphology.dilation(mask, skimage.morphology.disk(2))
        save_image(our_mask_dir / f"{idx:03d}.png", mask.astype(np.float32))
