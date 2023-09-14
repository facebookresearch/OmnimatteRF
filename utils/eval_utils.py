# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses_json import dataclass_json
from kornia.metrics import ssim
from torch import Tensor

logger = logging.getLogger(__name__)

lpips_net = None
lpips_device = None


@dataclass_json
@dataclass
class EvalFileRecord:
    datapoint: str
    key: str
    path: str


@dataclass_json
@dataclass
class EvalMetricRecord:
    datapoint: str
    key: str
    value: Union[float, str]


@dataclass_json
@dataclass
class EvalData:
    dataset: str
    experiment_name: str
    step: int
    images: List[EvalFileRecord] = field(default_factory=list)
    videos: List[EvalFileRecord] = field(default_factory=list)
    metrics: List[EvalMetricRecord] = field(default_factory=list)


@torch.no_grad()
def compute_psnr(a: Tensor, b: Tensor, mask: Optional[Tensor]) -> float:
    """
    Compute PSNR of torch images, values in range [0, 1]
    """
    if mask is not None:
        a = a[mask]
        b = b[mask]

    mse = F.mse_loss(a, b)
    return float(10 * torch.log10(1 / mse))


def to_bchw(a: Tensor):
    return torch.permute(a, (2, 0, 1))[None]


@torch.no_grad()
def compute_ssim(a: Tensor, b: Tensor, window_size: int = 11) -> float:
    """
    Compute SSIM of torch images, values in range [0, 1]
    """
    return float(ssim(to_bchw(a), to_bchw(b), window_size).mean())


def for_lpips(a: Tensor):
    return (to_bchw(a) * 2 - 1).to(lpips_device)


@torch.no_grad()
def compute_lpips(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute SSIM of torch images, values in range [0, 1]
    """
    return float(lpips_net(for_lpips(a), for_lpips(b)))


def enable_lpips(device: str):
    global lpips_net, lpips_device
    if lpips_net is not None:
        if device != lpips_device:
            lpips_net = lpips_net.to(device)
            lpips_device = device
        return

    import lpips
    lpips_net = lpips.LPIPS(net="alex").to(device)
    lpips_device = device


def compute_metrics(
    pred: Tensor, gt: Tensor, mask: Optional[Tensor] = None, extra=False,
) -> Dict[str, float]:
    metrics = {"psnr": compute_psnr(pred, gt, mask)}
    if mask is not None or not extra:
        return metrics

    metrics["ssim"] = compute_ssim(pred, gt)
    if lpips_net is not None:
        metrics["lpips"] = compute_lpips(pred, gt)
    return metrics


def compute_frame_metrics(
    frame_pred: Dict[str, Union[np.ndarray, Tensor]],
    frame_data: Dict[str, Union[np.ndarray, Tensor]],
    pred_key: str,
    gt_key: str,
    mask_key: Optional[str] = None,
    log_missing: bool = False,
) -> Dict[str, float]:
    """Convenient function to compute metrics of a frame if all provided keys present

    frame_pred: output from MattingTrainer
    frame_data: datapoint from MattingDataset
    mask_key: if not None, compute metrics in masked region only

    returns: dict of metrics, empty dict if not all keys present
    """
    missing_keys = []
    if pred_key not in frame_pred:
        missing_keys += [f"[{pred_key}] in frame_pred"]
    if gt_key not in frame_data:
        missing_keys += [f"[{gt_key}] in frame_data"]
    if mask_key is not None and mask_key not in frame_data:
        missing_keys += [f"[{mask_key}] in frame_data"]

    if len(missing_keys) > 0:
        if log_missing:
            logger.warning("Skip metric for missing keys: " + ", ".join(missing_keys))
        return {}

    def to_tensor(tensor: Union[np.ndarray, Tensor]):
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)

        assert isinstance(tensor, Tensor)
        return tensor.detach()

    pred = to_tensor(frame_pred[pred_key])
    gt = to_tensor(frame_data[gt_key])
    mask = None if mask_key is None else to_tensor(frame_data[mask_key])

    return compute_metrics(pred, gt, mask)
