# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def get_coords(H: int, W: int) -> Tensor:
    """Get a [H, W, 2] tensor of coordinates"""
    return torch.stack(
        torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        ),
        dim=-1,
    )


def get_rays(coords: Tensor, K: np.ndarray, c2w: Tensor) -> Tuple[Tensor, Tensor]:
    """Get ray origins and directions for coords

    coords: [*, 2]
    K: [3, 3]
    c2w: [3, 4]
    """
    x, y = coords[..., 0], coords[..., 1]
    dirs = torch.stack(
        [(x - K[0, 2]) / K[0, 0], -(y - K[1, 2]) / K[1, 1], -torch.ones_like(x)], dim=-1
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def alpha_composite(
    alpha: Tensor, data: List[Tensor], bg_data: List[Tensor]
) -> List[Tensor]:
    """
    alpha: [B, L, S, 1] in range [0, 1]
    data: list of [B, L, S, C] tensors of foreground (front to back)
    bg_data: list of [B, S, C] tensors of background (i.e. mulplied by cumulated (1-alpha))

    return: list of [B, S, C] composited data
    """
    L = alpha.shape[1]
    weights = torch.cat(
        [torch.ones_like(alpha[:, 0:1]), torch.cumprod(1 - alpha, dim=1)], dim=1
    )  # [B, L+1, S, 1]
    fg_weights = weights[:, :L] * alpha  # [B, L, S, 1]
    bg_weights = weights[:, L]  # [B, S, 1]

    return [
        torch.sum(fg_weights * data[i], dim=1) + bg_weights * bg_data[i]
        for i in range(len(data))
    ]


def detail_transfer(
    target: ndarray,
    image: ndarray,
    rgba_layers: ndarray,
) -> ndarray:
    """
    transfer residual to foreground layers

    target: [H, W, 3]
    image: [H, W, 3]
    rgba_layers: [L, H, W, 3]

    returns: a copy of rgba_layers with details added
    """
    residual = target - image
    trans_comp = np.zeros_like(target[..., 0:1])
    rgba_detail = rgba_layers.copy()
    n_layers = rgba_detail.shape[0]
    for i in range(n_layers):
        trans_i = 1 - trans_comp
        rgba_detail[i, ..., :3] += trans_i * residual
        alpha = rgba_detail[i, ..., 3:4]
        trans_comp = alpha + (1 - alpha) * trans_comp
    rgba_detail = np.clip(rgba_detail, 0, 1)
    return rgba_detail, trans_comp[..., 0]
