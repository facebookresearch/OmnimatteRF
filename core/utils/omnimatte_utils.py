# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from core.model.matting_dataset import MattingDataset
from core.model.render_context import RenderContext
from utils.render_utils import alpha_composite


def render_omnimatte_fg(
    dataset: MattingDataset,
    batch: Dict[str, Tensor],
    indices: Optional[Tensor],
    fg_infer_func: Callable[[Dict[str, Tensor], RenderContext], None],
    create_ctx_func: Callable[[], RenderContext],
    all_coords: Tensor,
    global_step: int,
    is_train: bool,
) -> Dict[str, Tensor]:
    """
    Render foreground layers with Omnimatte.

    dataset: MattingDataset
    batch: batch data, requires a [B, 2, H*W] background flow (key: bg_flow)
    indices: if supplied, only get results at these pixels
    fg_infer_func: typically fg_model.forward
    create_ctx_func: returns a RenderContext with proper device
    all_coords: cached coordinate array
    global_step: current training step
    is_train: if True, include GT and next-frame rendering for supervision

    Batch data requirements:
    bg_flow, [B, 2, H*W] background flow. For Omnimatte BG it's from homography.
    bg_rgba, [B, S, 4] background RGBA, optional.
              If not supplied, it should be rendered by fg model, which is Omnimatte with noise BG and homography.
              Also, in this case, the full rendered image will be in output as bg_full.
    """
    B, L, H, W = batch["masks"].shape
    N = H * W
    output = {}
    bg_flow = batch["bg_flow"]

    # render current frame
    ctx = create_ctx_func()
    device = ctx.device
    fg_infer_func(batch, ctx)

    rgba = ctx.output["rgba_layers"].view(B, L, 4, N) * 0.5 + 0.5
    flow = ctx.output["flow_layers"].view(B, L, 2, N)
    br_scale = ctx.output["br_scale"].view(B, 1, N)

    # bg rgba
    if "bg_rgba" in batch:
        bg_is_full = False
        bg_rgba = batch["bg_rgba"]  # [B, S, 4]
        bg_rgba_sampled = bg_rgba
    else:
        bg_is_full = True
        bg_warp = ctx.output["bg_warp"]  # [B, H, W, 2]
        bg_full = ctx.output["bg_rgba"]  # [4, H, W]
        bg_rgba = F.grid_sample(
            bg_full[None].expand(B, -1, -1, -1),
            bg_warp,
            mode="bilinear",
            align_corners=True,
        )

        bg_offset = ctx.output["bg_offset"]
        bg_offset_coords = (
            all_coords[None].expand(
                B, -1, -1).view(B, H, W, 2).to(device) + bg_offset
        )
        bg_offset_coords[..., 0] = bg_offset_coords[..., 0] / (W - 1) * 2 - 1
        bg_offset_coords[..., 1] = bg_offset_coords[..., 1] / (H - 1) * 2 - 1
        bg_rgba = F.grid_sample(
            bg_rgba,
            bg_offset_coords,
            mode="bilinear",
            align_corners=True,
        )

        bg_rgba = bg_rgba * 0.5 + 0.5
        bg_rgba = bg_rgba.view(B, 4, -1)  # [B, 4, H*W]

        # save full background in [1, H, W, 3] shape
        output["bg_full"] = bg_full[:3].permute(1, 2, 0)[None] * 0.5 + 0.5

        # regularize bg_offset
        output["bg_offset"] = bg_offset
        if is_train:
            output["bg_offset_target"] = torch.zeros_like(bg_offset)

    # sample RGBA
    if indices is not None:
        S = len(indices) // B
        b_indices = torch.arange(B).repeat_interleave(S)

        rgba_sampled = rgba[b_indices, ..., indices]
        rgba_sampled = rgba_sampled.view(
            B, S, L, 4).transpose(1, 2)  # [B, L, S, 4]

        br_scale_sampled = br_scale[b_indices, ..., indices]

        if bg_is_full:
            bg_rgba_sampled = bg_rgba[b_indices, ..., indices]
            bg_rgba_sampled = bg_rgba_sampled.view(B, S, 4)  # [B, S, 4]
    else:
        S = N

        rgba_sampled = rgba.transpose(2, 3)  # [B, L, N, 4]

        br_scale_sampled = br_scale

        if bg_is_full:
            bg_rgba_sampled = bg_rgba.transpose(1, 2)

    # make everything [B, ..., C] format for easier and more consistent code
    rgba = rgba.transpose(2, 3)  # [B, L, N, 4]
    flow = flow.transpose(2, 3)  # [B, L, N, 2]
    bg_flow = bg_flow.transpose(1, 2)  # [B, N, 2]
    br_scale = br_scale.view(B, N, 1)
    br_scale_sampled = br_scale_sampled.view(B, S, 1)
    if bg_is_full:
        bg_rgba = bg_rgba.transpose(1, 2)

    # alpha composite
    composite_rgba = alpha_composite(
        rgba_sampled[..., 3:4],
        [rgba_sampled],
        [bg_rgba_sampled],
    )[0]
    composite_flow = alpha_composite(
        rgba[..., 3:4],
        [flow],
        [bg_flow.to(device)],
    )[0]

    output.update(
        {
            "composite_rgb": torch.clamp(
                composite_rgba[..., :3] * br_scale_sampled, 0, 1
            ),  # [B, S, 3]
            "composite_alpha": torch.clamp(
                composite_rgba[..., 3] * br_scale_sampled[..., 0], 0, 1
            ),  # [B, S]
            "composite_flow": composite_flow,  # [B, N, 2]
            "rgb_layers": rgba[..., :3],  # [B, L, N, 3]
            "alpha_layers": rgba[..., 3],  # [B, L, N]
            "bg_layer": bg_rgba,  # [B, S, 4]
            "flow_layers": flow,  # [B, L, N, 2]
            "br_scale": br_scale,  # [B, N, 1]
            "bg_flow": bg_flow,  # [B, N, 2]
        }
    )

    if not is_train:
        return output

    n_extra_layers = dataset.global_data["mask_extra_layer"]
    if n_extra_layers > 0:
        output.update({
            "alpha_layers_extra": rgba[:, L-n_extra_layers:, :, 3]
        })

    gt = {
        "fg_gt_mask": batch["trimasks"].view(B, L, H * W),
        "fg_gt_flow": batch["flow"].view(B, H * W, 2),
        "fg_gt_flow_confidence": batch["flow_confidence"].view(B, H * W, 1),
        "flow_mean_dist_map": batch["flow_mean_dist_map"].view(B, H * W),
    }
    if indices is None:
        gt.update(
            {
                "fg_gt_rgb": batch["image"].view(B, H * W, 3),
            }
        )
    else:
        gt.update(
            {
                "fg_gt_rgb": batch["image"]
                .view(B, -1, 3)[b_indices, indices]
                .view(B, S, 3),
            }
        )

    output.update(
        {
            "global_step": global_step,
            "br_target": torch.ones_like(br_scale),
        }
    )

    # Render next frame

    # next frame data
    nf_batch = defaultdict(list)
    for i in range(B):
        frame = dataset[int(batch["data_idx"][i]) + 1]
        for k, v in frame.items():
            nf_batch[k].append(v)
    nf_batch = {k: torch.stack(v, dim=0) for k, v in nf_batch.items()}

    nf_ctx = create_ctx_func()
    fg_infer_func(nf_batch, nf_ctx)

    nf_rgba = nf_ctx.output["rgba_layers"]  # [B*L, 4, H, W]

    # warped data
    nf_coords = all_coords[None].expand(B, -1, -1)
    nf_coords = nf_coords.view(B, 1, N, 2).to(device)
    nf_coords = nf_coords + flow  # [B, L, N, 2]
    nf_coords[..., 0] = nf_coords[..., 0] / (W - 1) * 2 - 1
    nf_coords[..., 1] = nf_coords[..., 1] / (H - 1) * 2 - 1
    warped_rgba = F.grid_sample(
        nf_rgba,
        nf_coords.view(B * L, 1, N, 2),
        align_corners=True,
    ).view(B, L, 4, N)
    warped_rgba = warped_rgba * 0.5 + 0.5

    warped_rgba = torch.transpose(warped_rgba, 2, 3)  # [B, L, N, 4]

    output.update(
        {
            "warped_alpha_layers": warped_rgba[..., 3],
        }
    )

    return {**output, **{k: v.to(device) for k, v in gt.items()}}
