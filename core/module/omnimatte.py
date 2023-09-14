# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from core.model.render_context import RenderContext
from core.module import CommonModel, register_fg_model
from third_party.omnimatte.data.omnimatte_dataset import get_background_uv
from third_party.omnimatte.models.networks import Omnimatte
from utils.render_utils import get_coords
from utils.torch_utils import Padder, PositionalEncoder

logger = logging.getLogger(__name__)


def check_feature_cfg(mode: str, cfg: Dict[str, Any]):
    assert mode in {"zeros", "xyt", "noise"}, f"Unknown feature_mode {mode}"

    def err(error):
        raise ValueError(f"feature_mode is {mode}, {error}")

    if mode == "zeros":
        count = sum([k in cfg for k in ["channels", "pos_n_freq"]])
        if count == 1:
            return
        if count == 0:
            err("expecting either 'channels' or 'pos_n_freq' in feature_config")
        if count == 2:
            err("both 'channels' and 'pos_n_freq' are set, please only use one")
    if mode == "xyt":
        if "pos_n_freq" in cfg:
            return
        err("expecting 'pos_n_freq' in feature_config")
    if mode == "noise":
        if "channels" in cfg:
            return
        err("expecting 'channels' in feature_config")


@register_fg_model("omnimatte")
class OmnimatteModel(CommonModel):
    """Omnimatte-style UNet"""

    def __init__(
        self,
        hidden_channels: int,
        n_frames: int,
        max_frames: int,
        coarseness: int,
        image_hw: Tuple[int, int],
        feature_mode: str,
        feature_config: Dict[str, Any],
        network_normalization: str,
        homography_size: Optional[List[float]] = None,
        homography_bounds: Optional[List[float]] = None,
        use_eval_on_network: bool = False,
        save_feature_cache_to_disk: bool = True,
        feature_cache_device: str = "cpu",
    ):
        assert n_frames <= max_frames
        check_feature_cfg(feature_mode, feature_config)

        assert network_normalization in {"none", "batch"}

        super().__init__()
        self.n_frames = n_frames
        self.max_frames = max_frames
        self.homography_size = homography_size
        self.homography_bounds = homography_bounds
        self.use_eval_on_network = use_eval_on_network
        self.save_feature_cache_to_disk = save_feature_cache_to_disk

        H, W = image_hw
        self.padder = Padder(64, H, W)

        # cache coordinates
        self.coords = get_coords(H, W)

        # cache position encoding input of every frame
        self.feature_cache, feature_channels = self.build_feature_cache(
            feature_mode, feature_config, feature_cache_device
        )
        self.feature_mode = feature_mode

        # Omnimatte network
        self.network = Omnimatte(
            nf=hidden_channels,
            in_c=feature_channels + 3,  # (mask, flow, features)
            max_frames=max_frames,
            norm=network_normalization,
            coarseness=coarseness,
        )

    def build_feature_cache(
        self,
        mode: str,
        cfg: Dict[str, Any],
        device: str,
    ) -> Tuple[Tensor, int]:
        """Pre-compute the features to append to network input

        returns tuple of:
            [n_frames, C, H, W]: features
            C: number of channels
        """
        H, W = self.coords.shape[:2]
        n_frames = self.n_frames

        if mode == "xyt":
            pos_n_freq = cfg["pos_n_freq"]
            encoder = PositionalEncoder(3, pos_n_freq, pos_n_freq - 1)
            encoded = []
            for t in tqdm(range(self.n_frames)):
                if self.save_feature_cache_to_disk:
                    t_enc = load_xyt_cache(H, W, pos_n_freq, t)
                    if t_enc is not None:
                        encoded += [t_enc]
                        continue

                t_enc = []  # list of [b_size, enc_channels] tensors
                b_size = 16384
                for b_coords in torch.split(self.coords.view(-1, 2), b_size):
                    # xy -> xyt
                    b_coords = torch.cat(
                        [b_coords, torch.ones(len(b_coords), 1) * (t + 1)],
                        dim=1,
                    )
                    t_enc += [encoder(b_coords.to(device)).cpu()]
                t_enc = torch.cat(t_enc, dim=0)
                encoded += [t_enc]
                if self.save_feature_cache_to_disk:
                    save_xyt_cache(H, W, pos_n_freq, t, t_enc)

            encoded = (
                torch.stack(encoded, dim=0).view(
                    n_frames, H, W, -1).permute(0, 3, 1, 2)
            )
            return encoded, encoded.shape[1]

        if mode == "zeros":
            if "channels" in cfg:
                channels = cfg["channels"]
            else:
                pos_n_freq = cfg["pos_n_freq"]
                encoder = PositionalEncoder(3, pos_n_freq, pos_n_freq - 1)
                channels = encoder(torch.zeros(1, 3)).shape[-1]

            return torch.zeros(n_frames, channels, H, W), channels

        if mode == "noise":
            assert (
                self.homography_bounds is not None
            ), "Homography bounds must be provided when background is noise"
            # Follow naming in Omnimatte code
            zbar = torch.randn(1, cfg["channels"], H // 16, W // 16)
            self.zbar = nn.Parameter(zbar, requires_grad=False)
            self.zbar_up = nn.Parameter(
                F.interpolate(zbar, size=(H, W), mode="bilinear",
                              align_corners=False),
                requires_grad=False,
            )

            return None, cfg["channels"]

    def forward(self, batch: Dict[str, Tensor], ctx: RenderContext) -> None:
        device = ctx.device

        frame_indices = batch["data_idx"].view(-1)
        B, L, H, W = batch["masks"].shape
        masks = batch["masks"].view(B * L, 1, H, W).float().to(device)
        flows = torch.permute(batch["flow"], (0, 3, 1, 2)).to(device)

        if self.feature_mode == "noise":
            feats = []
            all_bg_warp = []
            h_bounds = self.homography_bounds
            h_size = self.homography_size
            for i in range(B):
                h_matrix = batch["homography"][i]
                bg_warp = get_background_uv(
                    h_matrix, h_size[1], h_size[0], h_bounds[:2], h_bounds[2:], W, H
                )
                bg_warp = bg_warp * 2 - 1
                bg_warp = bg_warp.permute(1, 2, 0)
                bg_warp = bg_warp[None].to(ctx.device)
                bg_zt = F.grid_sample(
                    self.zbar,
                    bg_warp,
                    mode="bilinear",
                    align_corners=False,
                )  # [1, C, H, W]
                feats.append(bg_zt)
                all_bg_warp.append(bg_warp)
            feats = torch.cat(feats, dim=0)  # [B, C, H, W]
            ctx.output.update({"bg_warp": torch.cat(all_bg_warp, dim=0)})
        else:
            feats = self.feature_cache[frame_indices].view(
                B, -1, H, W).to(device)

        # [[1, 2, 3, 4], [1, 2, 3, 4]] if B=2 and L=4
        layer_ids = torch.arange(1, L + 1, device=device).repeat(B)
        net_input = torch.cat(
            (
                masks * layer_ids.view(B * L, 1, 1, 1),
                torch.repeat_interleave(flows, L, dim=0) * masks,
                torch.repeat_interleave(feats, L, dim=0),
            ),
            dim=1,
        )  # [B*L, C, H, W]

        if self.feature_mode == "noise":
            bg_input = torch.cat(
                (
                    torch.zeros(1, 3, H, W, device=device),
                    self.zbar_up,
                ),
                dim=1,
            )  # [1, C, H, W]
            net_input = torch.cat((bg_input, net_input), dim=0)

        rgba, flow_out, _ = self.network.render(self.padder.pad(net_input))

        rgba = self.padder.unpad(rgba)
        flow_out = self.padder.unpad(flow_out)

        if self.feature_mode == "noise":
            bg_rgba = rgba[0]  # [4, H, W]
            rgba = rgba[1:]
            flow_out = flow_out[1:]
            ctx.output.update({"bg_rgba": bg_rgba})

        # trilinear interpolate time dim
        br_scale = F.interpolate(
            self.network.brightness_scale,
            size=(self.max_frames, 4, 7),
            mode="trilinear",
            align_corners=True,
        )  # [1, 1, max_frames, 4, 7]
        br_scale = br_scale[0, 0, frame_indices]  # [B, 4, 7]

        # bilinear upsample spatial dims
        br_scale = F.interpolate(
            br_scale.view(B, 1, 4, 7),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )  # [B, 1, H, W]

        ctx.output.update(
            {
                "rgba_layers": rgba,
                "flow_layers": flow_out,
                "br_scale": br_scale,
            }
        )

        if self.feature_mode == "noise":
            bg_offset = F.interpolate(
                self.network.bg_offset,
                size=(self.max_frames, 4, 7),
                mode="trilinear",
                align_corners=True,
            )  # [1, 2, max_frames, 4, 7]
            bg_offset = bg_offset[0, :, frame_indices]  # [2, B, 4, 7]
            bg_offset = bg_offset.transpose(0, 1)  # [B, 2, 4, 7]
            bg_offset = F.interpolate(
                bg_offset,
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )
            bg_offset = torch.permute(bg_offset, (0, 2, 3, 1))  # [B, H, W, 2]

            ctx.output.update(
                {
                    "bg_offset": bg_offset,
                }
            )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the Omnimatte network in train mode
        if mode or not self.use_eval_on_network:
            self.network.train(True)


def get_xyt_cache_file(H: int, W: int, pos_n_freq: int, t: int):
    return f"/tmp/omnimatte_xyt_cache/{H}x{W}_p{pos_n_freq}_t{t}.ckpt"


def load_xyt_cache(H: int, W: int, pos_n_freq: int, t: int) -> Optional[Tensor]:
    file = get_xyt_cache_file(H, W, pos_n_freq, t)
    if not os.path.isfile(file):
        return None
    return torch.load(file)


def save_xyt_cache(H: int, W: int, pos_n_freq: int, t: int, features: Tensor):
    file = Path(get_xyt_cache_file(H, W, pos_n_freq, t))
    os.makedirs(file.parent, exist_ok=True)
    torch.save(features, file)
