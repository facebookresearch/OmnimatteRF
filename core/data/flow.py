# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from core.data import DataSource, register_data_source
from third_party.RAFT.core.utils.frame_utils import readFlow
from utils.image_utils import read_image_np
from utils.io_utils import multi_glob_sorted


@register_data_source("flow")
class FlowDataSource(DataSource):
    def __init__(
        self,
        root: Path,
        image_hw: Tuple[int, int],
        flow_path: str,
        confidence_path: str,
    ):
        super().__init__(root, image_hw)
        self.flows = self.load_flows(root / flow_path)
        self.confidences = self.load_confidences(root / confidence_path)

        confidence_sum = self.confidences.sum(dim=(1, 2), keepdim=True)
        confidence_scaled_flow = self.flows * self.confidences[..., None] / confidence_sum[..., None]
        mean_flow = confidence_scaled_flow.sum(dim=(1, 2))  # (N, 2)

        self.mean_dist_map = (self.flows - mean_flow.view(len(mean_flow), 1, 1, 2)).abs().mean(dim=-1)

    def __len__(self) -> int:
        return len(self.flows)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "flow": self.flows[idx],
            "flow_confidence": self.confidences[idx],
            "flow_mean_dist_map": self.mean_dist_map[idx],
        }

    def get_keys(self) -> List[str]:
        return ["flow", "flow_confidence", "flow_mean_dist_map"]

    def load_flows(self, path: Path) -> Tensor:
        files = multi_glob_sorted(path, ["*.flo"])
        flows = torch.stack(
            [torch.from_numpy(readFlow(f)) for f in files], dim=0
        )  # [N-1, H, W, 2]

        flows, scale = self._scale_to_image_size(
            "flows", torch.permute(flows, (0, 3, 1, 2))
        )
        flows = torch.permute(flows, (0, 2, 3, 1))
        flows *= scale

        return flows

    def load_confidences(self, path: Path) -> Tensor:
        files = multi_glob_sorted(path, ["*.png"])
        confidences = torch.stack(
            [torch.from_numpy(read_image_np(f)) for f in files], dim=0
        )  # [N-1, H, W]

        confidences, _ = self._scale_to_image_size(
            "confidences", confidences[:, None])
        confidences = confidences[:, 0]

        return confidences
