# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import Tensor

from core.loss import Loss, register_loss


@register_loss("mean_flow_match")
class MeanFlowMatchLoss(Loss):
    def forward(self, alpha_layers: Tensor, mean_dist_map: Tensor):
        """
        alpha_layers: (B, L, N) layered alpha
        mean_dist_map: (B, N) flow error map
        """
        return self.alpha * torch.mean(alpha_layers * mean_dist_map[:, None])
