# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F


class Padder:
    def __init__(self, pad_to, H, W):
        self.pads = [
            (pad_to - H % pad_to) % pad_to,
            (pad_to - W % pad_to) % pad_to,
        ]
        self.hw = [H, W]

    def pad(self, tensor):
        Hpad, Wpad = self.pads
        if Hpad == 0 and Wpad == 0:
            return tensor
        return F.pad(
            tensor,
            (Wpad // 2, Wpad - Wpad // 2, Hpad // 2, Hpad - Hpad // 2),
            "constant",
        )

    def unpad(self, tensor):
        Hpad, Wpad = self.pads
        if Hpad == 0 and Wpad == 0:
            return tensor
        H, W = self.hw
        return tensor[
            ..., Hpad // 2: Hpad // 2 + H, Wpad // 2: Wpad // 2 + W
        ].contiguous()


class PositionalEncoder:

    def __init__(
        self,
        in_dims: int,
        num_freq: int,
        max_freq: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        self.in_dims = in_dims
        self.num_freq = num_freq
        self.max_freq = max_freq
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0., max_freq, steps=num_freq)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0, 2. ** max_freq, steps=num_freq)

    @property
    def out_dims(self):
        dims = len(self.freq_bands) * len(self.periodic_fns) * self.in_dims
        if self.include_input:
            dims += self.in_dims
        return dims

    def __call__(self, x):
        x = x[..., :self.in_dims]
        encoding = []
        if self.include_input:
            encoding.append(x)

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                encoding.append(p_fn(x * freq))

        return torch.cat(encoding, dim=-1)
