# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense FP16 linear used for checkpoint embeddings and the LM head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DenseLinear"]


class DenseLinear(nn.Module):
    """Plain FP16 linear layer with no bias."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        weight = weight.detach().to(torch.float16).contiguous()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def storage_bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()
