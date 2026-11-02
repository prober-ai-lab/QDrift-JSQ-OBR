# Adapted from: https://github.com/spcl/QuaRot (commit 5008669)
# License: Apache-2.0
"""RMS normalization used by rotated models."""

from __future__ import annotations

import torch


class RMSNorm(torch.nn.Module):
    """Root mean square normalization without a learnable scale."""

    def __init__(self, mean_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps: float = eps
        self.mean_dim: int = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension."""
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
