# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Per-channel activation smoothing.
"""

from __future__ import annotations

import torch


def channel_max(weight: torch.Tensor) -> torch.Tensor:
    """Return the absolute maximum per input channel of a weight matrix."""
    return weight.abs().amax(dim=0)


def smooth_scale(
    activation_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the per-input-channel smoothing scale.

    The scale follows ``s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)``.
    """
    if activation_max.shape != weight_max.shape:
        raise ValueError("activation_max and weight_max must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    numerator = activation_max.clamp_min(eps).pow(alpha)
    denominator = weight_max.clamp_min(eps).pow(1.0 - alpha)
    return (numerator / denominator).clamp_min(eps)


def smooth_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Multiply input channels of a weight matrix by the smoothing scale."""
    if scale.numel() != weight.shape[1]:
        raise ValueError("scale length does not match weight input channels")
    return weight * scale.reshape(1, -1).to(weight)
