# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pruning criteria.
"""

from __future__ import annotations

import torch


def wanda_score(weight: torch.Tensor, scaler_row: torch.Tensor) -> torch.Tensor:
    """Return the WANDA score ``|w| * ||x||``."""
    return weight.abs() * scaler_row.clamp_min(0).sqrt().reshape(1, -1)


def qwanda_score(
    weight: torch.Tensor,
    scaler_row: torch.Tensor,
    quant_error: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Return the Q-WANDA score."""
    coefficient = lam * weight.pow(2) - (1.0 - lam) * quant_error.pow(2)
    coefficient = coefficient.clamp_min(0)
    return coefficient * scaler_row.clamp_min(0).reshape(1, -1)


def magnitude_score(weight: torch.Tensor) -> torch.Tensor:
    """Return the magnitude score."""
    return weight.abs()


def random_score(
    weight: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a random score for a random-pruning baseline."""
    return torch.rand(
        weight.shape,
        device=weight.device,
        dtype=weight.dtype,
        generator=generator,
    )
