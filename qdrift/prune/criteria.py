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
    """Return the Q-WANDA score.

    ``S_ij^2 = max(0, lambda * w_ij^2 - (1-lambda) * eps_ij^2) * ||x_j||^2``,
    where ``eps = Q(w) - w`` is the per-layer quantization error.
    """
    coefficient = lam * weight.pow(2) - (1.0 - lam) * quant_error.pow(2)
    coefficient = coefficient.clamp_min(0)
    return coefficient * scaler_row.clamp_min(0).reshape(1, -1)


def hessian_inverse_diag(system: torch.Tensor) -> torch.Tensor:
    """Return ``diag(H^-1)`` of the damped, symmetric layer Hessian."""
    factor, info = torch.linalg.cholesky_ex(system)
    if bool(info):
        return torch.diagonal(torch.linalg.inv(system))
    return torch.diagonal(torch.cholesky_inverse(factor))


def qap_score(
    weight: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
    step: torch.Tensor,
) -> torch.Tensor:
    """Return the quantization-aware pruning (QAP) score.

    ``S_ij = (w_ij^2 - step_ij^2 / 12) / [H^-1]_jj``: the OBS loss of pruning
    ``w_ij`` minus the OBS loss of the rounding error it incurs if retained,
    taken in expectation (``E[eps^2] = step^2 / 12``) because compensation
    changes the value that is finally rounded.  Larger means keep; the score
    may be negative, which ranks a weight as cheaper to prune than to quantize.
    """
    return (weight.pow(2) - step.pow(2) / 12.0) / hessian_inv_diag.reshape(1, -1)


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
