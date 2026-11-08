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


def hessian_inverse_diag(system: torch.Tensor) -> torch.Tensor:
    """Return ``diag(H^-1)`` of the damped, symmetric layer Hessian."""
    factor, info = torch.linalg.cholesky_ex(system)
    if bool(info):
        return torch.diagonal(torch.linalg.inv(system))
    return torch.diagonal(torch.cholesky_inverse(factor))


def sparsegpt_score(
    weight: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
) -> torch.Tensor:
    """Return the SparseGPT (OBS) saliency ``w_ij^2 / [H^-1]_jj``.

    This is the loss of pruning ``w_ij`` when the other weights of the row are
    compensated; the mask is chosen in one shot from the dense weights, and the
    compensation itself is left to OBR.
    """
    return weight.pow(2) / hessian_inv_diag.reshape(1, -1)


def magnitude_score(weight: torch.Tensor) -> torch.Tensor:
    """Return the magnitude score."""
    return weight.abs()
