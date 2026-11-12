# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QDrift's mixture objective: its system matrix and drift term.

For one linear layer with weight ``W``, full-precision input ``x`` and the
input ``z`` of the partially compressed model, QDrift minimizes

    alpha E||W_hat z - W x||^2 + beta E||W_hat z - W z||^2 + gamma E||W_hat x - W x||^2

over the compressed weights, ``alpha + beta + gamma = 1``.  Each row is the
quadratic ``(w_hat - w*) H (w_hat - w*)^T + const`` with

    H  = (1 - gamma) E[z z^T] + gamma E[x x^T],
    w* = w - alpha w D H^-1,   D = E[(z - x) z^T],

so OBR's solver applies unchanged with the system ``H`` (damped) and the drift
term ``alpha W D``.  With ``alpha = 0`` there is no drift term at all, so the
solver runs OBR's own path (a zero drift term would reach the same optimum
through different floating-point arithmetic).  ``alpha = gamma = 0`` is OBR;
``alpha = 1, gamma = 0`` is the pure asymmetric target.  See docs/qdrift.md.
"""

from __future__ import annotations

import torch

__all__ = ["drift_weight", "mixture_hessians"]


def drift_weight(weight: torch.Tensor, hessian_diff: torch.Tensor) -> torch.Tensor:
    """Project a weight matrix through the (weighted) drift matrix."""
    return weight @ hessian_diff


def mixture_hessians(
    hessian_hat: torch.Tensor,
    hessian_fp: torch.Tensor | None,
    hessian_diff: torch.Tensor | None,
    mix_asym: float,
    mix_clean: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The mixture's undamped system ``(1 - gamma) H_z + gamma H_x`` and its
    weighted drift matrix ``alpha D``, None when ``alpha = 0`` (``hessian_fp``
    may be None when ``gamma = 0``, ``hessian_diff`` when ``alpha = 0``)."""
    hessian = hessian_hat
    if mix_clean > 0.0:
        if hessian_fp is None:
            raise ValueError("the clean-input term needs the full-precision Hessian")
        hessian = (1.0 - mix_clean) * hessian_hat + mix_clean * hessian_fp
    if mix_asym <= 0.0:
        return hessian, None
    if hessian_diff is None:
        raise ValueError("the asymmetric term needs the drift matrix")
    return hessian, mix_asym * hessian_diff
