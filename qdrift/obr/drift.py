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

import math

import torch

__all__ = [
    "auto_mixture",
    "drift_weight",
    "ledoit_wolf_intensity",
    "mixture_hessians",
    "precision_weight",
    "trust_region_weight",
]


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


def trust_region_weight(cancellable: float, error: float) -> float:
    """alpha with ``alpha^2 s = (1 - alpha^2) eps``: the drift correction may
    move the layer output by no more than the compression error the layer
    already injects (docs/mix-auto.md).  ``s`` is the in-sample cancellable
    drift ``tr(W D H^-1 D^T W^T)``, ``eps`` the OBR solution's error
    ``tr(E H E^T)``; no drift (``s = 0``) gives 1, ``s >> eps`` gives
    ``sqrt(eps / s)``."""
    if cancellable <= 0.0:
        return 1.0
    if error <= 0.0:
        return 0.0
    return math.sqrt(error / (cancellable + error))


def precision_weight(cancellable: float, error: float) -> float:
    """alpha = eps / (s + eps) (``--mix-rule prec``): each of the two targets
    weighted by the inverse of its cost, the asymmetric target's ``s`` (how
    far its drift correction moves the layer output, ``tr(W D H^-1 D^T W^T)``)
    and the symmetric target's ``eps`` (the error ``tr(E H E^T)`` of OBR's
    solution).  A stated rule, chosen for its behaviour; it is the square of
    :func:`trust_region_weight`.  No drift (``s = 0``) gives 1."""
    if cancellable <= 0.0:
        return 1.0
    if error <= 0.0:
        return 0.0
    return error / (cancellable + error)


def ledoit_wolf_intensity(
    hessian_hat: torch.Tensor,
    hessian_half: torch.Tensor,
    hessian_fp: torch.Tensor,
) -> float:
    """Ledoit-Wolf intensity of shrinking ``H_Xhat`` toward ``H_X``:
    ``pi / (pi + b)`` with ``pi`` the sampling variance of ``H_Xhat`` and
    ``b = ||H_Xhat - H_X||_F^2`` the (plug-in) squared bias of the target.
    ``pi`` comes from two equal halves of the calibration set
    (``hessian_half`` is the first half's; the second is ``2 H_Xhat - first``,
    so ``pi = ||H_half - H_Xhat||_F^2``).  It lies in (0, 1) unless
    ``H_Xhat = H_X`` exactly (no drift: every target coincides), where it is 1."""
    variance = float((hessian_half - hessian_hat).square().sum())
    bias = float((hessian_hat - hessian_fp).square().sum())
    if variance + bias <= 0.0:
        return 1.0
    return variance / (variance + bias)


def auto_mixture(alpha: float, intensity: float) -> tuple[float, float]:
    """``(alpha, gamma)`` of the derived mixture: the asymmetric target gets the
    trust-region weight, and the remaining ``1 - alpha`` (the two targets
    that are not drift corrections) is split between the clean and symmetric
    targets by the Ledoit-Wolf intensity."""
    return alpha, (1.0 - alpha) * intensity
