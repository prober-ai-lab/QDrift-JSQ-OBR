# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Wiener: activation-noise damping with Wiener shrinkage.

Per output row, with the calibration Hessian ``H`` (and, under QDrift, the
drift matrix ``D = E[(z - x) z^T]``), the compressed layer sees quantized
inputs ``z + n`` with ``E[n n^T] = Sigma_a ~= sigma_a^2 I`` after Hadamard
rotation.  The expected output error of a weight ``v`` with ``v_P = 0`` is

    f(v) = v^T (H + Sigma_a) v - 2 v^T b + const,    b = H w - D^T w,

so with ``lambda = max(noise_scale * sigma_a^2, 1e-4 mean(diag H))``:

    pruning:       v_K = (H_KK + lambda I)^-1 b_K,  v_P = 0           (target w*)
    quantization:  f(v) = (v - w*)^T (H + lambda I) (v - w*) + const   on v_P = 0

The first line is the Wiener estimate; unlike OBR's update it shrinks the
kept weights (``b_K = H_KK w_K + H_KP w_P - (D^T w)_K``).  The second makes
the early-half solve and the late-half GPTQ plain OBR/GPTQ on the system
``H + lambda I`` with target ``w*`` and no drift terms.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from qdrift.mechanisms.base import Compensation, LayerStats, Mechanism
from qdrift.obr.obr import (
    PruningCompensationState,
    compensate_pruning_error,
    compensate_quantization_error,
)

# lambda never falls below this fraction of mean(diag H), to keep the
# system factorizable when the activations are not quantized.
FLOOR = 1e-4


def noise_damping(
    mean_diag: float, act_noise: float | None, noise_scale: float = 1.0
) -> tuple[float, bool]:
    """``lambda_eff`` and whether the numerical floor set it."""
    floor = FLOOR * mean_diag
    noise = noise_scale * float(act_noise or 0.0)
    return max(noise, floor), floor > noise


def wiener_system(hessian: torch.Tensor, lam: float) -> torch.Tensor:
    """``H + lambda I``, symmetrized."""
    eye = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    system = hessian + lam * eye
    return 0.5 * (system + system.t())


def wiener_target(
    weight: torch.Tensor,
    mask: torch.Tensor,
    hessian: torch.Tensor,
    lam: float,
    diff: torch.Tensor | None = None,
    drift_weight: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> tuple[PruningCompensationState, torch.Tensor]:
    """Pruning state holding the Wiener target ``w*_K``, and ``H + lambda I``.

    The returned state carries no drift terms: the quantization stage is
    symmetric around ``w*``.
    """
    system = wiener_system(hessian, lam)
    drift = drift_weight
    if drift is None and diff is not None:
        drift = weight @ diff
    # compensate_pruning_error returns w_K + S^-1 (H_KP w_P - g_K) for a
    # linear term g; g = lambda w + D^T w turns that into the Wiener
    # target S^-1 (H_KK w_K + H_KP w_P - (D^T w)_K).
    shift = lam * weight if drift is None else lam * weight + drift
    state = compensate_pruning_error(
        weight, system, system, None, mask, alpha=alpha, drift_weight=shift
    )
    return replace(state, drift_weight=None), system


class WienerMechanism(Mechanism):
    """Activation-noise damping + Wiener shrinkage."""

    name = "wiener"
    needs_hessian = True
    needs_act_noise = True

    def __init__(self, noise_scale: float = 1.0) -> None:
        self.noise_scale = noise_scale

    def compensate(
        self,
        weight: torch.Tensor,
        mask: torch.Tensor,
        stats: LayerStats,
        scale: torch.Tensor,
        qparams: tuple[int, int, float],
        grid: str,
        alpha: float,
    ) -> Compensation:
        if stats.hessian is None:
            raise ValueError("the wiener mechanism needs the undamped Hessian")
        lam, floored = noise_damping(stats.mean_diag, stats.act_noise, self.noise_scale)
        state, system = wiener_target(
            weight, mask, stats.hessian, lam, stats.diff, stats.drift_weight, alpha
        )
        values = compensate_quantization_error(
            weight, state, scale, qparams, grid, alpha=alpha
        )
        meta = {
            "mechanism": self.name,
            "act_noise": float(stats.act_noise or 0.0),
            "noise_scale": self.noise_scale,
            "lambda": lam,
            "lambda_over_mean_diag": lam / max(stats.mean_diag, 1e-30),
            "floor": floored,
        }
        return Compensation(values=values, system=system, meta=meta)
