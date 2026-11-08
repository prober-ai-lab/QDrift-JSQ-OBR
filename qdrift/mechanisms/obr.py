# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OBR, the baseline mechanism.

Pruning compensation by the OBS update on OBR's damped system, then the
early-half rounding error compensated onto the late half.  Bit-identical to
the pipeline before mechanisms were factored out.
"""

from __future__ import annotations

import torch

from qdrift.mechanisms.base import Compensation, LayerStats, Mechanism
from qdrift.obr.obr import compensate_pruning_error, compensate_quantization_error


class OBRMechanism(Mechanism):
    """OBR (optionally with QDrift's asymmetric drift terms)."""

    name = "obr"

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
        state = compensate_pruning_error(
            weight,
            stats.system,
            stats.raw,
            stats.diff,
            mask,
            alpha=alpha,
            drift_weight=stats.drift_weight,
        )
        values = compensate_quantization_error(
            weight, state, scale, qparams, grid, alpha=alpha
        )
        meta = {
            "mechanism": self.name,
            "damping": stats.damping,
            "lambda": stats.damping * stats.mean_diag,
        }
        return Compensation(values=values, system=stats.system, meta=meta)
