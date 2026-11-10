# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OBR, the baseline mechanism.

Pruning compensation by the OBS update on OBR's damped system, then the
early-half rounding error compensated onto the late half.  Bit-identical to
the pipeline before mechanisms were factored out.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from qdrift.mechanisms.base import Compensation, LayerStats, Mechanism
from qdrift.obr.curvature import couple_rows
from qdrift.obr.obr import compensate_pruning_error, compensate_quantization_error
from qdrift.quant.quantize import fake_quantize


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
        meta = {
            "mechanism": self.name,
            "damping": stats.damping,
            "lambda": stats.damping * stats.mean_diag,
        }
        if stats.curvature is None:
            values = compensate_quantization_error(
                weight, state, scale, qparams, grid, alpha=alpha
            )
            return Compensation(values=values, system=stats.system, meta=meta)

        # Output-curvature coupling (qdrift.obr.curvature): after each row-wise
        # stage, move the entries that stage solved for to the Kronecker optimum.
        target = weight @ stats.system
        if state.drift_weight is not None:
            target -= state.drift_weight
        keep = state.keep

        def couple(retained, free):
            full = torch.zeros_like(weight).scatter_(1, keep, retained)
            full, residual = couple_rows(
                full, free, target, stats.curvature, stats.system, stats.curvature_iters
            )
            return full.gather(1, keep), residual

        retained, residual1 = couple(state.retained, mask)
        state = replace(state, retained=retained, delta=retained - weight.gather(1, keep))
        values = compensate_quantization_error(
            weight, state, scale, qparams, grid, alpha=alpha
        )
        # The early columns are rounded from here on; only the late ones move.
        early = torch.arange(keep.shape[1], device=keep.device)[None, :] < state.boundary[:, None]
        rounded = torch.where(early, fake_quantize(values, scale, qparams, grid), values)
        late = mask & (torch.arange(weight.shape[1], device=mask.device) >= int(weight.shape[1] * alpha))
        values, residual2 = couple(rounded, late)
        meta["curvature_residual"] = [residual1, residual2]
        return Compensation(values=values, system=stats.system, meta=meta)
