# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
The interface between calibration and a compensation mechanism.

Calibration (plain sequential or QDrift's drift-aware, dual-path one) hands
each layer a :class:`LayerStats`.  A :class:`Mechanism` turns the pre-mask
weight, the mask, and these statistics into the retained values that the
final weight quantizer (RTN or the late-half GPTQ) rounds, together with the
system matrix that GPTQ must use for those values.  See docs/mechanisms.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class LayerStats:
    """Calibration statistics of one layer, in the rotated basis.

    Attributes:
        mean_diag: ``mean(diag H)``, ``H = E[z z^T]`` over the layer's
            calibration inputs ``z`` (see docs/mechanisms.md).
        system: OBR's damped system ``H + damping * mean(diag H) I``.
        raw: the matrix OBR uses for cross terms; the pipeline passes
            ``system`` (off-diagonal blocks are those of ``H``).
        diff: the drift matrix ``E[(z - x) z^T]``, ``x`` the input the
            target output is computed on; None when ``z = x``.
        drift_weight: ``w @ diff`` for these rows when precomputed, else None.
        damping: OBR's relative damping.
    """

    mean_diag: float
    system: torch.Tensor
    raw: torch.Tensor
    diff: torch.Tensor | None = None
    drift_weight: torch.Tensor | None = None
    damping: float = 0.01


@dataclass
class Compensation:
    """What a mechanism returns for one layer (or row slice).

    Attributes:
        values: compensated retained values ``[rows, kept]`` in the column
            order of ``split_mask``, early columns not yet rounded.
        system: the matrix the late-half GPTQ must use for these values.
        meta: per-layer metadata written to the checkpoint.
    """

    values: torch.Tensor
    system: torch.Tensor
    meta: dict[str, object] = field(default_factory=dict)


class Mechanism(ABC):
    """A pruning + quantization compensation mechanism."""

    name: str = ""
    # Whether calibration replays the prefix with the runtime activation
    # quantizer and hands the layer Q_a(z) on the tokens whose max|z_t|
    # exceeds ``outlier_factor`` times the median, z elsewhere.
    quantizes_inputs: bool = False
    outlier_factor: float = 0.0

    @abstractmethod
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
        """Compensate pruning and early-half rounding of ``weight``.

        Args:
            weight: pre-mask rotated weight rows ``[rows, cols]``.
            mask: retained mask, ``True`` = keep, equal count per row.
            stats: calibration statistics of the layer.
            scale: quantization step of the retained values ``[rows, kept]``.
            qparams: ``(qmin, qmax, offset)`` of the weight grid.
            grid: ``midrise`` or ``midtread``.
            alpha: early/late split; columns before ``int(cols * alpha)``
                are rounded here and compensated onto the later ones.
        """
