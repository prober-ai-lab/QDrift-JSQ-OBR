# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DriftOBR compressed-prefix capture and drift terms."""

from __future__ import annotations

import torch

from qdrift.obr.calibrate import capture_inputs

__all__ = ["capture_compressed_input", "drift_weight", "shrink_drift_weight"]


@torch.no_grad()
def capture_compressed_input(
    model: torch.nn.Module,
    capture: str,
    chunks: list[torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Capture one layer input from the partially compressed model."""
    return capture_inputs(model, [capture], chunks, device)[capture]


def drift_weight(weight: torch.Tensor, hessian_diff: torch.Tensor) -> torch.Tensor:
    """Project a weight matrix through the drift Hessian difference."""
    return weight @ hessian_diff


@torch.no_grad()
def shrink_drift_weight(
    shift: torch.Tensor,
    weight: torch.Tensor,
    fp_input: torch.Tensor,
    compressed_input: torch.Tensor,
    seqlen: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """James-Stein shrinkage of the drift term ``shift = W D`` toward OBR.

    ``D = E[(z_hat - x)^T z_hat]`` averages ``S`` calibration sequences, so
    ``shift`` is the mean of the per-sequence terms ``W D_s`` and its sampling
    variance is ``sum_s ||W D_s - shift||^2 / (S (S - 1))``.  Each output row
    (``row``) or the whole layer (``layer``) is scaled by
    ``max(0, 1 - variance / ||shift||^2)``: 1 keeps DAC, 0 falls back to OBR.
    Returns the shrunk term and the factors (one per row).
    """
    count = compressed_input.shape[0] // seqlen
    if count < 2:
        raise ValueError("drift shrinkage needs at least two calibration sequences")
    squares = torch.zeros(shift.shape[0], device=shift.device, dtype=torch.float64)
    for index in range(count):
        rows = slice(index * seqlen, (index + 1) * seqlen)
        z_hat = compressed_input[rows].to(device=shift.device, dtype=torch.float32)
        x = fp_input[rows].to(device=shift.device, dtype=torch.float32)
        per_sequence = ((z_hat - x) @ weight.t()).t() @ z_hat / seqlen
        squares += per_sequence.double().pow(2).sum(dim=1)
        del z_hat, x, per_sequence
    signal = shift.double().pow(2).sum(dim=1)
    variance = (squares - count * signal).clamp_min(0) / (count * (count - 1))
    if mode == "layer":
        variance, signal = variance.sum().expand_as(signal), signal.sum().expand_as(signal)
    factor = (1 - variance / signal.clamp_min(1e-30)).clamp_min(0).to(shift.dtype)
    return shift * factor[:, None], factor
