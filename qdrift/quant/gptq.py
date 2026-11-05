# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Column-wise GPTQ quantization with a precomputed scale.

The pipeline computes the scale before pruning/compensation.  GPTQ therefore
receives that scale and only performs error feedback; it never re-searches the
clip ratio.  ``alpha`` keeps the OBR convention: the first ``alpha`` fraction
of columns is plain RTN, the rest uses sequential GPTQ.
"""

from __future__ import annotations

import torch

from qdrift.quant.quantize import dequantize, fake_quantize, quantize

__all__ = ["gptq_quantize", "rtn_quantize"]


def _hessian_inverse(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    n = hessian.shape[0]
    eye = torch.eye(n, device=hessian.device, dtype=hessian.dtype)
    damped = hessian.float() + damp * eye
    try:
        factor = torch.linalg.cholesky(damped)
        hinv = torch.cholesky_inverse(factor)
    except torch.linalg.LinAlgError:
        hinv = torch.linalg.inv(damped)
    return torch.linalg.cholesky(hinv, upper=True).to(hessian.dtype)


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
    alpha: float = 0.5,
    damp: float = 1e-6,
) -> torch.Tensor:
    """Return signed integer codes produced by GPTQ.

    Args:
        weight: Matrix to quantize, shape ``[out, in]``.
        hessian: Input Hessian in the same basis, shape ``[in, in]``.
        mask: Boolean retained mask, shape ``[out, in]``.
        scale: Elementwise quantization step, shape ``[out, in]``.
        qparams: ``(qmin, qmax, offset)`` from :func:`qrange`.
        grid: ``midrise`` or ``midtread``.
        alpha: Fraction of leading columns kept as plain RTN.
        damp: Relative diagonal damping for the Hessian inverse.
    """
    work = weight.float().clone()
    cols = work.shape[1]
    mask = mask.bool()
    hessian_inv = _hessian_inverse(hessian, damp)
    cut = int(alpha * cols)

    for col in range(cols):
        column = work[:, col]
        column_scale = scale[:, col]
        quantized = fake_quantize(
            column.unsqueeze(1),
            column_scale.unsqueeze(1),
            qparams,
            grid,
        ).squeeze(1)
        if col < cut:
            work[:, col] = torch.where(
                mask[:, col],
                quantized,
                torch.zeros_like(quantized),
            )
            continue

        target = torch.where(
            mask[:, col],
            quantized,
            torch.zeros_like(quantized),
        )
        error = column - target
        work[:, col] = target
        if col + 1 < cols:
            diagonal = hessian_inv[col, col].clamp_min(1e-12)
            work[:, col + 1 :] -= torch.outer(
                error / diagonal,
                hessian_inv[col, col + 1 :],
            )

    codes = quantize(work, scale, qparams, grid)
    return codes * mask.to(torch.int8)


def rtn_quantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest quantize and immediately dequantize.

    Args:
        weight: Weight matrix with shape ``[out_features, in_features]``.
        scale: Elementwise quantization step.
        qparams: Code range and reconstruction offset.
        grid: ``midrise`` or ``midtread``.

    Returns:
        The integer codes and the dequantized weight matrix.
    """
    codes = quantize(weight, scale, qparams, grid)
    return codes, dequantize(codes, scale, qparams)
