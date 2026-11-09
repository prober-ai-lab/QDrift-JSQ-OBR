# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic activation quantization, as the evaluation runtime applies it.

Mirrors ``benchmark.kernels.lib.int8_linear.quantize_activations``: optional
clipping of each token at ``clip_ratio * max|x|``, then one step per token
(or per group of ``group_size`` input features), ``step = max|x| / limit``
with ``limit = max(|qmin|, |qmax|)``.  Compression uses it to replay the
compressed prefix and, on the outlier tokens, to form the layer statistics
(the qir mechanism).
"""

from __future__ import annotations

import torch

from qdrift.quant.quantize import fake_quantize, qrange

__all__ = [
    "activation_step",
    "fake_quantize_activations",
    "outlier_tokens",
    "quantize_outlier_tokens",
]


def _group_width(cols: int, granularity: str, group_size: int | None) -> int:
    """Features sharing one step; per-token and per-tensor use the full row."""
    if granularity == "group" and group_size and 0 < group_size <= cols:
        return int(group_size)
    return cols


def _clip(x: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    if clip_ratio >= 1.0:
        return x
    bound = x.abs().amax(dim=1, keepdim=True) * clip_ratio
    return x.clamp(-bound, bound)


def activation_step(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str = "token",
    group_size: int | None = None,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Elementwise quantization step of ``x`` with shape ``[tokens, features]``."""
    x = _clip(x.float(), clip_ratio)
    rows, cols = x.shape
    width = _group_width(cols, granularity, group_size)
    groups = (cols + width - 1) // width
    padding = groups * width - cols
    padded = torch.nn.functional.pad(x, (0, padding)) if padding else x
    amax = padded.abs().reshape(rows, groups, width).amax(dim=2).clamp_min(1e-8)
    if bits == 1:
        step = amax
    else:
        qmin, qmax, _ = qrange(bits, grid)
        step = amax / max(abs(qmin), abs(qmax))
    return step.repeat_interleave(width, dim=1)[:, :cols]


def fake_quantize_activations(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str = "token",
    group_size: int | None = None,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize and dequantize ``x`` exactly as the evaluation runtime does."""
    step = activation_step(x, bits, grid, granularity, group_size, clip_ratio)
    clipped = _clip(x.float(), clip_ratio)
    if bits == 1:
        return torch.sign(clipped) * step
    return fake_quantize(clipped, step, qrange(bits, grid), grid)


@torch.no_grad()
def outlier_tokens(
    inputs: torch.Tensor,
    factor: float,
    device: str | torch.device = "cuda",
    chunk: int = 4096,
) -> torch.Tensor:
    """Rows of ``[tokens, features]`` inputs with ``max|x_t| > factor * median``.

    The median is over all rows.  The mask lives on the device of ``inputs``.
    """
    amax = torch.cat(
        [
            inputs[start : start + chunk].to(device=device, dtype=torch.float32)
            .abs()
            .amax(dim=1)
            for start in range(0, int(inputs.shape[0]), chunk)
        ]
    )
    return (amax > factor * amax.median()).to(inputs.device)


def quantize_outlier_tokens(
    inputs: torch.Tensor,
    factor: float,
    bits: int,
    grid: str,
    granularity: str = "token",
    group_size: int | None = None,
    clip_ratio: float = 1.0,
    device: str | torch.device = "cuda",
    chunk: int = 4096,
) -> torch.Tensor:
    """``fake_quantize_activations`` on the outlier tokens, the others unchanged.

    Outlier tokens are those of ``outlier_tokens(inputs, factor)``.  Rows are
    streamed through ``device`` in chunks; the result keeps the device and
    dtype of ``inputs``.
    """
    out = inputs.clone()
    rows = outlier_tokens(inputs, factor, device, chunk).nonzero().squeeze(1)
    for start in range(0, int(rows.numel()), chunk):
        index = rows[start : start + chunk]
        values = inputs[index].to(device=device, dtype=torch.float32)
        quantized = fake_quantize_activations(
            values, bits, grid, granularity, group_size, clip_ratio
        )
        out[index] = quantized.to(device=out.device, dtype=out.dtype)
    return out
