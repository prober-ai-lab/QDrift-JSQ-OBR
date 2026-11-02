# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused activation quantization for the INT8 Tensor Core path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["quantize_activations_triton"]


@triton.jit
def _quant_kernel(
    x_ptr,
    code_ptr,
    scale_ptr,
    rows,
    cols,
    group_size,
    clip_ratio,
    qmin,
    qmax,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    start = group * group_size
    end = tl.minimum(start + group_size, cols)
    offs = tl.arange(0, BLOCK)
    amax = 0.0
    for base in range(start, end, BLOCK):
        index = base + offs
        mask = index < end
        value = tl.load(
            x_ptr + row * cols + index,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(value)))
    amax = tl.maximum(amax * clip_ratio, 1e-8)
    if BITS == 1:
        scale = amax
    else:
        limit = tl.maximum(tl.abs(qmin), tl.abs(qmax)).to(tl.float32)
        scale = amax / limit
    tl.store(scale_ptr + row * tl.num_programs(1) + group, scale)
    for base in range(start, end, BLOCK):
        index = base + offs
        mask = index < end
        value = tl.load(
            x_ptr + row * cols + index,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if BITS == 1:
            code = tl.where(value > 0, 1.0, tl.where(value < 0, -1.0, 0.0))
        else:
            code = tl.floor(value / scale + 0.5)
            code = tl.maximum(tl.minimum(code, qmax), qmin)
        tl.store(
            code_ptr + row * cols + index,
            code.to(tl.int8),
            mask=mask,
        )


def quantize_activations_triton(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str,
    group_size: int | None,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Quantize activations to INT8 with Triton (token/group granularity)."""
    rows, cols = x.shape
    if granularity == "group":
        group = int(group_size or cols)
    else:
        group = cols
    if group <= 0 or group > cols:
        group = cols
    groups = (cols + group - 1) // group
    if bits == 1:
        qmin, qmax, offset = -1, 1, 0.0
    elif grid == "midrise":
        half = 2 ** (bits - 1)
        qmin, qmax, offset = -half, half - 1, 0.5
    else:
        qmin, qmax, offset = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1, 0.0
    codes = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    scales = torch.empty((rows, groups), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(min(1024, max(32, group)))
    _quant_kernel[(rows, groups)](
        x,
        codes,
        scales,
        rows,
        cols,
        group,
        clip_ratio,
        qmin,
        qmax,
        BITS=bits,
        BLOCK=block,
        num_warps=4,
    )
    return codes, scales, group, offset
