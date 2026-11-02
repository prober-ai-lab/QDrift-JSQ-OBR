# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integer quantization primitives.
"""

from __future__ import annotations

import torch


def qrange(bits: int, grid: str) -> tuple[int, int, float]:
    """Return the signed code range and reconstruction-level offset."""
    if bits < 1:
        raise ValueError(f"bits must be positive, got {bits}")
    if grid not in ("midrise", "midtread"):
        raise ValueError(f"unknown quantization grid {grid}")
    if grid == "midtread" and bits < 2:
        raise ValueError("mid-tread grid requires at least 2 bits")
    half = 2 ** (bits - 1)
    offset = 0.5 if grid == "midrise" else 0.0
    return -half, half - 1, offset


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
) -> torch.Tensor:
    """Quantize values to integer codes."""
    qmin, qmax, _ = qparams
    if grid == "midtread":
        return torch.round(x / scale).clamp(qmin, qmax).to(torch.int8)
    if grid == "midrise":
        level = torch.floor(x.abs() / scale)
        code = torch.sign(x) * level
        code = code - (x < 0).to(code.dtype)
        return code.clamp(qmin, qmax).to(torch.int8)
    raise ValueError(f"unknown quantization grid {grid}")


def dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
) -> torch.Tensor:
    """Reconstruct floating-point values from integer codes."""
    _, _, offset = qparams
    return (codes.float() + offset) * scale


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
) -> torch.Tensor:
    """Quantize and immediately dequantize."""
    return dequantize(quantize(x, scale, qparams, grid), scale, qparams)
