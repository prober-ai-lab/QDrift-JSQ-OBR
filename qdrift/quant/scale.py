# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Scale construction and clip search.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from qdrift.quant.quantize import fake_quantize, qrange

MSE_CLIP_GRID = 100
MSE_CLIP_MAXSHRINK = 0.8
MSE_CLIP_NORM = 2.4


def reduce_amax(
    x: torch.Tensor,
    granularity: str = "group",
    group_size: int | None = None,
    axis: int = -1,
) -> torch.Tensor:
    """Return absolute maxima broadcastable to ``x``."""
    if granularity == "tensor":
        return x.abs().max()
    if granularity in ("channel", "token"):
        return x.abs().amax(dim=axis, keepdim=True)
    if granularity != "group":
        raise ValueError(f"unknown granularity {granularity}")
    if axis not in (-1, x.ndim - 1):
        raise ValueError("group quantization currently supports the last axis")
    if group_size is None or group_size <= 0 or group_size >= x.shape[-1]:
        return x.abs().amax(dim=-1, keepdim=True)

    shape = x.shape
    rows, cols = 1, shape[-1]
    for dim in shape[:-1]:
        rows *= int(dim)
    flat = x.reshape(rows, cols)
    groups = (cols + group_size - 1) // group_size
    padding = groups * group_size - cols
    padded = torch.nn.functional.pad(flat, (0, padding)) if padding else flat
    amax = (
        padded.abs()
        .reshape(rows, groups, group_size)
        .amax(
            dim=2,
            keepdim=True,
        )
    )
    amax = amax.expand(rows, groups, group_size).reshape(rows, -1)[:, :cols]
    return amax.reshape(*shape[:-1], cols)


def scale_from_amax(
    amax: torch.Tensor,
    qparams: tuple[int, int, float],
    clip_rho: float = 1.0,
) -> torch.Tensor:
    """Build a quantization step from an absolute maximum."""
    if clip_rho <= 0:
        raise ValueError(f"clip_rho must be positive, got {clip_rho}")
    _, qmax, offset = qparams
    return clip_rho * amax.clamp_min(1e-12) / (qmax + offset)


def _mse_clip_channel(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official OBR per-channel/per-tensor MSE clip search.

    This mirrors ``WeightQuantizer.find_params(..., mse=True)`` from the OBR
    QuaRot implementation: symmetric min/max, 80 shrink ratios in [0.2, 1.0],
    and an L2.4 reconstruction error.  It is used for range clipping on
    channel/tensor weight scales; group scales keep the explicit rho grid.
    """
    qparams = qrange(bits, grid)
    _, qmax, offset = qparams
    limit = float(qmax + offset)
    if granularity == "tensor":
        xmin = x.min().clamp(max=0)
        xmax = x.max().clamp(min=0)
        amax = torch.maximum(xmin.abs(), xmax).clamp_min(1e-5)
        rows = 1
        matrix = x.reshape(1, -1)
        scales = torch.ones(1, device=x.device, dtype=x.dtype)
    else:
        rows = x.shape[0]
        xmin = x.min(dim=axis).values.clamp(max=0)
        xmax = x.max(dim=axis).values.clamp(min=0)
        amax = torch.maximum(xmin.abs(), xmax).clamp_min(1e-5)
        matrix = x
        scales = torch.ones(rows, device=x.device, dtype=x.dtype)
    best = torch.full((rows,), float("inf"), device=x.device, dtype=x.dtype)
    best_rho = torch.ones_like(scales)
    for index in range(int(MSE_CLIP_MAXSHRINK * MSE_CLIP_GRID)):
        rho = 1.0 - index / MSE_CLIP_GRID
        candidate = (rho * amax / limit).reshape(rows, 1)
        fake = fake_quantize(matrix.reshape(rows, -1), candidate, qparams, grid)
        error = (fake.float() - matrix.reshape(rows, -1).float()).abs()
        error = error.pow(MSE_CLIP_NORM).sum(dim=1)
        better = error < best
        best = torch.where(better, error, best)
        scales = torch.where(better, candidate.reshape(-1), scales)
        best_rho = torch.where(better, torch.full_like(scales, rho), best_rho)
    return scales.reshape(-1, 1), best_rho.reshape(-1, 1)


def search_clip(
    x: torch.Tensor,
    granularity: str,
    group_size: int | None,
    axis: int,
    bits: int,
    grid: str,
    rhos: Sequence[float],
    mse_clip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the clip ratio with the lowest reconstruction error."""
    if mse_clip and granularity in ("channel", "token", "tensor"):
        return _mse_clip_channel(x, bits, grid, granularity, axis)
    qparams = qrange(bits, grid)
    amax = reduce_amax(x, granularity, group_size, axis)
    best_scale = torch.ones_like(x if amax.ndim == x.ndim else amax)
    best_rho = torch.zeros_like(best_scale)

    rows, cols = x.shape
    use_group = (
        granularity == "group" and group_size is not None and 0 < group_size < cols
    )
    if use_group:
        groups = (cols + group_size - 1) // group_size
        padding = groups * group_size - cols
        xg = torch.nn.functional.pad(x, (0, padding)) if padding else x
        xg = xg.reshape(rows, groups, group_size)
        best_mse_group = torch.full(
            (rows, groups),
            float("inf"),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        best_mse = torch.full_like(best_scale, float("inf"))

    for rho in rhos:
        scale = scale_from_amax(amax, qparams, rho)
        fake = fake_quantize(x, scale, qparams, grid)
        if use_group:
            fg = torch.nn.functional.pad(fake, (0, padding)) if padding else fake
            fg = fg.reshape(rows, groups, group_size)
            mse = (fg - xg).pow(2).mean(dim=2)
            better = mse < best_mse_group
            best_mse_group = torch.where(better, mse, best_mse_group)
            selector = better.unsqueeze(-1).expand(rows, groups, group_size)
            selector = selector.reshape(rows, -1)[:, :cols]
            best_scale = torch.where(selector, scale, best_scale)
            best_rho = torch.where(
                selector,
                torch.full_like(scale, rho),
                best_rho,
            )
        elif granularity in ("channel", "token"):
            mse = (fake - x).pow(2).mean(dim=axis, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_scale = torch.where(better, scale, best_scale)
            best_rho = torch.where(
                better,
                torch.full_like(scale, rho),
                best_rho,
            )
        else:
            mse = (fake - x).pow(2).mean()
            if mse < best_mse:
                best_scale = scale
                best_rho = torch.full_like(scale, rho)
                best_mse = torch.full_like(best_mse, mse)

    return best_scale, best_rho


@dataclass
class ScaleResult:
    """Elementwise and compact views of a chosen quantization scale."""

    scale: torch.Tensor
    rho: torch.Tensor
    scale_compact: torch.Tensor
    rho_compact: torch.Tensor
    granularity: str
    group_size: int | None


def effective_group_size(group_size: int | None, width: int) -> int:
    """Return the group width actually used for a row of length ``width``."""
    if group_size is None or group_size <= 0 or group_size >= width:
        return width
    return int(group_size)


def _reduce_granularity(granularity: str, group_size: int | None, width: int) -> str:
    if granularity == "group":
        return "group" if effective_group_size(group_size, width) < width else "channel"
    return granularity


def compact_scale(
    scale: torch.Tensor,
    granularity: str,
    group_size: int | None,
    width: int,
) -> torch.Tensor:
    """Convert an elementwise scale to its compact storage form."""
    if granularity == "tensor":
        return scale.reshape(1).contiguous()
    if granularity in ("channel", "token"):
        if scale.ndim == 1 or scale.shape[0] == 1:
            return scale.reshape(1, 1).contiguous()
        return scale[:, :1].contiguous()
    group = effective_group_size(group_size, width)
    if group >= width:
        if scale.ndim == 1 or scale.shape[0] == 1:
            return scale.reshape(1, 1).contiguous()
        return scale[:, :1].contiguous()
    return scale[:, ::group].contiguous()


def expand_scale(
    scale: torch.Tensor,
    granularity: str,
    group_size: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    """Expand a compact stored scale back to elementwise ``[rows, width]``."""
    if granularity == "tensor":
        return scale.reshape(1).expand(rows, width)
    if granularity in ("channel", "token"):
        return scale.reshape(rows, 1).expand(rows, width)
    group = effective_group_size(group_size, width)
    if group >= width:
        return scale.reshape(rows, 1).expand(rows, width)
    return scale.repeat_interleave(group, dim=1)[:, :width]


def compute_scale(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str,
    group_size: int | None,
    rhos: tuple[float, ...],
    axis: int = -1,
    mse_clip: bool = False,
) -> ScaleResult:
    """Choose the scale for ``x`` without pruning or compensation.

    Args:
        x: Matrix of values to quantize.
        bits: Weight bit width.
        grid: ``midrise`` or ``midtread``.
        granularity: ``tensor``, ``channel``, ``token`` or ``group``.
        group_size: Group width for ``granularity='group'``.
        rhos: Candidate clip ratios. Use ``(1.0,)`` for full range.
        axis: Reduction axis for channel/token.
        mse_clip: Use the OBR per-channel MSE shrink search for range clipping.
    """
    if x.ndim != 2:
        raise ValueError("compute_scale expects a 2-D matrix")
    if not rhos:
        raise ValueError("rhos must not be empty")
    width = x.shape[-1]
    reduce_gran = _reduce_granularity(granularity, group_size, width)
    # search_clip returns [rows, 1] for channel/token, [rows, groups] for a
    # real group layout, and a scalar for tensor.
    scale, rho = search_clip(
        x,
        reduce_gran,
        group_size,
        axis,
        bits,
        grid,
        rhos,
        mse_clip=mse_clip,
    )
    if granularity == "tensor":
        scale = scale.reshape(1)
        rho = rho.reshape(1)
    # search_clip returns an elementwise scale for channel/token/group; only
    # tensor is scalar.
    scale_compact = compact_scale(scale, granularity, group_size, width)
    rho_compact = compact_scale(rho, granularity, group_size, width)
    return ScaleResult(
        scale=expand_scale(scale_compact, granularity, group_size, x.shape[0], width),
        rho=expand_scale(rho_compact, granularity, group_size, x.shape[0], width),
        scale_compact=scale_compact,
        rho_compact=rho_compact,
        granularity=granularity,
        group_size=group_size,
    )
