# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INT4/INT8 packing and per-token quantization helpers for the KV cache."""

from __future__ import annotations

import torch

__all__ = ["dequantize_per_token", "pack_i4", "quantize_per_token", "unpack_i4"]


def pack_i4(codes: torch.Tensor) -> torch.Tensor:
    """Pack two signed 4-bit values into one uint8 (low nibble first)."""
    if codes.shape[-1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    unsigned = (codes.to(torch.int16) & 0xF).to(torch.uint8)
    return (unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)).contiguous()


def unpack_i4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8-packed signed 4-bit values into int8 codes."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


def quantize_per_token(
    values: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token quantization along the last dimension."""
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    limit = float(2 ** (bits - 1) - 1)
    x = values.float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / limit
    codes = torch.round(x / scale).clamp(-limit, limit).to(torch.int8)
    return codes, scale.half()


def dequantize_per_token(
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-token codes back to float16."""
    return (codes.float() * scales.float()).half()
