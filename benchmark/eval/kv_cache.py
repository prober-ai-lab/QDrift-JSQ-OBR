# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantized KV cache for WAKV benchmark runs.

K/V tensors are stored as INT8 codes plus one scale per token.  The attention
kernel itself still receives dequantized K/V (Hugging Face SDPA), so the cache
memory is reduced and the quantize/dequantize path is exercised, while integer
Tensor Cores are used by the compressed Linear layers.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from qdrift.quant.quantize import qrange

__all__ = ["QuantizedDynamicCache"]


def _quant_params(bits: int, grid: str) -> tuple[int, int, float]:
    if bits == 1:
        return -1, 1, 0.0
    return qrange(bits, grid)


def _quantize_last_dim(
    x: torch.Tensor,
    bits: int,
    grid: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    qmin, qmax, offset = _quant_params(bits, grid)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if bits == 1:
        scale = amax
        codes = torch.sign(x)
    else:
        limit = max(abs(qmin), abs(qmax))
        scale = amax / limit
        if offset:
            # Mid-rise reconstructs (c + 0.5) * s: sign-magnitude floor,
            # matching qdrift.quant.quantize.quantize.
            level = torch.floor(x.abs() / scale)
            codes = torch.where(x < 0, -level - 1, level).clamp(qmin, qmax)
        else:
            codes = torch.round(x / scale).clamp(qmin, qmax)
    return codes.to(torch.int8), scale.to(torch.float16), offset


def _dequantize_last_dim(
    codes: torch.Tensor,
    scales: torch.Tensor,
    offset: float,
) -> torch.Tensor:
    return ((codes.float() + offset) * scales.float()).to(torch.float16)


class _QuantizedLayer(DynamicLayer):
    def __init__(self, k_bits: int, v_bits: int, grid: str) -> None:
        super().__init__()
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.grid = grid
        self.key_scales: torch.Tensor | None = None
        self.value_scales: torch.Tensor | None = None
        self.k_offset = 0.0
        self.v_offset = 0.0

    def _store(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        k_codes, k_scales, self.k_offset = _quantize_last_dim(
            key_states, self.k_bits, self.grid
        )
        v_codes, v_scales, self.v_offset = _quantize_last_dim(
            value_states, self.v_bits, self.grid
        )
        if not self.is_initialized:
            self.keys = k_codes
            self.values = v_codes
            self.key_scales = k_scales
            self.value_scales = v_scales
            self.is_initialized = True
        else:
            self.keys = torch.cat([self.keys, k_codes], dim=-2)
            self.values = torch.cat([self.values, v_codes], dim=-2)
            self.key_scales = torch.cat([self.key_scales, k_scales], dim=-2)
            self.value_scales = torch.cat([self.value_scales, v_scales], dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._store(key_states, value_states)
        keys = _dequantize_last_dim(self.keys, self.key_scales, self.k_offset)
        values = _dequantize_last_dim(self.values, self.value_scales, self.v_offset)
        return keys, values

    def reset(self) -> None:
        self.keys = self.values = None
        self.key_scales = self.value_scales = None
        self.is_initialized = False


class QuantizedDynamicCache(DynamicCache):
    """DynamicCache whose stored K/V payloads are INT8 quantized."""

    def __init__(self, config, k_bits: int, v_bits: int, grid: str) -> None:
        super().__init__(config=config)
        self.layers = [
            _QuantizedLayer(k_bits, v_bits, grid)
            for _ in range(config.num_hidden_layers)
        ]
