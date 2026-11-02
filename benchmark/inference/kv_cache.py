# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache with direct INT8/INT4 quantization and fused decode attention.

Quantized decode reads packed K/V codes directly in the fused Triton kernel.
The dequantize-then-SDPA path is kept only as a fallback for head dimensions
not covered by the fused kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmark.inference.kv_attention import quantized_decode_attention
from benchmark.inference.quant import pack_i4, quantize_per_token, unpack_i4

__all__ = ["KVCache"]


class KVCache:
    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        head_dim: int,
        max_len: int,
        bits: int = 16,
        fused: bool = True,
        device: str | torch.device = "cuda",
    ) -> None:
        if bits not in (16, 8, 4):
            raise ValueError(f"bits must be 16, 8 or 4, got {bits}")
        if bits == 4 and head_dim % 2:
            raise ValueError("head_dim must be even for INT4 KV")
        self.bits = bits
        self.fused = bool(fused)
        self.n_layers = n_layers
        self.batch = batch
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = torch.device(device)
        self.length = 0

        shape = (n_layers, batch, n_kv_heads, max_len)
        if bits == 16:
            self.k = torch.empty(*shape, head_dim, dtype=torch.float16, device=self.device)
            self.v = torch.empty(*shape, head_dim, dtype=torch.float16, device=self.device)
        else:
            code_dim = head_dim if bits == 8 else head_dim // 2
            code_dtype = torch.int8 if bits == 8 else torch.uint8
            self.k = torch.empty(*shape, code_dim, dtype=code_dtype, device=self.device)
            self.v = torch.empty(*shape, code_dim, dtype=code_dtype, device=self.device)
            self.ks = torch.empty(*shape, dtype=torch.float16, device=self.device)
            self.vs = torch.empty(*shape, dtype=torch.float16, device=self.device)

    def reset(self) -> None:
        self.length = 0

    def nbytes(self) -> int:
        total = 0
        for name in ("k", "v", "ks", "vs"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    @torch.no_grad()
    def _store(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        start: int,
    ) -> None:
        # keys/values: [B, Hkv, S, D]
        length = keys.shape[2]
        if self.bits == 16:
            self.k[layer, :, :, start : start + length] = keys
            self.v[layer, :, :, start : start + length] = values
            return

        k_codes, k_scale = quantize_per_token(keys, self.bits)
        v_codes, v_scale = quantize_per_token(values, self.bits)
        if self.bits == 4:
            k_codes = pack_i4(k_codes)
            v_codes = pack_i4(v_codes)
        self.k[layer, :, :, start : start + length] = k_codes
        self.v[layer, :, :, start : start + length] = v_codes
        self.ks[layer, :, :, start : start + length] = k_scale[..., 0]
        self.vs[layer, :, :, start : start + length] = v_scale[..., 0]

    @torch.no_grad()
    def write_prefill(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        # [B, S, Hkv, D] -> [B, Hkv, S, D]
        self._store(layer, keys.transpose(1, 2), values.transpose(1, 2), 0)

    @torch.no_grad()
    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        pos: int,
    ) -> None:
        # [B, 1, Hkv, D] -> [B, Hkv, 1, D]
        self._store(layer, keys.transpose(1, 2), values.transpose(1, 2), pos)

    @torch.no_grad()
    def decode_attention(
        self,
        layer: int,
        query: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        # query: [B, Hq, 1, D]
        if self.fused and self.bits < 16 and self.head_dim in (32, 64, 128):
            return quantized_decode_attention(
                query,
                self.k[layer],
                self.ks[layer],
                self.v[layer],
                self.vs[layer],
                length,
                self.bits,
            )

        keys = self.k[layer, :, :, :length]
        values = self.v[layer, :, :, :length]
        if self.bits < 16:
            if self.bits == 4:
                keys = unpack_i4(keys)
                values = unpack_i4(values)
            k_scale = self.ks[layer, :, :, :length, None]
            v_scale = self.vs[layer, :, :, :length, None]
            keys = (keys.float() * k_scale.float()).half()
            values = (values.float() * v_scale.float()).half()

        group = query.shape[1] // self.n_kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        return F.scaled_dot_product_attention(query, keys, values)
