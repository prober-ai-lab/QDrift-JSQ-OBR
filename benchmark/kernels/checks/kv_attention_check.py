# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness check for the fused INT8/INT4 decode-attention kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmark.inference.kv_attention import quantized_decode_attention
from benchmark.inference.quant import pack_i4, quantize_per_token, unpack_i4


def _dequant(codes: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 4:
        codes = unpack_i4(codes)
    return (codes.float() * scales.float()).half()


def _run(batch: int, heads: int, kv_heads: int, dim: int, length: int, bits: int) -> None:
    torch.manual_seed(0)
    group = heads // kv_heads
    q = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, kv_heads, length, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, kv_heads, length, dim, device="cuda", dtype=torch.float16)
    k_codes, k_scale = quantize_per_token(k, bits)
    v_codes, v_scale = quantize_per_token(v, bits)
    if bits == 4:
        k_packed, v_packed = pack_i4(k_codes), pack_i4(v_codes)
    else:
        k_packed, v_packed = k_codes, v_codes
    k_ref = _dequant(k_packed, k_scale, bits).repeat_interleave(group, dim=1)
    v_ref = _dequant(v_packed, v_scale, bits).repeat_interleave(group, dim=1)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref)
    out = quantized_decode_attention(
        q,
        k_packed,
        k_scale[..., 0].contiguous(),
        v_packed,
        v_scale[..., 0].contiguous(),
        length,
        bits,
    )
    error = (out.float() - ref.float()).abs().max().item()
    print(
        f"bits={bits} B={batch} Hq={heads} Hkv={kv_heads} D={dim} L={length}: "
        f"max={error:.5f}"
    )
    if error > 0.02:
        raise SystemExit("fused KV attention mismatch")


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    for bits in (8, 4):
        _run(2, 4, 2, 64, 37, bits)
        _run(2, 4, 2, 128, 129, bits)
        _run(1, 8, 2, 128, 257, bits)
    print("fused KV attention OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
