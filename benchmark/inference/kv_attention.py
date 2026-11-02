# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused INT8/INT4 decode attention over packed KV caches.

The kernel reads packed K/V codes directly, dequantizes tiles in registers,
and performs online-softmax attention.  This avoids materializing the full
dequantized prefix before SDPA.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

# Triton's launcher links -lcuda.  On headless CUDA containers only the driver
# stub is present, so expose it before Triton builds the launcher.
if "LIBRARY_PATH" not in os.environ:
    for candidate in (
        Path("/usr/lib/x86_64-linux-gnu/stubs"),
        Path("/usr/local/cuda/lib64/stubs"),
    ):
        if (candidate / "libcuda.so").exists():
            os.environ["LIBRARY_PATH"] = str(candidate)
            break

import triton
import triton.language as tl

__all__ = ["quantized_decode_attention"]


@triton.jit
def _decode_attention_kernel(
    Q,
    K,
    KS,
    V,
    VS,
    O,
    HQ,
    HKV,
    L,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sl,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + batch * stride_qb + q_head * stride_qh + offs_d * stride_qd
    ).to(tl.float32)

    m = -1.0e30
    l = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    offs_c = tl.arange(0, D // 2) if BITS == 4 else tl.arange(0, D)

    for start in range(0, L, BLOCK_N):
        index = start + offs_n
        mask = index < L

        k_codes = tl.load(
            K
            + batch * stride_kb
            + kv_head * stride_kh
            + index[:, None] * stride_kl
            + offs_c[None, :] * stride_kd,
            mask=mask[:, None],
            other=0,
        )
        if BITS == 4:
            low = (k_codes & 0x0F).to(tl.int32)
            high = ((k_codes >> 4) & 0x0F).to(tl.int32)
            low = tl.where(low >= 8, low - 16, low)
            high = tl.where(high >= 8, high - 16, high)
            k = tl.reshape(tl.join(low, high), (BLOCK_N, D)).to(tl.float32)
        else:
            k = k_codes.to(tl.float32)
        k_scale = tl.load(
            KS
            + batch * stride_sb
            + kv_head * stride_sh
            + index * stride_sl,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        k = k * k_scale[:, None]

        scores = tl.sum(q[None, :] * k, axis=1) * QK_SCALE
        scores = tl.where(mask, scores, -1.0e30)
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)

        v_codes = tl.load(
            V
            + batch * stride_vb
            + kv_head * stride_vh
            + index[:, None] * stride_vl
            + offs_c[None, :] * stride_vd,
            mask=mask[:, None],
            other=0,
        )
        if BITS == 4:
            low = (v_codes & 0x0F).to(tl.int32)
            high = ((v_codes >> 4) & 0x0F).to(tl.int32)
            low = tl.where(low >= 8, low - 16, low)
            high = tl.where(high >= 8, high - 16, high)
            v = tl.reshape(tl.join(low, high), (BLOCK_N, D)).to(tl.float32)
        else:
            v = v_codes.to(tl.float32)
        v_scale = tl.load(
            VS
            + batch * stride_sb
            + kv_head * stride_sh
            + index * stride_sl,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v = v * v_scale[:, None]

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    out = acc / l
    tl.store(
        O + batch * stride_ob + q_head * stride_oh + offs_d * stride_od,
        out.to(tl.float16),
    )


def _supported(head_dim: int) -> bool:
    return head_dim in (32, 64, 128)


def quantized_decode_attention(
    query: torch.Tensor,
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    length: int,
    bits: int,
) -> torch.Tensor:
    """Fused decode attention over a packed KV cache.

    ``query`` is ``[B, Hq, 1, D]`` FP16.  K/V are either INT8
    ``[B, Hkv, L, D]`` or packed INT4 ``[B, Hkv, L, D // 2]``.
    """
    if bits not in (8, 4):
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    batch, n_heads, _, head_dim = query.shape
    n_kv_heads = k_codes.shape[1]
    if n_heads % n_kv_heads:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if not _supported(head_dim):
        raise ValueError(f"unsupported head_dim {head_dim}")

    out = torch.empty_like(query)
    _decode_attention_kernel[(batch, n_heads)](
        query,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        out,
        n_heads,
        n_kv_heads,
        int(length),
        query.stride(0),
        query.stride(1),
        query.stride(3),
        k_codes.stride(0),
        k_codes.stride(1),
        k_codes.stride(2),
        k_codes.stride(3),
        k_scales.stride(0),
        k_scales.stride(1),
        k_scales.stride(2),
        v_codes.stride(0),
        v_codes.stride(1),
        v_codes.stride(2),
        v_codes.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(3),
        GROUP=n_heads // n_kv_heads,
        D=head_dim,
        BITS=bits,
        BLOCK_N=64 if head_dim >= 64 else 32,
        QK_SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return out
