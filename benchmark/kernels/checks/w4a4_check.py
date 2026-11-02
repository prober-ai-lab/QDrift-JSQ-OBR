# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness and microbenchmark check for the 4:8 sparse INT4 kernel.

Run on a CUDA host with CUTLASS headers available:

    CUTLASS_INCLUDE=/path/to/cutlass/include \
        python -m benchmark.kernels.checks.w4a4_check

The check builds a 4:8 pairwise INT4 weight, runs the CUTLASS sparse Tensor
Core path, and compares it against the QuaRot dense INT4 kernel on the same
packed codes.  It then reports per-GEMM latency for representative LLM shapes.
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kernels.lib.w4a4 import (
    build_sparse_int4_4to8_weight,
    quarot_module,
)
from qdrift.quant.pack import pack_i4


def pairwise_4to8_mask(rows: int, cols: int, device: str) -> torch.Tensor:
    if cols % 8:
        raise ValueError("cols must be divisible by eight")
    pairs = torch.zeros(rows, cols // 8, 4, dtype=torch.bool, device=device)
    low = torch.randint(0, 2, (rows, cols // 8), device=device)
    high = torch.randint(2, 4, (rows, cols // 8), device=device)
    pairs.scatter_(2, low.unsqueeze(2), True)
    pairs.scatter_(2, high.unsqueeze(2), True)
    return pairs.unsqueeze(3).expand(-1, -1, -1, 2).reshape(rows, cols)


def reference(
    module,
    x: torch.Tensor,
    codes_padded: torch.Tensor,
    weight_scale: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    q, a_scale = module.sym_quant_dynamic(x, clip_ratio)
    acc = module.matmul(q, pack_i4(codes_padded.contiguous()).contiguous())
    return module.sym_dequant(acc, a_scale, weight_scale, 32)


def check(out_features: int, in_features: int, rows: int, device: str) -> None:
    module = quarot_module()
    torch.manual_seed(0)
    mask = pairwise_4to8_mask(out_features, in_features, device)
    codes = torch.randint(
        -8, 8, (out_features, in_features), device=device, dtype=torch.int8
    ) * mask
    weight_scale = torch.rand(out_features, device=device).half() * 0.01 + 1e-3
    weight = build_sparse_int4_4to8_weight(codes, mask, weight_scale, 0.9)
    x = torch.randn(rows, in_features, device=device).half()

    sparse = weight.forward(x)
    padded_in = weight.padded_in_features
    x_padded = torch.nn.functional.pad(x, (0, padded_in - in_features))
    codes_padded = torch.nn.functional.pad(
        codes, (0, padded_in - in_features)
    )
    dense = reference(module, x_padded, codes_padded, weight_scale, 0.9)
    error = (sparse.float() - dense.float()).abs()
    print(
        f"correctness O={out_features} K={in_features} R={rows}: "
        f"max={error.max().item():.5f} mean={error.mean().item():.6f}"
    )
    if not torch.allclose(sparse.float(), dense.float(), atol=1e-2, rtol=1e-2):
        raise SystemExit("sparse INT4 kernel mismatch")


def event_time(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_shape(out_features: int, in_features: int, rows: int) -> None:
    module = quarot_module()
    device = "cuda"
    padded_in = (in_features + 255) // 256 * 256
    mask = pairwise_4to8_mask(out_features, padded_in, device)
    codes = torch.randint(
        -8, 8, (out_features, padded_in), device=device, dtype=torch.int8
    ) * mask
    weight = build_sparse_int4_4to8_weight(
        codes[:, :in_features].contiguous(),
        pairwise_4to8_mask(out_features, in_features, device),
        torch.ones(out_features, device=device, dtype=torch.float16),
        1.0,
    )
    full_packed = pack_i4(codes.contiguous()).contiguous()
    q = torch.randint(
        0, 256, (rows, padded_in // 2), device=device, dtype=torch.uint8
    )
    dense_ms = event_time(lambda: module.matmul(q, full_packed))
    sparse_ms = event_time(
        lambda: module.sparse_matmul(
            weight.packed, weight.meta, q, out_features, rows, padded_in
        )
    )
    print(
        f"gemm O={out_features} K={padded_in} R={rows}: "
        f"dense_int4={dense_ms * 1000:.1f}us "
        f"sparse_int4={sparse_ms * 1000:.1f}us "
        f"speedup={dense_ms / sparse_ms:.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    check(128, 896, args.rows, args.device)
    check(4864, 896, args.rows, args.device)
    check(4864, 4864, args.rows, args.device)
    for shape in ((4864, 896, 512), (4864, 4864, 512), (896, 4864, 32)):
        bench_shape(*shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
