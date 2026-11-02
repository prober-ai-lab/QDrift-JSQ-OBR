# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness and microbenchmark check for Marlin / Sparse-Marlin.

Run on a CUDA host:

    python -m benchmark.kernels.checks.marlin_check

This validates INT4 W4A16 Marlin and 2:4 sparse INT4 Sparse-Marlin against an
FP16 reference and reports GEMM latency for typical LLM shapes.
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kernels.marlin import build_marlin_weight


def mask_2to4(rows: int, cols: int, device: str) -> torch.Tensor:
    """Keep an arbitrary two of every four columns (standard 2:4)."""
    if cols % 4:
        raise ValueError("cols must be divisible by four")
    scores = torch.rand(rows, cols // 4, 4, device=device)
    keep = scores.topk(2, dim=2).indices
    mask = torch.zeros(rows, cols // 4, 4, dtype=torch.bool, device=device)
    mask.scatter_(2, keep, True)
    return mask.reshape(rows, cols)


def event_time(fn, warmup: int = 5, iters: int = 50) -> float:
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


def make_weight(out_features: int, in_features: int, group: int = 128):
    scales = (torch.rand(out_features, in_features // group, device="cuda") * 0.01 + 1e-3).half()
    codes = torch.randint(-8, 8, (out_features, in_features), device="cuda", dtype=torch.int8)
    mask = mask_2to4(out_features, in_features, "cuda")
    codes = codes * mask
    weight = (codes.float() * scales.repeat_interleave(group, dim=1)).half()
    return weight, scales, mask, codes


def check(out_features: int, in_features: int, rows: int) -> None:
    weight, scales, mask, _ = make_weight(out_features, in_features)
    x = torch.randn(rows, in_features, device="cuda").half()
    reference = x @ weight.T
    for sparse in (False, True):
        layer = build_marlin_weight(
            weight, scales, 128, sparse=sparse, mask=mask if sparse else None
        )
        output = layer.forward(x)
        error = (output.float() - reference.float()).abs()
        print(
            f"correctness O={out_features} K={in_features} R={rows} "
            f"sparse={sparse}: max={error.max().item():.5f} "
            f"mean={error.mean().item():.6f}"
        )
        if error.max().item() > 0.05:
            raise SystemExit("Marlin kernel mismatch")


def bench(out_features: int, in_features: int) -> None:
    weight, scales, mask, _ = make_weight(out_features, in_features)
    dense = build_marlin_weight(weight, scales, 128, sparse=False)
    sparse = build_marlin_weight(weight, scales, 128, sparse=True, mask=mask)
    for rows in (1, 16, 32, 512):
        x = torch.randn(rows, in_features, device="cuda").half()
        fp16 = event_time(lambda: x @ weight.T)
        dense_ms = event_time(lambda: dense.forward(x))
        sparse_ms = event_time(lambda: sparse.forward(x))
        print(
            f"gemm O={out_features} K={in_features} R={rows}: "
            f"fp16={fp16 * 1000:.1f}us "
            f"marlin={dense_ms * 1000:.1f}us "
            f"sparse_marlin={sparse_ms * 1000:.1f}us "
            f"sparse/fp16={sparse_ms / fp16:.2f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    check(4096, 4096, args.rows)
    check(11008, 4096, args.rows)
    check(4864, 896, args.rows)
    bench(4096, 4096)
    bench(11008, 4096)
    bench(4864, 896)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
