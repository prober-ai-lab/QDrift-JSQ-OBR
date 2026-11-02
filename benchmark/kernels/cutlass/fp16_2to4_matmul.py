# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""2:4 semi-structured sparse matmul through PyTorch's CUTLASS backend."""

from __future__ import annotations

import torch


def make_semi_structured(weight: torch.Tensor) -> torch.Tensor:
    """Compress a dense 2:4 fp16/bf16 weight matrix for sparse matmul.

    The CUTLASS backend is used because it exposes the 2:4 Tensor Core path on
    Ampere and Ada GPUs without cuSPARSELt plan-search overhead.
    """
    if weight.ndim != 2:
        raise ValueError("semi-structured weight must be a matrix")
    if not weight.is_contiguous():
        weight = weight.contiguous()
    previous = torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS
    torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        return torch.sparse.to_sparse_semi_structured(weight)
    finally:
        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = previous


def semi_structured_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``linear(x, weight, bias)`` with a 2:4 semi-structured weight."""
    return torch.nn.functional.linear(x, weight, bias)
