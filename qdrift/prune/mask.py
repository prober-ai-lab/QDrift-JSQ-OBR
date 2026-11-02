# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured and unstructured pruning masks.

Terminology:

* ``2:4``  -- keep any two of every four consecutive input columns.
* ``4:8``  -- OBR-style standard N:M: keep any four of every eight columns.
* ``2:4_pairwise`` -- INT4 sparse Tensor Core layout: keep one adjacent
  2-element pair out of every four columns.  The kernel packs two 4-column
  groups per metadata word, so this is also called the hardware 4:8 layout.
* ``unstructured`` -- keep the top-k scoring weights per output row.
"""

from __future__ import annotations

import torch


def mask_2to4(score: torch.Tensor) -> torch.Tensor:
    """Keep the two highest-scoring entries in every group of four."""
    rows, cols = score.shape
    if cols % 4 != 0:
        raise ValueError("2:4 pruning requires an input width divisible by 4")
    grouped = score.reshape(rows, cols // 4, 4)
    indices = torch.topk(grouped, 2, dim=2).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    return mask.reshape(rows, cols)


def mask_4to8(score: torch.Tensor) -> torch.Tensor:
    """Keep any four highest-scoring entries in every group of eight.

    This is the N:M semantics used by the OBR paper.  It is a strict superset
    of the hardware ``2:4_pairwise`` layout below.
    """
    rows, cols = score.shape
    if cols % 8 != 0:
        raise ValueError("4:8 pruning requires a width divisible by 8")
    grouped = score.reshape(rows, cols // 8, 8)
    indices = torch.topk(grouped, 4, dim=2).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    return mask.reshape(rows, cols)


def mask_2to4_pairwise(score: torch.Tensor) -> torch.Tensor:
    """Keep one adjacent pair in each group of four columns.

    This is the hardware INT4 sparse Tensor Core pattern.  CUTLASS packs two
    such 4-column groups per metadata word, so the physical metadata layout is
    often called 4:8; semantically this is still 2 of 4 columns.
    """
    rows, cols = score.shape
    if cols % 8 != 0:
        raise ValueError("2:4_pairwise pruning requires a width divisible by 8")
    grouped = score.reshape(rows, cols // 8, 4, 2)
    pair_score = grouped.sum(dim=3)
    low = pair_score[:, :, :2].argmax(dim=2)
    high = pair_score[:, :, 2:].argmax(dim=2) + 2
    pair_mask = torch.zeros_like(pair_score, dtype=torch.bool)
    pair_mask.scatter_(2, low.unsqueeze(2), True)
    pair_mask.scatter_(2, high.unsqueeze(2), True)
    return pair_mask.unsqueeze(3).expand(-1, -1, -1, 2).reshape(rows, cols)


def topk_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the highest-scoring entries in each output row."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    rows, cols = score.shape
    keep = max(1, cols - int(cols * sparsity))
    indices = torch.topk(score, keep, dim=1).indices
    mask = torch.zeros(rows, cols, dtype=torch.bool, device=score.device)
    mask.scatter_(1, indices, True)
    return mask


def mask_from_score(
    score: torch.Tensor,
    sparsity: float = 0.5,
    structure: str = "2:4",
) -> torch.Tensor:
    """Build a Boolean retained mask from a score matrix."""
    if structure == "unstructured":
        return topk_mask(score, sparsity)
    if structure == "2:4":
        return mask_2to4(score)
    if structure == "4:8":
        return mask_4to8(score)
    if structure == "2:4_pairwise":
        return mask_2to4_pairwise(score)
    if structure == "none":
        return torch.ones_like(score, dtype=torch.bool)
    raise ValueError(f"unknown pruning structure {structure}")
