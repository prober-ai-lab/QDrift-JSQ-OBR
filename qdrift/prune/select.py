# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N:M and pairwise masks chosen by their exact compensated cost.

Each row of the compensation objective is ``(v - w*) H (v - w*)^T``, ``w*``
the row's unconstrained optimum (``w`` for OBR, ``w - w D H^-1`` for DAC).
Pruning the set ``P`` with the rest of the row compensated costs
``w*_P [G_PP]^-1 w*_P^T``, ``G = H^-1`` (multi-weight OBS).  The blocks of
``g`` columns are decided left to right with the columns before the block
frozen and every later column free, so each decision is exact: with
``U = chol(G)`` upper (``G = U^T U``), the inverse Hessian of the columns from
``c`` on is ``U_{c:,c:}^T U_{c:,c:}``, whose leading ``g x g`` block is
``B = U_bb^T U_bb``, shared by all rows.  Pattern ``p`` (its pruned columns)
costs ``w_p [B_pp]^-1 w_p^T`` on the current row ``w``, and the chosen one
updates the rest of the row by ``w_{c:} -= (a U_bb^T) U_{b,c:}``,
``a = w_p [B_pp]^-1`` scattered onto the block.  See docs/method-derivation.md.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations

import torch

__all__ = ["block_patterns", "obs_mask"]


@cache
def block_patterns(structure: str) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """``(g, kept column sets)`` of the legal patterns of one ``g``-column block."""
    if structure == "2:4":
        return 4, tuple(combinations(range(4), 2))
    if structure == "4:8":
        return 8, tuple(combinations(range(8), 4))
    if structure == "2:4_pairwise":
        return 8, tuple((2 * a, 2 * a + 1, 2 * b, 2 * b + 1) for a in (0, 1) for b in (2, 3))
    if structure == "4:8_pairwise":
        return 8, tuple(
            (2 * a, 2 * a + 1, 2 * b, 2 * b + 1) for a, b in combinations(range(4), 2)
        )
    raise ValueError(f"no block patterns for structure {structure}")


@torch.no_grad()
def obs_mask(
    target: torch.Tensor,
    system: torch.Tensor,
    structure: str,
    blocksize: int = 128,
) -> torch.Tensor:
    """Retained mask of ``structure`` for the rows ``target`` (``w*``) under the
    damped Hessian ``system``; GPTQ-style lazy updates every ``blocksize``
    columns."""
    rows, cols = target.shape
    g, kept = block_patterns(structure)
    if cols % g or blocksize % g:
        raise ValueError(f"{structure} needs widths divisible by {g}")
    device = target.device
    pruned = torch.tensor(
        [[j for j in range(g) if j not in keep] for keep in kept], device=device
    )
    patterns = torch.ones(len(kept), g, dtype=torch.bool, device=device)
    patterns.scatter_(1, pruned, False)
    upper = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(system.float())), upper=True
    )
    w = target.float().clone()
    mask = torch.empty(rows, cols, dtype=torch.bool, device=device)
    arange = torch.arange(rows, device=device)
    for start in range(0, cols, blocksize):
        end = min(start + blocksize, cols)
        err = torch.zeros(rows, end - start, device=device)
        for c in range(start, end, g):
            u = upper[c : c + g, c : c + g]
            b = u.t() @ u
            inverse = torch.linalg.inv(b[pruned[:, :, None], pruned[:, None, :]])
            wp = w[:, c : c + g][:, pruned]  # rows x patterns x k
            a = torch.einsum("rpk,pkl->rpl", wp, inverse)
            choice = (a * wp).sum(dim=2).argmin(dim=1)
            coef = torch.zeros(rows, g, device=device)
            coef.scatter_(1, pruned[choice], a[arange, choice])
            mask[:, c : c + g] = patterns[choice]
            e = coef @ u.t()
            w[:, c:end] -= e @ upper[c : c + g, c:end]
            err[:, c - start : c - start + g] = e
        if end < cols:
            w[:, end:] -= err @ upper[start:end, end:]
    return mask
