# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact compensated-cost selection of N:M and pairwise masks.

    python -m pytest tests/test_mask_select.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress.pipeline import _unconstrained_target  # noqa: E402
from qdrift.compress.recipe import CompressionRecipe, PruneSpec  # noqa: E402
from qdrift.compress.runs import run_name  # noqa: E402
from qdrift.prune.mask import mask_from_score  # noqa: E402
from qdrift.prune.select import block_patterns, obs_mask  # noqa: E402

STRUCTURES = ["2:4", "4:8", "2:4_pairwise", "4:8_pairwise"]


def _problem(rows=5, cols=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(600, cols, generator=generator, dtype=torch.float64)
    x = x @ (0.4 * torch.randn(cols, cols, generator=generator, dtype=torch.float64) + torch.eye(cols, dtype=torch.float64))
    hessian = x.t() @ x / 600 + 0.01 * torch.eye(cols, dtype=torch.float64)
    return torch.randn(rows, cols, generator=generator, dtype=torch.float64), hessian


def _cost(w, hessian, pruned):
    """Exact OBS cost ``w_P [G_PP]^-1 w_P^T`` of pruning ``pruned`` from ``w``."""
    g = torch.linalg.inv(hessian)
    p = torch.tensor(pruned)
    return float(w[p] @ torch.linalg.solve(g[p][:, p], w[p]))


def _reference(w, hessian, structure):
    """Block-sequential selection with explicit trailing inverses."""
    g, kept = block_patterns(structure)
    w = w.clone()
    mask = torch.zeros_like(w, dtype=torch.bool)
    cols = w.numel()
    for c in range(0, cols, g):
        inv = torch.linalg.inv(hessian[c:, c:])
        best = None
        for keep in kept:
            p = [j for j in range(g) if j not in keep]
            cost = float(w[c:][p] @ torch.linalg.solve(inv[p][:, p], w[c:][p]))
            if best is None or cost < best[0]:
                best = (cost, keep, p)
        _, keep, p = best
        w[c:] -= torch.linalg.solve(inv[p][:, p], w[c:][p]) @ inv[p]
        mask[c + torch.tensor(keep)] = True
    return mask


@pytest.mark.parametrize("structure", STRUCTURES)
def test_matches_block_sequential_reference(structure) -> None:
    w, hessian = _problem()
    got = obs_mask(w, hessian, structure, blocksize=8)
    for row in range(w.shape[0]):
        assert torch.equal(got[row], _reference(w[row], hessian, structure))
    assert torch.equal(obs_mask(w, hessian, structure, blocksize=16), got)


@pytest.mark.parametrize("structure", STRUCTURES)
def test_mask_is_legal(structure) -> None:
    w, hessian = _problem(rows=7, cols=64, seed=1)
    got = obs_mask(w, hessian, structure)
    g, kept = block_patterns(structure)
    legal = {tuple(j in keep for j in range(g)) for keep in kept}
    blocks = got.reshape(-1, g)
    assert all(tuple(b.tolist()) in legal for b in blocks)


@pytest.mark.parametrize("structure", ["2:4", "4:8_pairwise"])
def test_single_block_is_the_exact_optimum(structure) -> None:
    g, kept = block_patterns(structure)
    w, hessian = _problem(rows=4, cols=g, seed=2)
    got = obs_mask(w, hessian, structure, blocksize=g)
    for row in range(w.shape[0]):
        costs = {keep: _cost(w[row], hessian, [j for j in range(g) if j not in keep]) for keep in kept}
        best = min(costs, key=costs.get)
        assert tuple(got[row].nonzero().squeeze(1).tolist()) == best


def test_never_worse_than_static_score_on_two_blocks() -> None:
    # Exhaustive search over two 4:8 pairwise blocks bounds both selections.
    w, hessian = _problem(rows=20, cols=16, seed=3)
    _, kept = block_patterns("4:8_pairwise")
    score = w.pow(2) / torch.diagonal(torch.linalg.inv(hessian))
    static = mask_from_score(score, 0.5, "4:8_pairwise")
    got = obs_mask(w, hessian, "4:8_pairwise", blocksize=8)
    total_got = total_static = total_best = 0.0
    for row in range(w.shape[0]):
        options = [
            [j for j in range(8) if j not in a] + [8 + j for j in range(8) if j not in b]
            for a, b in product(kept, kept)
        ]
        total_best += min(_cost(w[row], hessian, p) for p in options)
        total_got += _cost(w[row], hessian, (~got[row]).nonzero().squeeze(1).tolist())
        total_static += _cost(w[row], hessian, (~static[row]).nonzero().squeeze(1).tolist())
    assert total_best <= total_got + 1e-9 and total_got <= total_static


def test_dac_target_is_the_unconstrained_optimum() -> None:
    w, hessian = _problem(rows=3, cols=16, seed=4)
    shift = torch.randn(3, 16, dtype=torch.float64)
    target = _unconstrained_target(w, hessian, shift)
    # w* solves w* H = w H - W D.
    assert torch.allclose(target @ hessian, w @ hessian - shift, atol=1e-8)
    assert _unconstrained_target(w, hessian, None) is w


def test_select_is_named_and_round_trips() -> None:
    base = dict(structure="4:8", sparsity=0.5, criterion="sparsegpt")
    for select, frac, tag in (("score", 1.0, "_4to8p_"), ("obs", 1.0, "_4to8p-obs_"), ("score", 0.4, "_4to8p40_")):
        spec = PruneSpec(select=select, **base)
        recipe = CompressionRecipe(prune=spec, pairwise_fraction=frac)
        recipe.validate()
        assert tag in run_name(recipe)
        assert PruneSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        PruneSpec(structure="unstructured", select="obs").validate()
