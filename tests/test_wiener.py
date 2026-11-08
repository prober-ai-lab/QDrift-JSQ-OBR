# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wiener mechanism against a brute-force objective on samples.

Per row the objective is ``f(v) = mean_t (v^T z_t - w^T x_t)^2 + lambda |v|^2``
with compressed-path inputs ``z`` and full-precision inputs ``x`` (``z = x``
without QDrift).  Every reference below is built from the samples alone, in
float64, independently of the implementation.

    python -m pytest tests/test_wiener.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress import pipeline  # noqa: E402
from qdrift.compress.recipe import (  # noqa: E402
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    WeightQuantSpec,
)
from qdrift.mechanisms.wiener import noise_damping, wiener_target  # noqa: E402
from qdrift.prune.mask import mask_from_score  # noqa: E402
from qdrift.quant.activation import fake_quantize_activations  # noqa: E402
from qdrift.quant.quantize import fake_quantize, qrange  # noqa: E402
from qdrift.quant.scale import compute_scale  # noqa: E402


def problem(drift: bool, seed: int = 0) -> dict:
    """A small correlated layer; ``lam`` is 5% of mean(diag H)."""
    generator = torch.Generator().manual_seed(seed)
    rows, cols, tokens = 8, 32, 6000
    basis = torch.randn(cols, 12, generator=generator, dtype=torch.float64)
    x = torch.randn(tokens, 12, generator=generator, dtype=torch.float64) @ basis.t()
    x += 0.2 * torch.randn(tokens, cols, generator=generator, dtype=torch.float64)
    z = x + 0.15 * torch.randn(tokens, cols, generator=generator, dtype=torch.float64)
    if not drift:
        z = x
    weight = 0.05 * torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    hessian = z.t() @ z / tokens
    lam = 0.05 * float(torch.diagonal(hessian).mean())
    mask = mask_from_score(weight.abs(), 0.5, "unstructured")
    return {"x": x, "z": z, "w": weight, "mask": mask, "lam": lam, "drift": drift}


def objective(p: dict, v: torch.Tensor) -> torch.Tensor:
    """Per-row ``f(v)`` from the samples."""
    out = p["z"] @ v.t() - p["x"] @ p["w"].t()
    return out.pow(2).mean(0) + p["lam"] * v.pow(2).sum(1)


def brute_target(p: dict) -> torch.Tensor:
    """``argmin f`` over ``v`` with ``v_P = 0``, one row at a time."""
    tokens = p["x"].shape[0]
    hessian = p["z"].t() @ p["z"] / tokens
    cross = p["z"].t() @ p["x"] / tokens
    target = torch.zeros_like(p["w"])
    for row in range(p["w"].shape[0]):
        keep = torch.nonzero(p["mask"][row]).squeeze(1)
        system = hessian[keep][:, keep] + p["lam"] * torch.eye(
            len(keep), dtype=torch.float64
        )
        target[row, keep] = torch.linalg.solve(system, (cross @ p["w"][row])[keep])
    return target


def stats32(p: dict) -> dict:
    """The pipeline's float32 Hessian, drift matrix and damped system."""
    hat = p["z"].float() if p["drift"] else None
    hessian, system, diff = pipeline._hessians(p["x"].float(), hat, 0.01)
    return {"hessian": hessian, "system": system, "diff": diff}


@pytest.mark.parametrize("drift", [False, True])
def test_target_is_the_constrained_minimizer(drift: bool) -> None:
    """``wiener_target`` returns ``argmin f`` subject to ``v_P = 0``."""
    p = problem(drift)
    s = stats32(p)
    state, _ = wiener_target(
        p["w"].float(), p["mask"], s["hessian"], p["lam"], s["diff"]
    )
    target = torch.zeros_like(p["w"])
    target.scatter_(1, state.keep, state.retained.double())
    expected = brute_target(p)
    assert torch.allclose(target, expected, rtol=1e-3, atol=1e-5)
    # Wiener shrinks: it is not OBR's update w_K + delta with another lambda.
    assert not torch.allclose(target[p["mask"]], p["w"][p["mask"]], rtol=0.05)


@pytest.mark.parametrize("drift", [False, True])
def test_objective_is_quadratic_around_target(drift: bool) -> None:
    """On ``v_P = 0``: ``f(v) = (v - w*)^T (H + lambda I) (v - w*) + const``."""
    p = problem(drift)
    target = brute_target(p)
    hessian = p["z"].t() @ p["z"] / p["z"].shape[0]
    system = hessian + p["lam"] * torch.eye(hessian.shape[0], dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    offsets = []
    for _ in range(4):
        v = target + 0.05 * torch.randn(
            target.shape, generator=generator, dtype=torch.float64
        )
        v = v * p["mask"]
        quad = ((v - target) @ system * (v - target)).sum(1)
        offsets.append(objective(p, v) - quad)
    for offset in offsets[1:]:
        assert torch.allclose(offset, offsets[0], rtol=1e-9, atol=1e-12)


def recipe_for(method: str, quant: str) -> CompressionRecipe:
    """W4 per-channel mid-tread, 50% unstructured, wiener."""
    return CompressionRecipe(
        weight=WeightQuantSpec(
            bits=4,
            granularity=GranularitySpec("channel", None),
            grid="midtread",
            range="clip",
            method=quant,
        ),
        prune=PruneSpec(sparsity=0.5, structure="unstructured", criterion="magnitude"),
        compensation=CompensationSpec(method=method, mechanism="wiener"),
        pairwise_fraction=0.0,
    )


def brute_chain(p: dict, scale: torch.Tensor, quant: str) -> torch.Tensor:
    """Target, early rounding, exact late solve, then RTN or sequential GPTQ."""
    qparams = qrange(4, "midtread")
    rows, cols = p["w"].shape
    cut = cols // 2
    tokens = p["x"].shape[0]
    hessian = p["z"].t() @ p["z"] / tokens
    cross = p["z"].t() @ p["x"] / tokens
    system = hessian + p["lam"] * torch.eye(cols, dtype=torch.float64)
    target = brute_target(p)
    out = torch.zeros_like(target)
    for row in range(rows):
        keep = p["mask"][row]
        early = torch.nonzero(keep[:cut]).squeeze(1)
        late = cut + torch.nonzero(keep[cut:]).squeeze(1)
        v = torch.zeros(cols, dtype=torch.float64)
        v[early] = fake_quantize(
            target[row, early], scale[row, early], qparams, "midtread"
        )
        # argmin f over v_L with v_E rounded and v_P = 0.
        rhs = (cross @ p["w"][row])[late] - hessian[late][:, early] @ v[early]
        v[late] = torch.linalg.solve(system[late][:, late], rhs)
        if quant == "rtn":
            v = fake_quantize(v, scale[row], qparams, "midtread") * keep
        else:
            for col in range(cut, cols):
                rounded = fake_quantize(
                    v[col : col + 1], scale[row, col : col + 1], qparams, "midtread"
                )
                goal = rounded[0] if keep[col] else torch.zeros((), dtype=torch.float64)
                error = goal - v[col]
                v[col] = goal
                if col + 1 < cols:
                    rest = slice(col + 1, cols)
                    v[rest] -= (
                        torch.linalg.solve(system[rest, rest], system[rest, col])
                        * error
                    )
        out[row] = v
    return out


@pytest.mark.parametrize("method", ["obr", "drift"])
@pytest.mark.parametrize("quant", ["rtn", "gptq"])
def test_pipeline_quantization_stage(method: str, quant: str) -> None:
    """Early solve and late GPTQ both use ``H + lambda I`` around ``w*``."""
    p = problem(method == "drift")
    s = stats32(p)
    recipe = recipe_for(method, quant)
    weight = p["w"].float()
    result = pipeline._quantize_layer(
        recipe,
        qrange(4, "midtread"),
        "layer",
        weight,
        weight,
        s["hessian"],
        s["system"],
        s["system"],
        s["diff"],
        act_noise=p["lam"],
    )
    assert result.meta["mechanism"] == "wiener" and not result.meta["floor"]
    assert result.meta["lambda"] == pytest.approx(p["lam"])
    scale = compute_scale(
        weight,
        4,
        "midtread",
        "channel",
        None,
        recipe.weight.effective_rhos(),
        mse_clip=True,
    ).scale.double()
    expected = brute_chain(p, scale, quant)
    got = result.new_weight.double()
    codes_got = torch.round(got / scale)
    codes_expected = torch.round(expected / scale)
    assert float((codes_got == codes_expected).double().mean()) >= 0.98
    assert torch.allclose(objective(p, got), objective(p, expected), rtol=2e-2)


def test_noise_damping_floor() -> None:
    """``lambda_eff = max(sigma_a^2, 1e-4 mean diag H)`` and the floor flag."""
    assert noise_damping(1.0, 0.5) == (0.5, False)
    assert noise_damping(1.0, 1e-6) == (1e-4, True)
    assert noise_damping(1.0, None) == (1e-4, True)
    assert noise_damping(1.0, 0.5, 2.0) == (1.0, False)


@pytest.mark.parametrize("grid", ["midtread", "midrise"])
def test_activation_quantizer_matches_runtime(grid: str) -> None:
    """The compression-side A4 model equals the evaluation runtime's."""
    runtime = pytest.importorskip("benchmark.kernels.lib.int8_linear")
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 256, generator=generator)
    x[:, 3] *= 20.0
    for clip in (1.0, 0.9):
        codes, scales, group, offset = runtime.quantize_activations(
            x, 4, grid, "token", None, clip
        )
        expect = (codes.float() + offset) * scales.repeat_interleave(group, dim=1)
        got = fake_quantize_activations(x, 4, grid, "token", None, clip)
        assert torch.allclose(got, expect, rtol=1e-6, atol=1e-6)
