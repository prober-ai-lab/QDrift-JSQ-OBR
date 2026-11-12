# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QDrift's mixture objective alpha (a) + beta (b) + gamma (c).

    python -m pytest tests/test_mixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress import pipeline as P  # noqa: E402
from qdrift.compress.recipe import (  # noqa: E402
    CompensationSpec,
    CompressionRecipe,
    PruneSpec,
    WeightQuantSpec,
    GranularitySpec,
)
from qdrift.compress.runs import run_name  # noqa: E402
from qdrift.quant.quantize import qrange  # noqa: E402


def _inputs(rows=512, cols=32, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=generator, dtype=dtype)
    x = x @ (0.3 * torch.randn(cols, cols, generator=generator, dtype=dtype) + torch.eye(cols, dtype=dtype))
    z = x + 0.3 * torch.randn(rows, cols, generator=generator, dtype=dtype) @ (
        0.5 * torch.eye(cols, dtype=dtype) + 0.2 * torch.randn(cols, cols, generator=generator, dtype=dtype)
    )
    return x, z


def test_names_and_validation() -> None:
    for comp, tag in (
        (CompensationSpec(method="obr"), "_seq_seed42"),
        (CompensationSpec(method="drift"), "_qdrift_seed42"),
        (CompensationSpec(method="drift", mix_asym=0.5, mix_clean=0.25), "_qdrift-a0.5c0.25_seed42"),
        (CompensationSpec(method="drift", mix_asym=0.0, mix_clean=0.0), "_qdrift-a0c0_seed42"),
    ):
        recipe = CompressionRecipe(compensation=comp)
        recipe.validate()
        assert run_name(recipe).endswith(tag)
        assert CompensationSpec.from_dict(comp.to_dict()) == comp
    assert "mix_asym" not in CompensationSpec(method="drift").to_dict()
    for bad in (
        CompensationSpec(method="drift", mix_asym=0.8, mix_clean=0.3),
        CompensationSpec(method="drift", mix_asym=-0.1),
        CompensationSpec(method="obr", mix_asym=0.5),
    ):
        with pytest.raises(ValueError):
            bad.validate()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mix", [(1.0, 0.0), (0.0, 0.0), (0.4, 0.35), (0.0, 1.0)])
def test_mixture_statistics(streaming, mix, monkeypatch) -> None:
    x, z = _inputs()
    a, c = mix
    if streaming:
        monkeypatch.setattr(P, "_HESSIAN_STREAM_ROWS", 100)
        hessian, system, diff = P._hessians_streaming(x, z, 0.0, "cpu", mix)
    else:
        hessian, system, diff = P._hessians(x, z, 0.0, mix)
    n = len(z)
    hz, hx, d = z.t() @ z / n, x.t() @ x / n, (z - x).t() @ z / n
    assert torch.allclose(hessian, (1 - c) * hz + c * hx, atol=1e-4)
    assert torch.allclose(system, hessian, atol=1e-4)
    assert torch.allclose(diff, a * d, atol=1e-4)


@pytest.mark.parametrize("mix", [(1.0, 0.0), (0.5, 0.2), (0.0, 0.6), (0.3, 0.7)])
def test_target_is_the_mixture_optimum(mix) -> None:
    """``w* = W - (alpha W D) H_mix^-1`` solves the normal equations of
    alpha (a) + beta (b) + gamma (c)."""
    x, z = _inputs(dtype=torch.float64)
    a, c = mix
    b = 1.0 - a - c
    weight = torch.randn(6, x.shape[1], generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    hessian, _, diff = P._hessians(x, z, 0.0, mix)
    got = P._unconstrained_target(weight, hessian, P.drift_weight(weight, diff))
    n = len(z)
    hz, hx, xz = z.t() @ z / n, x.t() @ x / n, x.t() @ z / n
    # d/dV: a (V hz - W xz) + b (V hz - W hz) + c (V hx - W hx) = 0
    lhs = (a + b) * hz + c * hx
    rhs = weight @ (a * xz + b * hz + c * hx)
    assert torch.allclose(got, torch.linalg.solve(lhs, rhs.t()).t(), atol=1e-8)


def _layer_recipe(comp: CompensationSpec) -> CompressionRecipe:
    return CompressionRecipe(
        weight=WeightQuantSpec(bits=4, granularity=GranularitySpec("channel", None), grid="midtread", method="gptq"),
        prune=PruneSpec(sparsity=0.5, structure="unstructured", criterion="sparsegpt"),
        compensation=comp,
    )


def test_obr_is_the_zero_mixture(monkeypatch) -> None:
    """With alpha = gamma = 0 the compressed layer is OBR's, whatever the drift."""
    monkeypatch.setattr(P.torch.cuda, "set_device", lambda device: None)
    x, z = _inputs(rows=256, cols=64, seed=5)
    weight = torch.randn(16, 64, generator=torch.Generator().manual_seed(7))
    qparams = qrange(4, "midtread")
    obr = P._process_full_layer(_layer_recipe(CompensationSpec(method="obr")), qparams, "l", weight, z, None, "cpu", True)
    mix = P._process_full_layer(
        _layer_recipe(CompensationSpec(method="drift", mix_asym=0.0, mix_clean=0.0)),
        qparams, "l", weight, x, z, "cpu", True,
    )
    dac = P._process_full_layer(_layer_recipe(CompensationSpec(method="drift")), qparams, "l", weight, x, z, "cpu", True)
    assert torch.allclose(mix.new_weight.float(), obr.new_weight.float(), atol=1e-6)
    assert not torch.allclose(dac.new_weight.float(), obr.new_weight.float(), atol=1e-6)
