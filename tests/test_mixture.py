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
    if a == 0.0:
        assert diff is None
    else:
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
    drift = P.drift_weight(weight, diff) if diff is not None else None
    got = P._unconstrained_target(weight, hessian, drift)
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
    asym = P._process_full_layer(_layer_recipe(CompensationSpec(method="drift")), qparams, "l", weight, x, z, "cpu", True)
    assert torch.equal(mix.new_weight, obr.new_weight)
    clean = P._process_full_layer(
        _layer_recipe(CompensationSpec(method="drift", mix_asym=0.0, mix_clean=1.0)),
        qparams, "l", weight, z, z, "cpu", True,
    )
    assert torch.equal(clean.new_weight, obr.new_weight)
    assert not torch.allclose(asym.new_weight.float(), obr.new_weight.float(), atol=1e-6)


def test_variant_recipes_validate_with_a_mixture() -> None:
    """The fp16 baselines drop the mixture; every variant must still validate."""
    from qdrift.compress.recipe import variant_recipes

    recipe = CompressionRecipe(
        prune=PruneSpec(sparsity=0.5, structure="4:8"),
        compensation=CompensationSpec(method="drift", mix_asym=0.25, mix_clean=0.25),
        pairwise_fraction=1.0,
    )
    variants = variant_recipes(recipe)
    for variant in variants.values():
        variant.validate()
    assert variants["dense"].compensation.method == "none"
    assert variants["sparse"].compensation.method == "obr"
    assert variants["joint"].compensation.mix_asym == 0.25


def test_localized_mixture_scope() -> None:
    """``mix_blocks`` / ``mix_modules`` give the mixture inside the scope and
    OBR's (0, 0) outside; the defaults are the global mixture."""
    attn, gate, down = "model.layers.{}.self_attn.q_proj", "model.layers.{}.mlp.gate_proj", "model.layers.{}.mlp.down_proj"
    full = CompensationSpec(method="drift")
    assert full.is_global_mix() and full.mix_for(None) == (1.0, 0.0)
    early = CompensationSpec(method="drift", mix_blocks=(0, 7))
    assert early.mix_for(attn.format(7)) == (1.0, 0.0)
    assert early.mix_for(attn.format(8)) == (0.0, 0.0)
    assert early.mix_for(down.format(31)) == (0.0, 0.0)
    mlp_in = CompensationSpec(method="drift", mix_asym=0.25, mix_modules="mlp_in")
    assert mlp_in.mix_for(gate.format(3)) == (0.25, 0.0)
    assert mlp_in.mix_for(gate.format(3).replace("gate", "up")) == (0.25, 0.0)
    assert mlp_in.mix_for(down.format(3)) == (0.0, 0.0)
    assert mlp_in.mix_for(attn.format(3)) == (0.0, 0.0)
    attn_only = CompensationSpec(method="drift", mix_modules="attn")
    assert attn_only.mix_for("model.layers.0.self_attn.o_proj") == (1.0, 0.0)
    assert attn_only.mix_for(down.format(0)) == (0.0, 0.0)
    for comp, tag in (
        (early, "_qdrift-L0-7_seed42"),
        (CompensationSpec(method="drift", mix_modules="attn"), "_qdrift-Mattn_seed42"),
        (CompensationSpec(method="drift", mix_asym=0.25, mix_blocks=(8, 31)), "_qdrift-a0.25c0-L8-31_seed42"),
    ):
        recipe = CompressionRecipe(compensation=comp)
        recipe.validate()
        assert run_name(recipe).endswith(tag)
        assert CompensationSpec.from_dict(comp.to_dict()) == comp
    assert "mix_blocks" not in full.to_dict() and "mix_modules" not in full.to_dict()
    for bad in (
        CompensationSpec(method="obr", mix_blocks=(0, 3)),
        CompensationSpec(method="drift", mix_blocks=(5, 2)),
        CompensationSpec(method="drift", mix_modules="mlp"),
    ):
        with pytest.raises(ValueError):
            bad.validate()
    # Layers sharing one input must share one mixture.
    recipe = CompressionRecipe(compensation=mlp_in)
    assert P._mix(recipe, [gate.format(1), gate.format(1).replace("gate", "up")]) == (0.25, 0.0)
    with pytest.raises(ValueError):
        P._mix(recipe, [gate.format(1), down.format(1)])


def test_localized_mixture_layers(monkeypatch) -> None:
    """In scope the layer equals the global mixture's, out of scope OBR's."""
    monkeypatch.setattr(P.torch.cuda, "set_device", lambda device: None)
    x, z = _inputs(rows=256, cols=64, seed=5)
    weight = torch.randn(16, 64, generator=torch.Generator().manual_seed(7))
    qparams = qrange(4, "midtread")
    inside, outside = "model.layers.2.mlp.down_proj", "model.layers.9.mlp.down_proj"
    obr = P._process_full_layer(_layer_recipe(CompensationSpec(method="obr")), qparams, outside, weight, z, None, "cpu", True)
    glob = P._process_full_layer(_layer_recipe(CompensationSpec(method="drift")), qparams, inside, weight, x, z, "cpu", True)
    local = _layer_recipe(CompensationSpec(method="drift", mix_blocks=(0, 3)))
    assert torch.equal(P._process_full_layer(local, qparams, inside, weight, x, z, "cpu", True).new_weight, glob.new_weight)
    assert torch.equal(P._process_full_layer(local, qparams, outside, weight, x, z, "cpu", True).new_weight, obr.new_weight)


def test_auto_rule_names_and_validation() -> None:
    comp = CompensationSpec(method="drift", mix_rule="auto")
    recipe = CompressionRecipe(compensation=comp)
    recipe.validate()
    assert run_name(recipe).endswith("_qdrift-auto_seed42")
    assert CompensationSpec.from_dict(comp.to_dict()) == comp
    assert "mix_rule" not in CompensationSpec(method="drift").to_dict()
    with pytest.raises(ValueError):
        comp.mix_for("model.layers.0.self_attn.q_proj")
    table = replace_table(comp, (("l", 0.3, 0.1),))
    assert table.mix_for("l") == (0.3, 0.1)
    for bad in (
        CompensationSpec(method="obr", mix_rule="auto"),
        CompensationSpec(method="drift", mix_rule="auto", mix_asym=0.25),
        CompensationSpec(method="drift", mix_rule="auto", mix_blocks=(0, 3)),
        CompensationSpec(method="drift", mix_rule="nope"),
    ):
        with pytest.raises(ValueError):
            bad.validate()


def replace_table(comp, rows):
    from dataclasses import replace

    return replace(comp, mix_table=rows)


def test_trust_region_and_shrinkage() -> None:
    from qdrift.obr.drift import auto_mixture, ledoit_wolf_intensity, trust_region_weight

    assert trust_region_weight(0.0, 1.0) == 1.0
    assert trust_region_weight(3.0, 1.0) == pytest.approx(0.5)
    alpha = trust_region_weight(0.45, 0.03)
    assert alpha ** 2 * 0.45 == pytest.approx((1 - alpha ** 2) * 0.03)
    x, z = _inputs(rows=4096, cols=16, seed=3)
    h_hat = z.t() @ z / z.shape[0]
    half = z[:2048].t() @ z[:2048] / 2048
    h_fp = x.t() @ x / x.shape[0]
    lw = ledoit_wolf_intensity(h_hat, half, h_fp)
    assert 0.0 < lw < 1.0
    # More drift (a larger H_Xhat - H_X) shrinks less.
    assert ledoit_wolf_intensity(h_hat, half, 0.5 * h_fp) < lw
    assert ledoit_wolf_intensity(h_hat, half, h_hat) == 1.0
    a, g = auto_mixture(0.4, 0.25)
    assert (a, g) == (0.4, pytest.approx(0.15)) and 1 - a - g > 0


def test_auto_mix_group(monkeypatch) -> None:
    """The derived mixture is interior and matches the closed forms."""
    monkeypatch.setattr(P.torch.cuda, "set_device", lambda device: None)
    x, z = _inputs(rows=512, cols=64, seed=11)
    generator = torch.Generator().manual_seed(2)

    class _Layer:
        def __init__(self, rows):
            self.weight = type("W", (), {"data": torch.randn(rows, 64, generator=generator)})()

    layers = {"a": _Layer(16), "b": _Layer(8)}
    recipe = _layer_recipe(CompensationSpec(method="drift", mix_rule="auto"))
    qparams = qrange(4, "midtread")
    (alpha, gamma), info = P._auto_mix(recipe, qparams, ["a", "b"], layers, x, z, "cpu")
    assert 0.0 < alpha < 1.0 and 0.0 < gamma and alpha + gamma < 1.0
    h = z.t() @ z / z.shape[0]
    d = (z - x).t() @ z / z.shape[0]
    system = P._damped(h, recipe.compensation.damping)
    expected = sum(
        float(torch.trace(layers[n].weight.data @ d @ torch.linalg.solve(system, d.t() @ layers[n].weight.data.t())))
        for n in ("a", "b")
    )
    assert info["cancellable"] == pytest.approx(expected, rel=1e-4)
    assert alpha ** 2 * info["cancellable"] == pytest.approx((1 - alpha ** 2) * info["error"], rel=1e-6)
    # The reference is OBR's own layer.
    obr = P._process_full_layer(_layer_recipe(CompensationSpec(method="obr")), qparams, "a", layers["a"].weight.data, z, None, "cpu", True)
    delta = obr.new_weight.float() - layers["a"].weight.data
    assert float(((delta @ h) * delta).sum()) < info["error"]


def _cli_recipe(*flags: str) -> CompressionRecipe:
    import argparse

    import main as cli

    parser = argparse.ArgumentParser()
    cli._add_common_arguments(parser)
    recipe = cli._build_recipe(parser.parse_args(list(flags)))
    recipe.validate()
    return recipe


def test_cli_defaults_to_auto() -> None:
    """``--comp drift`` alone derives the weights; fixed weights must be asked
    for, so the stored run names (and recipe hashes) of the grid, the pure
    asymmetric target and OBR stay unchanged."""
    auto = _cli_recipe("--comp", "drift")
    assert auto.compensation.mix_rule == "auto"
    assert run_name(auto).endswith("_qdrift-auto_seed42")
    assert run_name(_cli_recipe("--comp", "obr")).endswith("_seq_seed42")
    pure = _cli_recipe("--comp", "drift", "--mix-rule", "fixed")
    assert run_name(pure) == run_name(CompressionRecipe(compensation=CompensationSpec(method="drift")))
    assert pure.recipe_hash() == CompressionRecipe(compensation=CompensationSpec(method="drift")).recipe_hash()
    grid = _cli_recipe("--comp", "drift", "--mix-rule", "fixed", "--mix-asym", "0.25", "--mix-clean", "0")
    assert run_name(grid).endswith("_qdrift-a0.25c0_seed42")
    local = _cli_recipe("--comp", "drift", "--mix-rule", "fixed", "--mix-blocks", "0-7")
    assert run_name(local).endswith("_qdrift-L0-7_seed42")
    for flags in (("--mix-asym", "1"), ("--mix-clean", "0"), ("--mix-blocks", "0-3"), ("--mix-modules", "attn")):
        with pytest.raises(SystemExit):
            _cli_recipe("--comp", "drift", *flags)
