# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAC on the deployed input ``x_hat = Q_a(z)``: ``H = E[x_hat x_hat^T]`` and
``D = E[(x_hat - x) x_hat^T]``.

    python -m pytest tests/test_dac_hessian.py
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
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
)
from qdrift.compress.runs import run_name  # noqa: E402
from qdrift.obr.drift import drift_weight, shrink_drift_weight  # noqa: E402
from qdrift.quant.activation import fake_quantize_activations  # noqa: E402


def _inputs(rows=300, cols=24):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(rows, cols, generator=generator)
    z = x + 0.2 * torch.randn(rows, cols, generator=generator)
    return x, z


def test_quantizer_only_for_dac_with_quantized_activations() -> None:
    assert P._hessian_quantizer(CompressionRecipe(compensation=CompensationSpec(method="obr"))) is None
    assert P._hessian_quantizer(
        CompressionRecipe(
            activation=ActivationQuantSpec(bits=16), compensation=CompensationSpec(method="drift")
        )
    ) is None
    assert P._hessian_quantizer(
        CompressionRecipe(compensation=CompensationSpec(method="drift", mechanism="qir"))
    ) is None
    assert P._hessian_quantizer(
        CompressionRecipe(compensation=CompensationSpec(method="drift", drift_act="none"))
    ) is None
    for mode in ("prefix", "layer"):
        assert P._hessian_quantizer(
            CompressionRecipe(compensation=CompensationSpec(method="drift", drift_act=mode))
        ) is not None


@pytest.mark.parametrize("streaming", [False, True])
def test_hessian_and_drift_on_quantized_input(streaming, monkeypatch) -> None:
    x, z = _inputs()
    recipe = CompressionRecipe(compensation=CompensationSpec(method="drift", drift_act="layer"))
    quant = P._hessian_quantizer(recipe)
    if streaming:
        monkeypatch.setattr(P, "_HESSIAN_STREAM_ROWS", 64)
        mask_h, system, diff = P._hessians_streaming(x, z, 0.0, "cpu", quant)
    else:
        mask_h, system, diff = P._hessians(x, z, 0.0, quant)
    act = recipe.activation
    zq = fake_quantize_activations(z, act.bits, act.grid, act.granularity.kind)
    assert not torch.allclose(zq, z)
    assert torch.allclose(mask_h, zq.t() @ zq / len(z), atol=1e-5)
    assert torch.allclose(system, zq.t() @ zq / len(z), atol=1e-5)
    assert torch.allclose(diff, (zq - x).t() @ zq / len(z), atol=1e-5)


def test_shrinkage_uses_quantized_input() -> None:
    x, z = _inputs(rows=320)
    recipe = CompressionRecipe(
        compensation=CompensationSpec(method="drift", drift_act="layer", drift_shrink="row")
    )
    quant = P._hessian_quantizer(recipe)
    zq = quant(z)
    weight = torch.randn(6, z.shape[1], generator=torch.Generator().manual_seed(1))
    shift = drift_weight(weight, (zq - x).t() @ zq / len(z))
    got, factor = shrink_drift_weight(shift, weight, x, z, 64, "row", quant)
    want, want_factor = shrink_drift_weight(shift, weight, x, zq, 64, "row")
    assert torch.allclose(factor, want_factor) and torch.allclose(got, want)


def test_drift_act_is_named_and_round_trips() -> None:
    for mode, tag in (("none", "_qdrift_seed42"), ("layer", "_qdrift-alayer_seed42"), ("prefix", "_qdrift-aprefix_seed42")):
        comp = CompensationSpec(method="drift", drift_act=mode)
        recipe = CompressionRecipe(compensation=comp)
        recipe.validate()
        assert run_name(recipe).endswith(tag)
        assert CompensationSpec.from_dict(comp.to_dict()) == comp
    # Records written before drift_act are the original DAC, the default.
    assert "drift_act" not in CompensationSpec(method="drift").to_dict()
    assert CompensationSpec.from_dict({"method": "drift"}).drift_act == "none"
    w2 = CompressionRecipe(activation=ActivationQuantSpec(bits=16), compensation=CompensationSpec(method="drift", drift_act="layer"))
    assert run_name(w2).endswith("_qdrift_seed42")
