# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAC's drift regression with a cross-validated ridge.

    python -m pytest tests/test_drift_ridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress.recipe import CompensationSpec, CompressionRecipe  # noqa: E402
from qdrift.compress.runs import run_name  # noqa: E402
from qdrift.obr.drift import heldout_errors, ridge_drift_weight  # noqa: E402

DTYPE = torch.float64


def _layer(rows=6, cols=12, sequences=16, seqlen=40, predictable=True, seed=0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(sequences * seqlen, cols, generator=generator, dtype=DTYPE)
    z = z @ (0.3 * torch.randn(cols, cols, generator=generator, dtype=DTYPE) + torch.eye(cols, dtype=DTYPE))
    noise = torch.randn(z.shape, generator=generator, dtype=DTYPE)
    if predictable:
        drift = z @ (0.2 * torch.randn(cols, cols, generator=generator, dtype=DTYPE)) + 0.05 * noise
    else:
        drift = noise
    weight = torch.randn(rows, cols, generator=generator, dtype=DTYPE)
    return weight, z - drift, z, seqlen


def test_heldout_errors_match_brute_force() -> None:
    weight, x, z, _ = _layer()
    train, held = slice(0, 400), slice(400, 640)
    zt, dt = z[train], (z - x)[train]
    zh, dh = z[held], (z - x)[held]
    h, d = zt.t() @ zt / len(zt), dt.t() @ zt / len(zt)
    damping = 0.01
    gamma = damping * float(torch.diagonal(h).mean())
    got = heldout_errors(weight, h, d, zh.t() @ zh, dh.t() @ zh,
                         float((dh @ weight.t()).pow(2).sum()), damping, (1.0, 10.0, float("inf")))
    for value, c in zip(got, (1.0, 10.0, float("inf"))):
        b = torch.zeros_like(d) if c == float("inf") else d @ torch.linalg.inv(h + c * gamma * torch.eye(h.shape[0], dtype=DTYPE))
        want = float((dh @ weight.t() - zh @ (weight @ b).t()).pow(2).sum())
        assert value == pytest.approx(want, rel=1e-8)


def test_limits_are_dac_and_obr() -> None:
    weight, x, z, seqlen = _layer()
    d = (z - x).t() @ z / len(z)
    dac, info = ridge_drift_weight(weight, x, z, seqlen, 0.01, scales=(1.0,))
    assert info["choice"] == 1.0
    assert torch.allclose(dac, weight @ d, atol=1e-9)
    obr, info = ridge_drift_weight(weight, x, z, seqlen, 0.01, scales=(float("inf"),))
    assert info["choice"] == float("inf") and bool((obr == 0).all())


def test_choice_follows_predictability() -> None:
    weight, x, z, seqlen = _layer(predictable=True)
    _, info = ridge_drift_weight(weight, x, z, seqlen, 0.01)
    assert info["choice"] != float("inf")
    weight, x, z, seqlen = _layer(predictable=False, seed=1)
    _, info = ridge_drift_weight(weight, x, z, seqlen, 0.01)
    assert info["choice"] >= 100.0
    assert min(info["cv"][:-1]) > 0.97  # nothing beats OBR by more than noise


def test_ridge_target_solves_the_regularized_regression() -> None:
    weight, x, z, seqlen = _layer(seed=2)
    shift, info = ridge_drift_weight(weight, x, z, seqlen, 0.01, scales=(10.0,))
    h, d = z.t() @ z / len(z), (z - x).t() @ z / len(z)
    gamma = 0.01 * float(torch.diagonal(h).mean())
    eye = torch.eye(h.shape[0], dtype=DTYPE)
    # The solver's target W - shift H_sys^-1 is W - W D (H + 10 g I)^-1.
    target = weight - shift @ torch.linalg.inv(h + gamma * eye)
    assert torch.allclose(target, weight - weight @ d @ torch.linalg.inv(h + 10 * gamma * eye), atol=1e-9)


def test_drift_ridge_is_named_and_validated() -> None:
    comp = CompensationSpec(method="drift", drift_ridge="cv")
    recipe = CompressionRecipe(compensation=comp)
    recipe.validate()
    assert run_name(recipe).endswith("_qdrift-cvr_seed42")
    assert CompensationSpec.from_dict(comp.to_dict()) == comp
    assert run_name(CompressionRecipe(compensation=CompensationSpec(method="drift"))).endswith("_qdrift_seed42")
    for bad in (CompensationSpec(method="obr", drift_ridge="cv"),
                CompensationSpec(method="drift", drift_ridge="cv", drift_shrink="row")):
        with pytest.raises(ValueError):
            bad.validate()


def test_single_device_path_uses_the_ridge(monkeypatch) -> None:
    import qdrift.compress.pipeline as pipeline

    weight, x, z, seqlen = _layer(seed=3)
    weight, x, z = weight.float(), x.float(), z.float()
    seen = {}
    monkeypatch.setattr(pipeline.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(pipeline, "_quantize_layer", lambda *args: seen.setdefault("drift", args[9]))
    recipe = CompressionRecipe(
        compensation=CompensationSpec(method="drift", drift_ridge="cv"), seqlen=seqlen,
    )
    pipeline._process_full_layer(recipe, None, "layer", weight, x, z, "cpu", True)
    want, _ = ridge_drift_weight(weight, x, z, seqlen, recipe.compensation.damping)
    assert torch.allclose(seen["drift"], want, atol=1e-4)
