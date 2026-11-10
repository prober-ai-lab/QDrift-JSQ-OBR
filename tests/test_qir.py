# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qir mechanism: statistics with the outlier tokens runtime-quantized.

Per row the objective is ``f(v) = mean_t (v^T z_hat_t - w^T x_t)^2`` with
``v_P = 0``, ``z_hat_t = Q_a(z_t)`` on outlier tokens and ``z_t`` elsewhere,
compressed-path inputs ``z`` and target inputs ``x`` (``x = z`` without
QDrift).  References are built from the samples alone, in float64.

    python -m pytest tests/test_qir.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress.recipe import (  # noqa: E402
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
)
from qdrift.mechanisms import get_mechanism  # noqa: E402
from qdrift.obr.calibrate import CalibrationStart, SequentialCalibration  # noqa: E402
from qdrift.obr.obr import compensate_pruning_error  # noqa: E402
from qdrift.prune.mask import mask_from_score  # noqa: E402
from qdrift.quant.activation import (  # noqa: E402
    fake_quantize_activations,
    outlier_tokens,
    quantize_outlier_tokens,
)


@pytest.mark.parametrize("drift", [False, True])
def test_pruning_stage_is_least_squares(drift: bool) -> None:
    """OBR's pruning update on (H, D) of z_hat solves the per-row fit."""
    generator = torch.Generator().manual_seed(0)
    rows, cols, tokens = 6, 32, 4000
    basis = torch.randn(cols, 10, generator=generator, dtype=torch.float64)
    x = torch.randn(tokens, 10, generator=generator, dtype=torch.float64) @ basis.t()
    x += 0.2 * torch.randn(tokens, cols, generator=generator, dtype=torch.float64)
    x[:, 5] *= 8.0
    x[::40] *= 30.0
    z = x + 0.2 * torch.randn_like(x) if drift else x
    zq = quantize_outlier_tokens(z, 8.0, 4, "midtread", device="cpu")
    weight = 0.05 * torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    mask = mask_from_score(weight.abs(), 0.5, "unstructured")

    hessian = zq.t() @ zq / tokens
    diff = (zq - x).t() @ zq / tokens
    state = compensate_pruning_error(weight, hessian, hessian, diff, mask)

    target = x @ weight.t()
    for row in range(rows):
        keep = state.keep[row]
        expect = torch.linalg.lstsq(zq[:, keep], target[:, row : row + 1]).solution
        assert torch.allclose(state.retained[row], expect.squeeze(1), atol=1e-8)


def test_only_outlier_tokens_are_quantized() -> None:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1000, 64, generator=generator)
    x[::50, 7] *= 300.0
    hot = outlier_tokens(x, 8.0, device="cpu")
    assert torch.equal(hot.nonzero().squeeze(1), torch.arange(0, 1000, 50))
    got = quantize_outlier_tokens(x, 8.0, 4, "midtread", device="cpu", chunk=7)
    assert got.dtype == x.dtype and got.device == x.device
    assert torch.equal(got[hot], fake_quantize_activations(x[hot], 4, "midtread"))
    assert torch.equal(got[~hot], x[~hot])


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(16, 16, bias=False)
        self.second = torch.nn.Linear(16, 16, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(hidden))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block(), _Block()])


def test_replay_quantizes_every_linear_but_the_captured_one() -> None:
    torch.manual_seed(0)
    model = _Model()
    hidden = torch.randn(1, 40, 16)
    start = CalibrationStart(hidden=[hidden], kwargs=[{}], inputs={})
    quantized = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

    def act_quant(values: torch.Tensor) -> torch.Tensor:
        return fake_quantize_activations(values, 4, "midtread")

    replay = SequentialCalibration(model, start, act_quant, quantized)
    got = replay.capture("model.layers.1.second")
    replay.close()
    block0, block1 = model.model.layers
    with torch.no_grad():
        h = block0.second(act_quant(block0.first(act_quant(hidden.reshape(-1, 16)))))
        expect = block1.first(act_quant(h))
        assert torch.allclose(got, expect, atol=1e-6)
    assert all(not module._forward_pre_hooks for module in quantized)


def test_qir_is_obr_on_outlier_quantized_inputs() -> None:
    mechanism = get_mechanism("qir")
    assert mechanism.quantizes_inputs and mechanism.outlier_factor == 8.0
    assert not get_mechanism("obr").quantizes_inputs
    assert get_mechanism("qir", 0.0).outlier_factor == 0.0


def test_outlier_factor_is_named_only_when_not_default() -> None:
    default = CompensationSpec(mechanism="qir")
    assert default.mechanism_tag() == "qir"
    assert "outlier_factor" not in default.to_dict()
    every_token = CompensationSpec(mechanism="qir", outlier_factor=0.0)
    assert every_token.mechanism_tag() == "qir-k0"
    assert CompensationSpec.from_dict(every_token.to_dict()) == every_token


def test_qir_needs_quantized_activations() -> None:
    recipe = CompressionRecipe(
        model="m",
        activation=ActivationQuantSpec(bits=16),
        compensation=CompensationSpec(method="drift", mechanism="qir"),
    )
    with pytest.raises(ValueError, match="quantized activations"):
        recipe.validate()


@pytest.mark.parametrize("grid", ["midtread", "midrise"])
def test_activation_quantizer_matches_runtime(grid: str) -> None:
    """The compression-side A4 quantizer equals the evaluation runtime's."""
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
