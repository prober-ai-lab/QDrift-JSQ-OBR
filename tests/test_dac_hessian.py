# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAC's system Hessian on the runtime-quantized input, ``E[Q_a(z) Q_a(z)^T]``.

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
    assert P._hessian_quantizer(CompressionRecipe(compensation=CompensationSpec(method="drift"))) is not None


@pytest.mark.parametrize("streaming", [False, True])
def test_hessian_on_quantized_input_and_drift_on_z(streaming, monkeypatch) -> None:
    x, z = _inputs()
    recipe = CompressionRecipe(compensation=CompensationSpec(method="drift"))
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
    assert torch.allclose(diff, (z - x).t() @ z / len(z), atol=1e-5)
