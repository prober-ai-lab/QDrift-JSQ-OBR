# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAC's drift shrinkage: James-Stein factors from per-sequence drift terms.

    python -m pytest tests/test_drift_shrink.py
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
from qdrift.obr.drift import drift_weight, shrink_drift_weight  # noqa: E402


def _shift(weight, fp_input, compressed_input):
    count = compressed_input.shape[0]
    diff = (compressed_input - fp_input).t() @ compressed_input / count
    return drift_weight(weight, diff)


def test_identical_sequences_keep_dac() -> None:
    generator = torch.Generator().manual_seed(0)
    seqlen, cols = 64, 16
    x = torch.randn(seqlen, cols, generator=generator)
    z = x + 0.3 * torch.randn(seqlen, cols, generator=generator)
    fp_input, compressed_input = x.repeat(4, 1), z.repeat(4, 1)
    weight = torch.randn(8, cols, generator=generator)
    shift = _shift(weight, fp_input, compressed_input)
    for mode in ("row", "layer"):
        got, factor = shrink_drift_weight(
            shift, weight, fp_input, compressed_input, seqlen, mode
        )
        assert torch.allclose(factor, torch.ones_like(factor), atol=1e-4)
        assert torch.allclose(got, shift, atol=1e-5)


def test_noise_drift_falls_back_to_obr() -> None:
    """Drift independent of z has zero mean term; the factors collapse."""
    generator = torch.Generator().manual_seed(1)
    seqlen, cols, count = 32, 64, 64
    z = torch.randn(count * seqlen, cols, generator=generator)
    x = z + torch.randn(count * seqlen, cols, generator=generator)
    weight = torch.randn(8, cols, generator=generator)
    shift = _shift(weight, x, z)
    _, factor = shrink_drift_weight(shift, weight, x, z, seqlen, "row")
    assert float(factor.max()) < 0.4


def test_shrinkage_is_named_and_round_trips() -> None:
    plain = CompressionRecipe(compensation=CompensationSpec(method="drift"))
    shrunk = CompressionRecipe(
        compensation=CompensationSpec(method="drift", drift_shrink="row")
    )
    assert run_name(plain).endswith("_qdrift_seed42")
    assert run_name(shrunk).endswith("_qdrift-jsrow_seed42")
    assert "drift_shrink" not in plain.compensation.to_dict()
    spec = shrunk.compensation
    assert CompensationSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="needs --comp drift"):
        CompensationSpec(method="obr", drift_shrink="row").validate()


def test_single_device_path_shrinks(monkeypatch) -> None:
    """``_process_full_layer`` hands the shrunk drift term to the solver."""
    import qdrift.compress.pipeline as pipeline

    generator = torch.Generator().manual_seed(2)
    seqlen, cols, count = 32, 64, 64
    z = torch.randn(count * seqlen, cols, generator=generator)
    x = z + torch.randn(count * seqlen, cols, generator=generator)
    weight = torch.randn(8, cols, generator=generator)
    seen = {}
    monkeypatch.setattr(pipeline.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        pipeline, "_quantize_layer", lambda *args: seen.setdefault("drift", args[9])
    )
    for mode in ("none", "row"):
        recipe = CompressionRecipe(
            compensation=CompensationSpec(method="drift", drift_shrink=mode),
            seqlen=seqlen,
        )
        seen.clear()
        pipeline._process_full_layer(recipe, None, "layer", weight, x, z, "cpu", True)
        expected = _shift(weight, x, z)
        if mode == "row":
            expected, _ = shrink_drift_weight(expected, weight, x, z, seqlen, "row")
        assert torch.allclose(seen["drift"], expected, atol=1e-4)
