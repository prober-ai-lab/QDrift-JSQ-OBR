# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The activation quantizer uses the symmetric max|x| / 7 range at 4 bits.

Unstructured and pairwise-4:8 W4A4 must share one activation quantizer: the
deployed QuaRot INT4 kernel's symmetric mid-tread range (codes -7..7).  Weights
keep the full asymmetric range (-8..7).

    python -m pytest tests/test_activation_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.kernels.lib.int8_linear import (  # noqa: E402
    _quantize_codes,
    quantize_activations,
)


def test_activations_are_symmetric_max_over_7() -> None:
    """4-bit activations quantize on the symmetric grid: scale = amax / 7,
    codes in -7..7, the extreme negative input clamped to -7."""
    # One row whose largest magnitude is the negative extreme.
    x = torch.tensor([[-8.0, 4.0, 1.0, -2.0, 7.0, 0.0, -1.0, 3.0]])
    codes, scales, group, offset = quantize_activations(
        x, bits=4, grid="midtread", granularity="token", group_size=None
    )
    assert offset == 0.0
    amax = x.abs().amax()
    assert torch.allclose(scales.flatten(), (amax / 7.0).float(), atol=1e-5)
    assert int(codes.min()) >= -7 and int(codes.max()) <= 7
    # -8 is the row max in magnitude, so it maps to the symmetric extreme -7.
    assert int(codes[0, 0]) == -7


def test_weights_keep_the_asymmetric_range() -> None:
    """``_quantize_codes`` default (weights) keeps the full -8..7 range, so the
    activation change does not touch weight quantization."""
    x = torch.tensor([[-8.0, 4.0, 1.0, -2.0, 7.0, 0.0, -1.0, 3.0]])
    codes, scales, group, offset = _quantize_codes(x, bits=4, grid="midtread", group_size=8)
    amax = x.abs().amax()
    assert torch.allclose(scales.flatten(), (amax / 8.0).float(), atol=1e-5)
    # With scale = amax / 8 the negative extreme reaches the asymmetric code -8.
    assert int(codes[0, 0]) == -8


@pytest.mark.parametrize("bits", [4])
def test_symmetric_flag_drops_the_extra_negative_code(bits) -> None:
    """The symmetric flag is what separates the two ranges."""
    x = torch.tensor([[-8.0, 7.0, -3.0, 2.0, 5.0, -6.0, 1.0, 0.0]])
    asym, _, _, _ = _quantize_codes(x, bits=bits, grid="midtread", group_size=8)
    sym, _, _, _ = _quantize_codes(x, bits=bits, grid="midtread", group_size=8, symmetric=True)
    assert int(asym.min()) == -(2 ** (bits - 1))
    assert int(sym.min()) == -(2 ** (bits - 1) - 1)
