# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
QIR: quantized-input reconstruction on the outlier tokens.

At deployment a compressed layer computes ``W_hat Q_a(z)``: its input ``z``
comes from the compressed prefix, whose own linears also quantize their
inputs, and is then quantized per token by the layer's activation quantizer
``Q_a``.  For most tokens the rounding noise ``Q_a(z) - z`` is small and
unstructured, and fitting it only fits the calibration sample.  Tokens with a
massive activation (``max|z_t|`` above ``OUTLIER_FACTOR`` times the median)
get a step so large that nearly all their other channels round to zero; that
error is systematic and recurs on every input.  QIR fits the layer to

    z_hat_t = Q_a(z_t)  for outlier tokens,   z_hat_t = z_t  otherwise,

per row ``min_v E[(v^T z_hat - w^T x)^2]``, ``v_P = 0``, with ``x`` the
full-precision input under QDrift calibration and ``x = z`` otherwise.  The
pipeline builds ``H = E[z_hat z_hat^T]`` and ``D = E[(z_hat - x) z_hat^T]``
and OBR's solver is used unchanged; without outlier tokens QIR is OBR.
"""

from __future__ import annotations

from qdrift.mechanisms.obr import OBRMechanism

OUTLIER_FACTOR = 8.0


class QIRMechanism(OBRMechanism):
    """OBR on inputs whose outlier tokens are runtime-quantized."""

    name = "qir"
    quantizes_inputs = True
    outlier_factor = OUTLIER_FACTOR
