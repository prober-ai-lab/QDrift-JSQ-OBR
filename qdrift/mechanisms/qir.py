# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
QIR: quantized-input reconstruction.

At deployment a compressed layer computes ``W_hat Q_a(z)``: its input ``z``
comes from the compressed prefix, whose own linears also quantize their
inputs, and is then quantized by the layer's activation quantizer ``Q_a``.
QIR fits the layer to that input, per row,

    min_v E[(v^T Q_a(z) - w^T x)^2],    v_P = 0,

with ``x`` the full-precision input under QDrift calibration and ``x = z``
otherwise.  This is OBR's objective with ``z`` replaced by ``z_q = Q_a(z)``:
the pipeline builds ``H = E[z_q z_q^T]`` and ``D = E[(z_q - x) z_q^T]`` from
the quantized inputs, and OBR's solver is used unchanged.  The noise
``n = z_q - z`` enters exactly, through ``E[z n^T]`` and ``E[n n^T]``,
instead of through a model of its variance.
"""

from __future__ import annotations

from qdrift.mechanisms.obr import OBRMechanism


class QIRMechanism(OBRMechanism):
    """OBR on the statistics of the runtime-quantized layer inputs."""

    name = "qir"
    quantizes_inputs = True
