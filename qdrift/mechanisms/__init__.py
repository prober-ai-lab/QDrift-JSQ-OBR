# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Compensation mechanisms behind one interface; see docs/mechanisms.md.
"""

from __future__ import annotations

from qdrift.compress.recipe import MECHANISMS
from qdrift.mechanisms.base import Compensation, LayerStats, Mechanism
from qdrift.mechanisms.obr import OBRMechanism
from qdrift.mechanisms.qir import QIRMechanism


__all__ = [
    "MECHANISMS",
    "Compensation",
    "LayerStats",
    "Mechanism",
    "OBRMechanism",
    "QIRMechanism",
    "get_mechanism",
]


def get_mechanism(name: str) -> Mechanism:
    """Instantiate a mechanism by its CLI name."""
    if name == "obr":
        return OBRMechanism()
    if name == "qir":
        return QIRMechanism()
    raise ValueError(f"unknown mechanism {name}; use one of {MECHANISMS}")
