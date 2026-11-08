# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Compensation mechanisms behind one interface; see docs/mechanisms.md.
"""

from __future__ import annotations

from qdrift.compress.recipe import MECHANISMS
from qdrift.mechanisms.base import Compensation, LayerStats, Mechanism
from qdrift.mechanisms.obr import OBRMechanism
from qdrift.mechanisms.wiener import WienerMechanism


__all__ = [
    "MECHANISMS",
    "Compensation",
    "LayerStats",
    "Mechanism",
    "OBRMechanism",
    "WienerMechanism",
    "get_mechanism",
]


def get_mechanism(name: str, noise_scale: float = 1.0) -> Mechanism:
    """Instantiate a mechanism by its CLI name."""
    if name == "obr":
        return OBRMechanism()
    if name == "wiener":
        return WienerMechanism(noise_scale)
    raise ValueError(f"unknown mechanism {name}; use one of {MECHANISMS}")
