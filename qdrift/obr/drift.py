# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DriftOBR compressed-prefix capture and drift terms."""

from __future__ import annotations

import torch

from qdrift.obr.calibrate import capture_inputs

__all__ = ["capture_compressed_input", "drift_weight"]


@torch.no_grad()
def capture_compressed_input(
    model: torch.nn.Module,
    capture: str,
    chunks: list[torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Capture one layer input from the partially compressed model."""
    return capture_inputs(model, [capture], chunks, device)[capture]


def drift_weight(weight: torch.Tensor, hessian_diff: torch.Tensor) -> torch.Tensor:
    """Project a weight matrix through the drift Hessian difference."""
    return weight @ hessian_diff
