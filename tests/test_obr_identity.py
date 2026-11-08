# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity of the OBR mechanism across refactors.

Runs the pipeline's per-layer compression on synthetic layers and compares
sha256 hashes of every output tensor with ``data/obr_reference.json``, which
was recorded with the code before the mechanism refactor (commit 965f0a5).

    python -m pytest tests/test_obr_identity.py
    python tests/test_obr_identity.py --record   # only on a reference commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress import pipeline  # noqa: E402
from qdrift.compress.recipe import (  # noqa: E402
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    WeightQuantSpec,
)
from qdrift.quant.quantize import qrange  # noqa: E402

REFERENCE = Path(__file__).parent / "data" / "obr_reference.json"

# name: (method, quant, bits, grid, granularity, structure, criterion)
CASES = {
    "obr_gptq_w4_ch_mt_u": ("obr", "gptq", 4, "midtread", "channel", "unstructured", "magnitude"),
    "obr_rtn_w3_g64_mr_u": ("obr", "rtn", 3, "midrise", "group", "unstructured", "magnitude"),
    "drift_gptq_w4_ch_mt_u": ("drift", "gptq", 4, "midtread", "channel", "unstructured", "magnitude"),
    "drift_rtn_w2_g64_mr_u": ("drift", "rtn", 2, "midrise", "group", "unstructured", "sparsegpt"),
    "obr_gptq_w4_ch_mt_24": ("obr", "gptq", 4, "midtread", "channel", "2:4", "wanda"),
}


def synthetic_layer(seed: int = 0) -> dict[str, torch.Tensor]:
    """A small correlated layer with full-precision and compressed inputs."""
    generator = torch.Generator().manual_seed(seed)
    rows, cols, tokens = 48, 256, 1024
    basis = torch.randn(cols, 32, generator=generator)
    fp = torch.randn(tokens, 32, generator=generator) @ basis.t()
    fp += 0.3 * torch.randn(tokens, cols, generator=generator)
    hat = fp + 0.1 * torch.randn(tokens, cols, generator=generator)
    weight = 0.02 * torch.randn(rows, cols, generator=generator)
    return {"weight": weight, "fp": fp, "hat": hat}


def recipe_for(case: tuple) -> CompressionRecipe:
    """The recipe of one identity case."""
    method, quant, bits, grid, kind, structure, criterion = case
    return CompressionRecipe(
        weight=WeightQuantSpec(
            bits=bits,
            granularity=GranularitySpec(kind, 64 if kind == "group" else None),
            grid=grid,
            range="clip",
            method=quant,
        ),
        prune=PruneSpec(sparsity=0.5, structure=structure, criterion=criterion),
        compensation=CompensationSpec(method=method),
        pairwise_fraction=0.0,
    )


def digest(tensor: torch.Tensor) -> str:
    """sha256 of a tensor's raw bytes."""
    data = tensor.detach().cpu().contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def run_case(name: str) -> dict[str, str]:
    """Hashes of every output tensor of one case."""
    torch.set_num_threads(1)
    layer = synthetic_layer()
    recipe = recipe_for(CASES[name])
    hat = layer["hat"] if recipe.compensation.method == "drift" else None
    hessian_fp, hessian_system, hessian_diff = pipeline._hessians(
        layer["fp"], hat, recipe.compensation.damping
    )
    result = pipeline._quantize_layer(
        recipe,
        qrange(recipe.weight.bits, recipe.weight.grid),
        name,
        layer["weight"],
        layer["weight"],
        hessian_fp,
        hessian_system,
        hessian_system,
        hessian_diff,
    )
    hashes = {key: digest(value) for key, value in sorted(result.tensors.items())}
    hashes["new_weight"] = digest(result.new_weight)
    return hashes


def test_obr_bit_identity() -> None:
    """Every case reproduces the recorded hashes exactly."""
    reference = json.loads(REFERENCE.read_text())
    for name in CASES:
        assert run_case(name) == reference[name], name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        hashes = {name: run_case(name) for name in CASES}
        REFERENCE.write_text(json.dumps(hashes, indent=1) + "\n")
        print(f"recorded {len(hashes)} cases to {REFERENCE}")
    else:
        test_obr_bit_identity()
        print("bit-identical")
