# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run names, the v2 storage roots, and the metadata every v2 result carries.

A run is named ``{model}_{mechanism}_w{W}a{A}_{pattern}_{calib}_seed{n}``,
e.g. ``llama2-7b_obr_w4a4_s50_qdrift_seed0``.  Checkpoints go under
``checkpoints/v2/`` and results under ``results/v2/``, so nothing written
here can overwrite an older result; an existing name is refused unless the
caller passes ``overwrite``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from qdrift.compress.recipe import CompressionRecipe

__all__ = [
    "CHECKPOINTS_V2",
    "RESULTS_V2",
    "claim",
    "git_rev",
    "run_metadata",
    "run_name",
]

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_V2 = ROOT / "checkpoints" / "v2"
RESULTS_V2 = ROOT / "results" / "v2"

# compensation.method -> calibration tag
CALIBRATION_TAGS = {"drift": "qdrift", "obr": "seq", "none": "nocomp"}
# The calibration size left out of run names (samples, tokens each), from the first
# campaign; every other size, including the paper's 128x1024, is named (c128x1024).
CALIBRATION_SIZE = (32, 1024)


def pattern_tag(recipe: CompressionRecipe) -> str:
    """``s50`` for 50% unstructured, ``s0`` dense, ``2to4`` / ``4to8`` for
    N:M, with ``p`` when every layer is pairwise (``p40`` for a 40% share)
    and ``-obs`` for the exact compensated-cost mask selection."""
    structure = recipe.prune.structure
    if structure not in ("2:4", "4:8"):
        return f"s{round(recipe.prune.sparsity * 100)}"
    tag = structure.replace(":", "to")
    fraction = recipe.pairwise_fraction
    if fraction >= 1.0:
        tag = f"{tag}p"
    elif fraction > 0.0:
        tag = f"{tag}p{round(fraction * 100)}"
    if recipe.prune.select != "score":
        tag = f"{tag}-{recipe.prune.select}"
    return tag


def calibration_tag(recipe: CompressionRecipe) -> str:
    """``qdrift`` / ``seq`` / ``nocomp``; QDrift's non-default mixture weights
    are appended as ``-a{alpha}c{gamma}`` (``qdrift-a0c0`` is OBR's objective)."""
    comp = recipe.compensation
    return CALIBRATION_TAGS[comp.method] + comp.mix_tag()


def run_name(recipe: CompressionRecipe) -> str:
    """The v2 run/checkpoint/result name of a recipe.

    The pruning criterion is named only when it is not SparseGPT and the
    late-half quantizer only when it is not GPTQ, the v2 defaults, and the
    calibration size only when it is not ``CALIBRATION_SIZE``, e.g.
    ``llama2-7b_obr_w4a4_s50_wanda_c128x2048_qdrift_seed42``.
    """
    criterion = recipe.prune.criterion
    mask = [] if criterion == "sparsegpt" else [criterion]
    quant = [] if recipe.weight.method == "gptq" else [recipe.weight.method]
    size = (recipe.nsamples, recipe.seqlen)
    calib = [] if size == CALIBRATION_SIZE else [f"c{size[0]}x{size[1]}"]
    return "_".join(
        [
            recipe.model,
            recipe.compensation.mechanism_tag(),
            f"w{recipe.weight.bits}a{recipe.activation.bits}",
            pattern_tag(recipe),
            *mask,
            *quant,
            *calib,
            calibration_tag(recipe),
            f"seed{recipe.rotation.seed}",
        ]
    )


def claim(paths: list[Path], overwrite: bool = False) -> None:
    """Refuse to reuse a name unless ``overwrite``; create parent folders."""
    taken = [str(path) for path in paths if path.exists()]
    if taken and not overwrite:
        raise FileExistsError(
            "refusing to overwrite " + ", ".join(taken) + " (pass --overwrite)"
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def git_rev() -> str:
    """Commit of the deployed code.

    ``QDRIFT_REV`` wins when set: the sync/job wrappers export the true revision
    on server copies whose ``.git`` is stale (rsync/tar overwrite files but not
    ``.git``) or absent. Only the dev checkout, with no ``QDRIFT_REV``, reads the
    local git HEAD.
    """
    env = os.environ.get("QDRIFT_REV")
    if env:
        return env
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(
    recipe: CompressionRecipe, layers: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Fields every v2 checkpoint and result JSON carries.

    ``layers`` is the checkpoint's per-layer metadata; its mechanism fields
    (``mechanism``, ``lambda``, ``damping``) are copied per layer.
    """
    keys = ("mechanism", "lambda", "damping")
    per_layer = {
        name: {key: meta[key] for key in keys if key in meta}
        for name, meta in (layers or {}).items()
    }
    return {
        "name": run_name(recipe),
        "model": recipe.model,
        "mechanism": recipe.compensation.mechanism_tag(),
        "wbits": recipe.weight.bits,
        "abits": recipe.activation.bits,
        "sparsity": recipe.prune.sparsity,
        "pattern": pattern_tag(recipe),
        "criterion": recipe.prune.criterion,
        "calibration": calibration_tag(recipe),
        "quant": recipe.weight.method,
        "calib_size": f"{recipe.nsamples}x{recipe.seqlen}",
        "seed": recipe.rotation.seed,
        "git": git_rev(),
        "layers": per_layer,
        "config": recipe.to_dict(),
    }
