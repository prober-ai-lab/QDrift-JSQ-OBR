# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that weight offload is bit-identical to on-device compression.

Run on a GPU box::

    python scripts/check_offload.py --model llama2-7b --max-layers 8

Compresses the same recipe twice -- once with the whole model on the GPU and
once with the decoder blocks parked on the host and shuttled one at a time --
and asserts that every checkpoint tensor is identical.  Offload only changes
where weights live, not the math, so the checkpoints must match exactly.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from qdrift.compress.pipeline import compress
from qdrift.compress.recipe import (
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    KVQuantSpec,
    PruneSpec,
    RotationSpec,
    WeightQuantSpec,
)


def _granularity(kind: str, group_size: int) -> GranularitySpec:
    return GranularitySpec(kind=kind, group_size=group_size if kind == "group" else None)


def _recipe(model: str, max_layers: int) -> CompressionRecipe:
    """A W4A4, Hadamard-rotated, SparseGPT s0.5 recipe over the first blocks."""
    recipe = CompressionRecipe(
        model=model,
        weight=WeightQuantSpec(
            bits=4, granularity=_granularity("channel", 128), grid="midtread",
            range="clip", method="gptq",
        ),
        activation=ActivationQuantSpec(
            bits=4, granularity=_granularity("token", 128), grid="midtread",
            clip_ratio=1.0,
        ),
        kv_cache=KVQuantSpec(
            k_bits=4, v_bits=4, granularity=_granularity("token", 128),
            grid="midtread", k_clip_ratio=1.0, v_clip_ratio=1.0,
        ),
        prune=PruneSpec(
            sparsity=0.5, structure="unstructured", criterion="sparsegpt",
            select="score",
        ),
        rotation=RotationSpec(mode="hadamard", seed=42),
        compensation=CompensationSpec(
            method="drift", damping=0.01, mix_asym=0.25, mix_clean=0.25,
        ),
        nsamples=8,
        seqlen=512,
        batch_size=4,
        pairwise_fraction=0.0,
    )
    return replace(recipe, max_layers=max_layers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    recipe = _recipe(args.model, args.max_layers)
    with tempfile.TemporaryDirectory() as tmp:
        on = compress(recipe, device=args.device, output_dir=tmp,
                      name="ondevice", offload=False)
        off = compress(recipe, device=args.device, output_dir=tmp,
                       name="offload", offload=True)
        a = load_file(str(on))
        b = load_file(str(off))

    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        print("FAIL: tensor sets differ")
        print("  only on-device:", sorted(keys_a - keys_b)[:8])
        print("  only offload:  ", sorted(keys_b - keys_a)[:8])
        return 1
    mismatched = [k for k in a if not torch.equal(a[k], b[k])]
    print(f"tensors={len(a)} mismatched={len(mismatched)}")
    if mismatched:
        for k in mismatched[:8]:
            da = (a[k].float() - b[k].float()).abs().max().item()
            print(f"  {k}: max|delta|={da:.3e}")
        print("FAIL: offload is not bit-identical")
        return 1
    print("PASS: offload is bit-identical to on-device compression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
