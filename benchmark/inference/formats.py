# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment-format benchmark: memory, prefill / decode latency and throughput of
one compressed v2 checkpoint, or of the uncompressed FP16 model.

At a fixed (bits, sparsity, mask) OBR and QDrift write the same format (same packed
codes, mask layout and kernels), so a format is timed once, on any checkpoint of it.
Each call measures one model in one format on one GPU with the graph-friendly
inference runtime (``benchmark.inference``) and writes
``results/benchmarks/formats__<model>__<format>__b<batch>.json``.

    python -m benchmark.inference.formats --checkpoint checkpoints/v2/<run>.safetensors \
        --format w4a4-4to8p --batch 1 --prefill 2048 --decode 24
    python -m benchmark.inference.formats --checkpoint <any v2 checkpoint of the model> \
        --format fp16 --batch 1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from benchmark.inference.checkpoint import build_checkpoint_model
from benchmark.inference.config import BenchConfig
from benchmark.inference.linear import DenseLinear
from benchmark.inference.measure import measure_e2e
from benchmark.inference.weights import SafetensorStore
from qdrift.compress.recipe import CompressionRecipe, RotationSpec
from qdrift.models.backbone import resolve_model_source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "benchmarks"
MANIFEST = "modelzoo/manifest.yaml"


def _fp16_factory(source: str):
    """Linear factory that ignores the checkpoint and loads the original FP16 weight."""
    store = SafetensorStore(source)

    def make(config, tensors, name, regime, recipe, device, allow_sparse):
        return DenseLinear(store.get(f"{name}.weight", device).half())

    return make, store


def _release_unused_codes(model) -> None:
    """Marlin layers compute from their repacked weight only; the runtime also keeps
    the checkpoint's packed codes for other paths.  Drop that second copy so the
    measured memory is the deployed format's."""
    for module in model.modules():
        case = getattr(module, "case", None)
        if case is not None and str(getattr(case, "kernel", "")).startswith(
                ("marlin", "sparse_marlin")):
            case.packed = torch.empty(0, device=case.packed.device, dtype=torch.uint8)
            case.scales = torch.empty(0, device=case.scales.device, dtype=torch.float16)
            case.mask = None
    torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    recipe = CompressionRecipe.from_config(config)
    source = resolve_model_source(recipe.model, MANIFEST)
    fp16 = args.format == "fp16"
    store = None
    factory = None
    if fp16:
        # The original model: no rotation (norm scales stay in the norms), FP16 linears, KV16.
        recipe = replace(recipe, rotation=RotationSpec(mode="none"))
        factory, store = _fp16_factory(source)
    kv_bits = 16 if fp16 else int(recipe.kv_cache.k_bits)
    torch.cuda.empty_cache()
    start = time.perf_counter()
    # Pairwise 4:8 runs the sparse INT4 kernel (``joint``).  Unstructured sparsity has
    # no hardware path, so it deploys on the dense integer kernel (``quant``: the
    # pruned weights are zero codes, the output is the same).
    pairwise = recipe.prune.structure in ("2:4_pairwise", "4:8_pairwise") or (
        recipe.prune.structure in ("2:4", "4:8") and recipe.pairwise_fraction >= 1.0)
    regime = "joint" if pairwise and not args.no_sparse else "quant"
    model, _ = build_checkpoint_model(
        recipe, checkpoint, regime, MANIFEST, args.device, args.prefill, args.decode,
        linear_factory=factory, allow_sparse=not args.no_sparse,
    )
    load_s = time.perf_counter() - start
    _release_unused_codes(model)
    if store is not None:
        store.close()
    cfg = BenchConfig(
        model_dir=source, regime="dense" if fp16 else "joint_sparse_int4", device=args.device,
        batch=args.batch, prefill_len=args.prefill, decode_steps=args.decode,
        kv_bits=kv_bits, warmup=args.warmup, trials=args.trials,
        cuda_graph=args.cuda_graph,
    )
    measured = measure_e2e(model, cfg)
    if args.cuda_graph and measured["timing_mode"] != "cuda_graph":
        # Eager timing is launch-bound (4-5x slower decode for the quantized
        # kernels) and is not comparable across formats: never record it.
        raise SystemExit("CUDA-graph capture failed; refusing to record eager timing")
    return {
        "model": recipe.model,
        "format": args.format,
        "checkpoint": None if fp16 else str(checkpoint),
        "gpu": torch.cuda.get_device_name(args.device),
        "batch": args.batch,
        "prefill_len": args.prefill,
        "decode_steps": args.decode,
        "kv_bits": kv_bits,
        "regime": "fp16" if fp16 else regime,
        "load_s": load_s,
        **measured,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", required=True,
                        help="label, e.g. fp16, w4a16-s50, w4a4-s50, w4a4-4to8p")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=2048)
    # Decode steps are captured as one CUDA graph, one kernel node set per step; the
    # driver caps live graph nodes per process (~360k simple nodes on the 4090s), which
    # LLaMA-2-13B pairwise W4A4 reaches at ~34 steps.  24 steps fit every model and format.
    parser.add_argument("--decode", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-sparse", action="store_true", help="dense kernels even for pairwise 4:8")
    args = parser.parse_args()
    record = run(args)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"formats__{record['model']}__{args.format}__b{args.batch}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("timing_mode", "prefill", "decode", "memory")}, indent=1))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
