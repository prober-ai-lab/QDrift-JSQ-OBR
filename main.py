# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for compression and benchmarking."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

from benchmark.eval.evaluation import (
    PPL_DATASETS,
    ZEROSHOT_TASKS,
    ppl_eval,
    run_benchmark,
    zeroshot_eval,
)
from qdrift.compress.pipeline import compress, compress_regimes, parse_devices
from qdrift.compress.recipe import (
    CURVATURE,
    DRIFT_SHRINK,
    MECHANISMS,
    PRUNING_CRITERIA,
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    KVQuantSpec,
    PruneSpec,
    RotationSpec,
    WeightQuantSpec,
)


def _sparsity(value: str) -> float:
    parsed = float(value)
    allowed = tuple(round(index / 10, 1) for index in range(10))
    if parsed not in allowed:
        raise argparse.ArgumentTypeError(
            f"sparsity must be one of {allowed}, got {value}"
        )
    return round(parsed, 1)


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"pairwise fraction must be in [0, 1], got {value}")
    return parsed


def _granularity(kind: str, group_size: int) -> GranularitySpec:
    return GranularitySpec(
        kind=kind,
        group_size=group_size if kind == "group" else None,
    )


def _add_recipe_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument("--w-bits", "--bits", dest="w_bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--sparsity", type=_sparsity, default=0.5)
    parser.add_argument("--structure", choices=("unstructured", "2:4", "4:8"), default="4:8")
    parser.add_argument("--criterion", choices=PRUNING_CRITERIA, default="wanda")
    parser.add_argument("--w-gran", choices=("tensor", "channel", "group"), default="channel")
    parser.add_argument("--w-group-size", "--group-size", dest="w_group_size", type=int, default=128)
    parser.add_argument("--a-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--kv-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midtread", help="grid for weights, and for activations/KV unless --a-grid/--kv-grid is given")
    parser.add_argument("--a-grid", choices=("midrise", "midtread"), default=None, help="activation grid (default: --grid)")
    parser.add_argument("--kv-grid", choices=("midrise", "midtread"), default=None, help="KV-cache grid (default: --grid)")
    parser.add_argument("--range", choices=("full", "clip"), default="clip")
    parser.add_argument("--quant", choices=("rtn", "gptq"), default="gptq")
    parser.add_argument("--comp", choices=("none", "obr", "drift"), default="obr", help="calibration objective: obr=symmetric, drift=QDrift asymmetric")
    parser.add_argument("--mechanism", choices=MECHANISMS, default="obr", help="obr: statistics on full-precision activations; qir: with the outlier tokens runtime-quantized (docs/mechanisms.md)")
    parser.add_argument("--outlier-factor", type=float, default=8.0, help="qir: quantize tokens whose max|z_t| exceeds this times the median (0 = all tokens)")
    parser.add_argument("--drift-shrink", choices=DRIFT_SHRINK, default="none", help="drift: shrink DAC's drift term toward OBR by its noise across calibration sequences, per layer or per output row")
    parser.add_argument("--curvature", choices=CURVATURE, default="none", help="fisher: couple the output rows of the compensation through the layer's output curvature (empirical Fisher of the calibration NLL)")
    parser.add_argument("--curvature-damping", type=float, default=0.1, help="damping of the output curvature, relative to its mean diagonal")
    parser.add_argument("--curvature-iters", type=int, default=256, help="conjugate-gradient steps per compensation stage for --curvature")
    parser.add_argument("--rotation", choices=("hadamard", "none"), default="hadamard", help="hadamard=fast global Hadamard; none=identity")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for calibration, PPL, zero-shot, and e2e")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairwise-fraction", type=_fraction, default=0.4, help="fraction of least sensitive layers using pairwise masks; 1.0=all pair, 0.0=all dense")
    # fmt: on


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    _add_recipe_arguments(parser)


def _add_compress_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--devices", default="", help="comma-separated CUDA devices, e.g. 0,1,2,3")
    parser.add_argument("--parallel", choices=("auto", "workers", "intra"), default="auto", help="auto/workers: independent variant workers; intra: one process using all devices for row/layer parallelism")
    parser.add_argument("--single", action="store_true", help="only write the joint checkpoint (skip dense/sparse/quant)")
    # fmt: on


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--tasks", default="all", help="all | ppl | zeroshot | e2e")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ppl-datasets", default=",".join(PPL_DATASETS))
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-chunks", type=int, default=None, help="cap on PPL chunks per dataset (default: the whole split)")
    parser.add_argument("--lm-eval-tasks", default=",".join(ZEROSHOT_TASKS))
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--e2e-seqlen", type=int, default=1024)
    parser.add_argument("--e2e-decode-tokens", type=int, default=64)
    parser.add_argument("--e2e-warmup", type=int, default=3)
    parser.add_argument("--e2e-iters", type=int, default=5)
    parser.add_argument("--compile", action="store_true", help="compile compressed Linear modules with torch.compile")
    parser.add_argument("--engine", choices=("hf", "inference"), default="hf", help="e2e backend: hf (default) or the graph-friendly checkpoint runtime")
    parser.add_argument("--e2e-regimes", default="dense,sparse,quant,joint", help="comma-separated checkpoint regimes for the inference engine")
    parser.add_argument("--no-sparse-accel", action="store_true", help="force dense/int fallback instead of sparse Tensor Core acceleration")
    parser.add_argument("--e2e-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    # fmt: on


def _build_recipe(args: argparse.Namespace) -> CompressionRecipe:
    group = args.w_group_size
    return CompressionRecipe(
        model=args.model,
        weight=WeightQuantSpec(
            bits=args.w_bits,
            granularity=_granularity(args.w_gran, group),
            grid=args.grid,
            range=args.range,
            method=args.quant,
        ),
        activation=ActivationQuantSpec(
            bits=args.a_bits,
            granularity=_granularity(args.a_gran, group),
            grid=args.a_grid or args.grid,
            clip_ratio=1.0,
        ),
        kv_cache=KVQuantSpec(
            k_bits=args.kv_bits,
            v_bits=args.kv_bits,
            granularity=_granularity(args.kv_gran, group),
            grid=args.kv_grid or args.grid,
            k_clip_ratio=1.0,
            v_clip_ratio=1.0,
        ),
        prune=PruneSpec(
            sparsity=args.sparsity,
            structure=args.structure,
            criterion=args.criterion,
        ),
        rotation=RotationSpec(mode=args.rotation, seed=args.seed),
        compensation=CompensationSpec(
            method=args.comp,
            damping=args.damping,
            mechanism=args.mechanism,
            outlier_factor=args.outlier_factor,
            drift_shrink=args.drift_shrink,
            curvature=args.curvature,
            curvature_damping=args.curvature_damping,
            curvature_iters=args.curvature_iters,
        ),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        pairwise_fraction=args.pairwise_fraction,
    )

def _devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        return parse_devices(args.devices)
    return ["cuda"]


def _compress(args: argparse.Namespace, recipe: CompressionRecipe) -> int:
    devices = _devices(args)
    paths = compress_regimes(
        recipe,
        manifest_path="modelzoo/manifest.yaml",
        device=devices[0],
        output_dir="checkpoints",
        single=args.single,
        devices=devices,
        parallel=args.parallel,
    )
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--overwrite", action="store_true", help="replace an existing v2 checkpoint/result of the same name")
    parser.add_argument("--ppl", action=argparse.BooleanOptionalAction, default=True, help="evaluate WikiText-2 (full test) and C4 PPL at 2048 tokens")
    parser.add_argument("--c4-chunks", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--zeroshot", action=argparse.BooleanOptionalAction, default=False, help="evaluate the lm-eval zero-shot tasks")
    parser.add_argument("--lm-eval-tasks", default=",".join(ZEROSHOT_TASKS))
    parser.add_argument("--drop-checkpoint", action="store_true", help="delete the checkpoint once it is evaluated")
    # fmt: on


def _run(args: argparse.Namespace, recipe: CompressionRecipe) -> int:
    """v2: compress the joint checkpoint under its run name and evaluate it."""
    from benchmark.eval.runtime import load_runtime_model
    from qdrift.compress.runs import (
        CHECKPOINTS_V2,
        RESULTS_V2,
        claim,
        run_metadata,
        run_name,
    )
    from qdrift.data.dataloader import get_loaders

    name = run_name(recipe)
    checkpoint = CHECKPOINTS_V2 / f"{name}.safetensors"
    sidecar = checkpoint.with_suffix(".json")
    result = RESULTS_V2 / f"{name}.json"
    claim([checkpoint, sidecar, result], args.overwrite)
    devices = _devices(args)
    start = time.perf_counter()
    compress(
        recipe,
        manifest_path="modelzoo/manifest.yaml",
        device=devices[0],
        output_dir=str(CHECKPOINTS_V2),
        devices=devices,
        name=name,
    )
    layers = json.loads(sidecar.read_text(encoding="utf-8")).get("layers")
    record = {
        **run_metadata(recipe, layers),
        "checkpoint": str(checkpoint),
        "compress_s": time.perf_counter() - start,
    }
    if args.ppl or args.zeroshot:
        model, tokenizer, _ = load_runtime_model(
            checkpoint,
            "joint",
            manifest_path="modelzoo/manifest.yaml",
            device=devices[0],
            allow_sparse=False,
        )
    if args.ppl:
        record["ppl"] = {}
        for dataset in ("wikitext2", "c4"):
            chunks = get_loaders(
                dataset,
                tokenizer,
                nsamples=args.c4_chunks if dataset == "c4" else None,
                seqlen=2048,
                eval_mode=True,
                seed=42,
            )
            value = ppl_eval(
                model, chunks, device=devices[0], batch_size=args.eval_batch_size
            )
            record["ppl"][dataset] = value[0] if isinstance(value, tuple) else value
            print(f"{name} ppl {dataset} = {record['ppl'][dataset]:.4f}", flush=True)
    if args.zeroshot:
        tasks = tuple(task for task in args.lm_eval_tasks.split(",") if task)
        record["zeroshot"] = zeroshot_eval(model, tokenizer, tasks, batch_size=16)
        print(f"{name} zeroshot avg = {record['zeroshot']['avg']:.2f}", flush=True)
    if args.drop_checkpoint:
        checkpoint.unlink()
        record["checkpoint"] = None
    record["total_s"] = time.perf_counter() - start
    result.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved {result}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to compression or benchmarking."""
    parser = argparse.ArgumentParser(description="QDrift compression and benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser(
        "compress",
        help="write the four comparison checkpoints",
    )
    _add_common_arguments(compress_parser)
    _add_compress_arguments(compress_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="compress one joint checkpoint by v2 run name and evaluate its PPL",
    )
    _add_common_arguments(run_parser)
    _add_compress_arguments(run_parser)
    _add_run_arguments(run_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="evaluate the four checkpoints selected by the recipe",
    )
    _add_common_arguments(benchmark_parser)
    _add_benchmark_arguments(benchmark_parser)

    args = parser.parse_args(argv)
    recipe = _build_recipe(args)
    recipe.validate()
    if args.command == "compress":
        return _compress(args, recipe)
    if args.command == "run":
        return _run(args, recipe)
    return run_benchmark(recipe, args)


if __name__ == "__main__":
    raise SystemExit(main())
