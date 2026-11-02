# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for compression and benchmarking."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from benchmark.eval.evaluation import PPL_DATASETS, ZEROSHOT_TASKS, run_benchmark
from qdrift.compress.pipeline import compress_regimes, parse_devices
from qdrift.compress.recipe import (
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    KVQuantSpec,
    PruneSpec,
    RotationSpec,
    SmoothSpec,
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


def _granularity(kind: str, group_size: int) -> GranularitySpec:
    return GranularitySpec(
        kind=kind,
        group_size=group_size if kind == "group" else None,
    )


def _add_recipe_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--w-bits", "--bits", dest="w_bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=16)
    parser.add_argument("--k-bits", type=int, default=16)
    parser.add_argument("--v-bits", type=int, default=16)
    parser.add_argument("--sparsity", type=_sparsity, default=0.5)
    parser.add_argument(
        "--structure",
        choices=("unstructured", "2:4", "4:8", "2:4_pairwise"),
        default="2:4",
    )
    parser.add_argument("--criterion", choices=("wanda", "qwanda", "magnitude", "random"), default="qwanda")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.4)
    parser.add_argument("--w-gran", choices=("tensor", "channel", "group"), default="channel")
    parser.add_argument("--w-group-size", "--group-size", dest="w_group_size", type=int, default=128)
    parser.add_argument("--a-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--a-group-size", type=int, default=128)
    parser.add_argument("--kv-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--kv-group-size", type=int, default=128)
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midrise")
    parser.add_argument("--a-grid", choices=("midrise", "midtread"), default="midtread")
    parser.add_argument("--kv-grid", choices=("midrise", "midtread"), default="midtread")
    parser.add_argument("--range", choices=("full", "clip"), default="clip")
    parser.add_argument("--clip-rhos", default="0.85,0.9,0.95,1.0")
    parser.add_argument("--a-clip", type=float, default=0.9)
    parser.add_argument("--k-clip", type=float, default=0.85)
    parser.add_argument("--v-clip", type=float, default=0.95)
    parser.add_argument("--quant", choices=("rtn", "gptq"), default="rtn")
    parser.add_argument("--gptq-alpha", type=float, default=0.5)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--comp", choices=("none", "obr", "drift"), default="obr")
    parser.add_argument("--comp-alpha", type=float, default=0.5)
    parser.add_argument("--shift-scale", type=float, default=1.0)
    parser.add_argument("--rotation", choices=("hadamard", "none"), default="hadamard", help="hadamard=fast global Hadamard; none=identity")
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=0)
    # fmt: on


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    _add_recipe_arguments(parser)
    # fmt: off
    parser.add_argument("--manifest", default="modelzoo/manifest.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    # fmt: on


def _add_compress_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--devices", default="", help="comma-separated CUDA devices, e.g. 0,1,2,3")
    parser.add_argument("--gpus", type=int, default=1, help="use cuda:0..cuda:N-1 (ignored when --devices is set)")
    parser.add_argument("--parallel", choices=("auto", "workers", "intra"), default="auto", help="auto/workers: independent variant workers; intra: one process using all devices for row/layer parallelism")
    parser.add_argument("--single", action="store_true", help="only write the joint checkpoint (skip dense/sparse/quant)")
    # fmt: on


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("--tasks", default="all", help="all | ppl | zeroshot | e2e")
    parser.add_argument("--output", default="results/benchmarks.json")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ppl-datasets", default=",".join(PPL_DATASETS))
    parser.add_argument("--ppl-seqlen", type=int, default=1024)
    parser.add_argument("--ppl-chunks", type=int, default=32)
    parser.add_argument("--lm-eval-tasks", default=",".join(ZEROSHOT_TASKS))
    parser.add_argument("--lm-eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--e2e-seqlen", type=int, default=512)
    parser.add_argument("--e2e-decode-tokens", type=int, default=32)
    parser.add_argument("--e2e-warmup", type=int, default=3)
    parser.add_argument("--e2e-iters", type=int, default=5)
    parser.add_argument("--compile", action="store_true", help="compile compressed Linear modules with torch.compile")
    parser.add_argument("--engine", choices=("hf", "inference"), default="hf", help="e2e backend: hf (default) or the graph-friendly checkpoint runtime")
    parser.add_argument("--e2e-batch", type=int, default=1)
    parser.add_argument("--e2e-kv-bits", type=int, choices=(16, 8, 4), default=None, help="override KV bits for the inference engine")
    parser.add_argument("--e2e-regimes", default="dense,quant,joint", help="comma-separated checkpoint regimes for the inference engine")
    parser.add_argument("--no-sparse-accel", action="store_true", help="force dense/int fallback instead of sparse Tensor Core acceleration")
    parser.add_argument("--e2e-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    # fmt: on


def _build_recipe(args: argparse.Namespace) -> CompressionRecipe:
    clip_rhos = tuple(float(value) for value in args.clip_rhos.split(",") if value)
    return CompressionRecipe(
        model=args.model,
        weight=WeightQuantSpec(
            bits=args.w_bits,
            granularity=_granularity(args.w_gran, args.w_group_size),
            grid=args.grid,
            range=args.range,
            clip_rhos=clip_rhos,
            method=args.quant,
            gptq_alpha=args.gptq_alpha,
            gptq_blocksize=args.gptq_blocksize,
        ),
        activation=ActivationQuantSpec(
            bits=args.a_bits,
            granularity=_granularity(args.a_gran, args.a_group_size),
            grid="midrise" if args.a_bits == 1 else args.a_grid,
            clip_ratio=args.a_clip,
        ),
        kv_cache=KVQuantSpec(
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            granularity=_granularity(args.kv_gran, args.kv_group_size),
            grid="midrise" if min(args.k_bits, args.v_bits) == 1 else args.kv_grid,
            k_clip_ratio=args.k_clip,
            v_clip_ratio=args.v_clip,
        ),
        prune=PruneSpec(
            sparsity=args.sparsity,
            structure=args.structure,
            criterion=args.criterion,
            lam=args.lam,
        ),
        rotation=RotationSpec(mode=args.rotation, seed=args.seed),
        smoothing=SmoothSpec(alpha=None if args.no_smooth else args.smooth_alpha),
        compensation=CompensationSpec(
            method=args.comp,
            alpha=args.comp_alpha,
            shift_scale=args.shift_scale,
            damping=args.damping,
        ),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        max_layers=args.max_layers,
    )


def _devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        return parse_devices(args.devices)
    if args.gpus > 1:
        return [f"cuda:{index}" for index in range(args.gpus)]
    return [args.device]


def _compress(args: argparse.Namespace, recipe: CompressionRecipe) -> int:
    devices = _devices(args)
    paths = compress_regimes(
        recipe,
        manifest_path=args.manifest,
        device=devices[0],
        output_dir=args.checkpoint_dir,
        single=args.single,
        devices=devices,
        parallel=args.parallel,
    )
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)
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
    return run_benchmark(recipe, args)


if __name__ == "__main__":
    raise SystemExit(main())
