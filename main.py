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
    MASK_SELECT,
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


def _block_range(value: str) -> tuple[int, int]:
    first, sep, last = value.partition("-")
    if not sep or not first.isdigit() or not last.isdigit() or int(first) > int(last):
        raise argparse.ArgumentTypeError(f"block range must be FIRST-LAST, got {value}")
    return int(first), int(last)


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
    parser.add_argument("--mask-select", choices=MASK_SELECT, default="score", help="N:M/pairwise mask: score=one-shot criterion; obs=each block's legal pattern by its exact compensated cost")
    parser.add_argument("--w-gran", choices=("tensor", "channel", "group"), default="channel")
    parser.add_argument("--w-group-size", "--group-size", dest="w_group_size", type=int, default=128)
    parser.add_argument("--a-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--kv-gran", choices=("tensor", "token", "group"), default="token")
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midtread", help="grid for weights, and for activations/KV unless --a-grid/--kv-grid is given")
    parser.add_argument("--a-grid", choices=("midrise", "midtread"), default=None, help="activation grid (default: --grid)")
    parser.add_argument("--kv-grid", choices=("midrise", "midtread"), default=None, help="KV-cache grid (default: --grid)")
    parser.add_argument("--range", choices=("full", "clip"), default="clip")
    parser.add_argument("--quant", choices=("rtn", "gptq"), default="gptq")
    parser.add_argument("--comp", choices=("none", "obr", "drift"), default="obr", help="calibration objective: obr=OBR (symmetric on the drifted input), drift=QDrift's mixture (per-layer weights derived by --mix-rule auto unless --mix-rule fixed)")
    parser.add_argument("--mix-asym", type=float, default=None, help="drift, --mix-rule fixed: weight alpha of the asymmetric target E||W_hat X_hat - W X||^2 (default 1; 1 with --mix-clean 0 = the pure asymmetric target)")
    parser.add_argument("--mix-clean", type=float, default=None, help="drift, --mix-rule fixed: weight gamma of the clean-input target E||W_hat X - W X||^2; the symmetric target gets 1 - alpha - gamma (default 0; 0, 0 = OBR)")
    parser.add_argument("--mix-blocks", type=_block_range, default=None, help="drift, --mix-rule fixed: apply the mixture only to decoder blocks FIRST-LAST (inclusive); OBR (0, 0) elsewhere. Localization ablation")
    parser.add_argument("--mix-modules", choices=("all", "attn", "mlp_in", "mlp_out"), default="all", help="drift, --mix-rule fixed: apply the mixture only to attn (q/k/v/o), mlp_in (gate/up) or mlp_out (down); OBR (0, 0) elsewhere")
    parser.add_argument("--mix-rule", choices=("fixed", "auto", "prec"), default=None, help="drift: auto (default) = derive each layer group's (alpha, gamma) from its calibration statistics (docs/mix-auto.md); prec = the same statistics with alpha = eps / (s + eps); fixed = the global --mix-asym/--mix-clean (grid and ablations)")
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
    parser.add_argument("--offload", action="store_true", help="keep the decoder blocks on the host and compress one block at a time on the GPU; for models that do not fit one card (e.g. LLaMA-2-13B on 24 GB). Bit-identical to the on-device path")
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


def _mixture(args: argparse.Namespace) -> dict:
    """QDrift's mixture settings.  ``--comp drift`` derives the per-layer
    weights (``auto``) unless ``--mix-rule fixed`` is given; fixed weights
    are never selected implicitly, so an old ``--mix-asym 1 --mix-clean 0``
    command cannot silently turn into ``auto``."""
    explicit = (
        args.mix_asym is not None or args.mix_clean is not None
        or args.mix_blocks is not None or args.mix_modules != "all"
    )
    rule = args.mix_rule or ("auto" if args.comp == "drift" else "fixed")
    if rule != "fixed" and explicit:
        raise SystemExit("--mix-asym/--mix-clean/--mix-blocks/--mix-modules need --mix-rule fixed")
    return {
        "mix_asym": 1.0 if args.mix_asym is None else args.mix_asym,
        "mix_clean": 0.0 if args.mix_clean is None else args.mix_clean,
        "mix_blocks": args.mix_blocks,
        "mix_modules": args.mix_modules,
        "mix_rule": rule,
    }


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
            select=args.mask_select,
        ),
        rotation=RotationSpec(mode=args.rotation, seed=args.seed),
        compensation=CompensationSpec(
            method=args.comp,
            damping=args.damping,
            **_mixture(args),
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
    parser.add_argument("--ppl", action=argparse.BooleanOptionalAction, default=True, help="evaluate WikiText-2 PPL at 2048 tokens: test (reported) and validation (for design decisions)")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--zeroshot", action=argparse.BooleanOptionalAction, default=False, help="evaluate the lm-eval zero-shot tasks")
    parser.add_argument("--lm-eval-tasks", default=",".join(ZEROSHOT_TASKS))
    parser.add_argument("--drop-checkpoint", action="store_true", help="delete the checkpoint once it is evaluated")
    parser.add_argument("--eval-shard", action="store_true", help="shard the eval model across all visible GPUs (device_map=auto) for models too big for one card; off by default so 7B/8B stays byte-identical")
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
        offload=args.offload,
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
            shard=args.eval_shard,
        )
    if args.ppl:
        record["ppl"] = {}
        for dataset in PPL_DATASETS:
            chunks = get_loaders(
                dataset,
                tokenizer,
                nsamples=None,
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
