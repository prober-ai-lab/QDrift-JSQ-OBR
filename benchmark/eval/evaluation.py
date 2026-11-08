# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy metrics and the unified evaluation pipeline."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from benchmark.eval.runtime import (
    checkpoint_stats,
    detect_sparse_backend,
    hardware_fingerprint,
    load_runtime_model,
    measure_runtime,
    precompile_runtime,
)
from benchmark.kernels.lib.registry import kernel_plan
from qdrift.compress.recipe import CompressionRecipe, variant_recipes
from qdrift.data.dataloader import PPL_DATASETS, PPL_SPLITS, get_loaders

__all__ = [
    "ACCURACY_TASKS",
    "EFFICIENCY_TASKS",
    "PPL_DATASETS",
    "REGIMES",
    "ZEROSHOT_TASKS",
    "build_accuracy_result",
    "evaluate_regimes",
    "run_benchmark",
]

ACCURACY_TASKS = ("ppl", "zeroshot")
EFFICIENCY_TASKS = ("e2e",)
REGIMES = ("dense", "sparse", "quant", "joint")
ZEROSHOT_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
)
MANIFEST_PATH = "modelzoo/manifest.yaml"
CHECKPOINT_DIR = Path("checkpoints")
BENCHMARK_DIR = Path("results/benchmarks")
DEVICE = "cuda"


@torch.no_grad()
def ppl_eval(
    model: torch.nn.Module,
    chunks: list[torch.Tensor],
    device: str,
    batch_size: int = 1,
) -> tuple[float, int]:
    """Evaluate fixed-length, non-overlapping PPL chunks.

    The loss is summed in fp32 over every predicted token.  An fp16 loss
    rounds each batch mean to ~2e-3 nats, which quantizes PPL into ~0.1%
    steps and hides real differences between recipes.
    """
    if not chunks:
        raise ValueError("no evaluation chunks")
    nll = 0.0
    tokens = 0
    for start in range(0, len(chunks), batch_size):
        inputs = torch.stack(chunks[start : start + batch_size]).to(device)
        logits = model(inputs).logits
        # One row at a time keeps the fp32 copy of the logits small.
        for row in range(inputs.shape[0]):
            nll += torch.nn.functional.cross_entropy(
                logits[row, :-1].float(),
                inputs[row, 1:],
                reduction="sum",
            ).item()
            tokens += inputs.shape[1] - 1
    return math.exp(nll / tokens), tokens


def zeroshot_eval(
    model: torch.nn.Module,
    tokenizer,
    tasks: tuple[str, ...],
    batch_size: int,
) -> dict[str, float]:
    """Run the FlatQuant/LM-Eval zero-shot task set."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError(
            "zero-shot evaluation requires the optional lm-eval package"
        ) from error
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    result = lm_eval.simple_evaluate(
        hflm,
        tasks=list(tasks),
        batch_size=batch_size,
    )["results"]
    scores: dict[str, float] = {}
    for task in tasks:
        entry = result[task]
        value = entry.get("acc_norm,none", entry.get("acc,none"))
        if value is None:
            raise ValueError(f"lm-eval task {task} has no accuracy metric")
        scores[task] = float(value) * 100.0
    scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def build_accuracy_result(
    values: dict[str, dict[str, object]],
    tasks: tuple[str, ...],
    ppl_datasets: tuple[str, ...],
    ppl_seqlen: int,
    zeroshot_tasks: tuple[str, ...],
) -> dict[str, object]:
    """Format per-regime PPL and zero-shot values."""
    result: dict[str, object] = {}
    if "ppl" in tasks:
        ppl_result: dict[str, object] = {}
        for dataset in ppl_datasets:
            dense = values["dense"]["ppl"][dataset]
            assert isinstance(dense, tuple)
            dense_ppl = dense[0]
            entry: dict[str, object] = {
                "dataset": dataset,
                "split": PPL_SPLITS[dataset],
                "seqlen": ppl_seqlen,
                "chunks": dense[1] // (ppl_seqlen - 1),
                "tokens": dense[1],
                "dense": dense_ppl,
            }
            for regime in REGIMES[1:]:
                entry[regime] = values[regime]["ppl"][dataset][0]
            entry["delta"] = {
                regime: entry[regime] - dense_ppl for regime in REGIMES[1:]
            }
            entry["ratio"] = {
                regime: entry[regime] / dense_ppl for regime in REGIMES[1:]
            }
            ppl_result[dataset] = entry
        result["ppl"] = ppl_result
    if "zeroshot" in tasks:
        zeroshot: dict[str, object] = {"tasks": list(zeroshot_tasks)}
        dense = values["dense"]["zeroshot"]
        assert isinstance(dense, dict)
        for regime in REGIMES:
            zeroshot[regime] = values[regime]["zeroshot"]
        zeroshot["delta"] = {
            regime: {
                task: values[regime]["zeroshot"][task] - dense[task]
                for task in (*zeroshot_tasks, "avg")
            }
            for regime in REGIMES[1:]
        }
        zeroshot["ratio"] = {
            regime: {
                task: values[regime]["zeroshot"][task] / dense[task]
                for task in (*zeroshot_tasks, "avg")
            }
            for regime in REGIMES[1:]
        }
        result["zeroshot"] = zeroshot
    return result


def evaluate_regimes(
    paths: dict[str, Path],
    manifest: str = "modelzoo/manifest.yaml",
    device: str = "cuda",
    tasks: tuple[str, ...] = ("ppl", "zeroshot", "e2e"),
    ppl_datasets: tuple[str, ...] = PPL_DATASETS,
    ppl_seqlen: int = 2048,
    ppl_chunks: int | None = None,
    zeroshot_tasks: tuple[str, ...] = ZEROSHOT_TASKS,
    batch_size: int = 16,
    seed: int = 42,
    e2e_seqlen: int = 1024,
    e2e_decode_tokens: int = 64,
    e2e_warmup: int = 3,
    e2e_iters: int = 5,
    compile_model: bool = False,
    allow_sparse: bool = True,
    efficiency_regimes: tuple[str, ...] = REGIMES,
) -> dict[str, object]:
    """Run accuracy and end-to-end efficiency on four real checkpoints."""
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "evaluation needs the four real checkpoints; missing:\n  "
            + "\n  ".join(missing)
        )

    accuracy_tasks = tuple(task for task in tasks if task in ACCURACY_TASKS)
    efficiency_requested = any(task in tasks for task in EFFICIENCY_TASKS)
    values: dict[str, dict[str, object]] = {
        regime: {"ppl": {}, "zeroshot": {}} for regime in REGIMES
    }
    efficiency: dict[str, object] = {}
    chunks: dict[str, list[torch.Tensor]] | None = None

    for regime in REGIMES:
        if not accuracy_tasks and regime not in efficiency_regimes:
            continue
        path = paths[regime]
        config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        torch.cuda.empty_cache()
        start = time.perf_counter()
        model, tokenizer, loaded_recipe = load_runtime_model(
            path,
            regime,
            manifest_path=manifest,
            device=device,
            compile_model=compile_model,
            allow_sparse=allow_sparse,
        )
        load_s = time.perf_counter() - start
        precompile_s = precompile_runtime(model)
        try:
            if accuracy_tasks and chunks is None:
                chunks = {
                    dataset: get_loaders(
                        dataset,
                        tokenizer,
                        nsamples=ppl_chunks,
                        seqlen=ppl_seqlen,
                        eval_mode=True,
                        seed=seed,
                    )
                    for dataset in ppl_datasets
                }
            if "ppl" in accuracy_tasks:
                values[regime]["ppl"] = {
                    dataset: ppl_eval(
                        model,
                        chunks[dataset],
                        device=device,
                        batch_size=batch_size,
                    )
                    for dataset in ppl_datasets
                }
            if "zeroshot" in accuracy_tasks:
                values[regime]["zeroshot"] = zeroshot_eval(
                    model,
                    tokenizer,
                    zeroshot_tasks,
                    batch_size,
                )
            if efficiency_requested and regime in efficiency_regimes:
                torch.cuda.reset_peak_memory_stats()
                stats = measure_runtime(
                    model,
                    seqlen=e2e_seqlen,
                    decode_tokens=e2e_decode_tokens,
                    warmup=e2e_warmup,
                    iters=e2e_iters,
                    device=device,
                    kv_cache=loaded_recipe.kv_cache,
                    batch=batch_size,
                )
                stats["load_s"] = load_s
                stats["precompile_s"] = precompile_s
                stats["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
                stats.update(checkpoint_stats(path, config))
                efficiency[regime] = stats
        finally:
            del model
            torch.cuda.empty_cache()

    result: dict[str, object] = {}
    if accuracy_tasks:
        result["accuracy"] = build_accuracy_result(
            values,
            accuracy_tasks,
            ppl_datasets,
            ppl_seqlen,
            zeroshot_tasks,
        )
    if efficiency_requested:
        result["efficiency"] = efficiency
    return result


FORMAT = "qdrift-benchmarks"
SCHEMA_VERSION = 1
REGIME_ALIASES_16 = {
    "dense": "dense",
    "sparse": "sparse_fp16",
    "quant": "int4_marlin",
    "joint": "joint_sparse_int4",
}
REGIME_ALIASES_4 = {
    "dense": "dense",
    "sparse": "sparse_fp16",
    "quant": "w4a4",
    "joint": "joint_sparse_w4a4",
}
ALL_TASKS = ACCURACY_TASKS + EFFICIENCY_TASKS


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _empty() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "runs": {},
    }


def load_store(path: str | Path) -> dict[str, Any]:
    """Load the store, creating an empty structure if it does not exist."""
    output = Path(path)
    if not output.exists():
        return _empty()
    data = json.loads(output.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(
            f"{output} is not a {FORMAT} file (format={data.get('format')!r})"
        )
    data.setdefault("runs", {})
    return data


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def write_store(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write the store."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def upsert_run(
    path: str | Path,
    checkpoint_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Merge ``payload`` into ``runs[checkpoint_id]`` and return the store."""
    data = load_store(path)
    run = data["runs"].setdefault(checkpoint_id, {})
    _merge(run, payload)
    run["updated_at"] = _now()
    data["updated_at"] = run["updated_at"]
    write_store(path, data)
    return data


def _parse_tasks(value: str) -> tuple[str, ...]:
    if value.strip() == "all":
        return ALL_TASKS
    tasks: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in ALL_TASKS:
            raise SystemExit(f"unknown task {item!r}; choose from {ALL_TASKS}")
        if item not in tasks:
            tasks.append(item)
    return tuple(tasks)


def _regime_paths(
    recipe: CompressionRecipe,
    checkpoint_dir: Path,
) -> dict[str, Path]:
    """Resolve the four checkpoints selected by the shared recipe."""
    return {
        name: checkpoint_dir / f"{variant.checkpoint_id()}.safetensors"
        for name, variant in variant_recipes(recipe).items()
    }


def _summary(path: Path, payload: dict[str, object]) -> str:
    parts = [path.name]
    accuracy = payload.get("accuracy")
    if isinstance(accuracy, dict):
        ppl = accuracy.get("ppl")
        if isinstance(ppl, dict):
            for dataset, entry in ppl.items():
                if not isinstance(entry, dict):
                    continue
                parts.append(
                    f"ppl {dataset} "
                    + " ".join(
                        f"{name}={entry[name]:.3f}"
                        for name in ("dense", "sparse", "quant", "joint")
                        if isinstance(entry.get(name), (int, float))
                    )
                )
        zeroshot = accuracy.get("zeroshot")
        if isinstance(zeroshot, dict):
            parts.append(
                "zeroshot "
                + " ".join(
                    f"{name}={zeroshot[name]['avg']:.2f}"
                    for name in ("dense", "sparse", "quant", "joint")
                    if isinstance(zeroshot.get(name), dict)
                )
            )
    efficiency = payload.get("efficiency")
    aliases = payload.get("regime_aliases", REGIME_ALIASES_16)
    if isinstance(efficiency, dict):
        for regime in ("dense", "sparse", "quant", "joint"):
            entry = efficiency.get(regime) or efficiency.get(aliases.get(regime, regime))
            if not isinstance(entry, dict):
                continue
            prefill = entry.get("prefill")
            decode = entry.get("decode")
            if isinstance(prefill, dict) and isinstance(decode, dict):
                parts.append(
                    f"{regime}:"
                    f"prefill {prefill.get('p50_ms'):.1f}ms/"
                    f"{prefill.get('tokens_per_s'):.1f}tok/s "
                    f"decode {decode.get('p50_ms'):.1f}ms/"
                    f"{decode.get('tokens_per_s'):.1f}tok/s"
                )
    return " ".join(parts)


def _benchmark_output_path(recipe: CompressionRecipe, tasks: tuple[str, ...]) -> Path:
    suffix = "-".join(tasks) if tasks else "all"
    return BENCHMARK_DIR / f"{recipe.checkpoint_id()}__{suffix}.json"


def run_benchmark(recipe: CompressionRecipe, args: Any) -> int:
    """Resolve the four checkpoints and write the benchmark result."""
    tasks = _parse_tasks(args.tasks)
    sparse_backend = detect_sparse_backend(DEVICE)
    structure = recipe.prune.structure
    structured_sparse = structure in ("2:4", "4:8")
    hardware_sparse = structure in ("2:4", "4:8")
    allow_sparse = (
        hardware_sparse
        and bool(sparse_backend["available"])
        and not getattr(args, "no_sparse_accel", False)
    )
    if structured_sparse:
        if allow_sparse:
            print(
                f"sparse Tensor Core backend: {sparse_backend['backend']} "
                f"({sparse_backend['device']}, sm {sparse_backend['capability']})",
                flush=True,
            )
        elif structure == "4:8":
            print(
                "OBR 4:8 is not representable by 4:8 sparse Tensor Cores; "
                "joint uses dense INT4 (channel/tensor scales) or INT8. "
                "Use --pairwise-fraction 1.0 for hardware sparse acceleration.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "sparse Tensor Core acceleration unavailable or disabled; "
                "sparse/joint e2e will be skipped, accuracy is still evaluated",
                file=sys.stderr,
                flush=True,
            )
    if allow_sparse:
        efficiency_regimes = REGIMES
    elif structure in ("2:4", "4:8"):
        # Arbitrary N:M OBR masks are dense-INT4 accelerated in joint.
        efficiency_regimes = ("dense", "quant", "joint")
    else:
        efficiency_regimes = ("dense", "quant")

    paths = _regime_paths(recipe, CHECKPOINT_DIR)
    output_path = _benchmark_output_path(recipe, tasks)
    if args.dry_run:
        for name, path in paths.items():
            print(f"{name}: {path}")
        print(f"output: {output_path}")
        return 0

    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        print(
            "checkpoint(s) not found for this recipe:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "run scripts/compress.sh with the same recipe before benchmarking",
            file=sys.stderr,
        )
        return 2

    no_sidecar = [
        str(path.with_suffix(".json"))
        for path in paths.values()
        if not path.with_suffix(".json").exists()
    ]
    if no_sidecar:
        print(
            "missing sidecar metadata for:\n  " + "\n  ".join(no_sidecar),
            file=sys.stderr,
        )
        return 2

    joint_path = paths["joint"]
    config = json.loads(joint_path.with_suffix(".json").read_text(encoding="utf-8"))
    checkpoint_id = str(config.get("checkpoint_id", joint_path.stem))
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{args.eval_seed:06x}"
    accuracy_tasks = tuple(task for task in tasks if task in ACCURACY_TASKS)
    efficiency_requested = "e2e" in tasks
    if getattr(args, "engine", "hf") == "inference":
        from benchmark.inference.checkpoint import evaluate_inference

        metrics: dict[str, object] = {}
        if accuracy_tasks:
            metrics.update(
                evaluate_regimes(
                    paths,
                    manifest=MANIFEST_PATH,
                    device=DEVICE,
                    tasks=accuracy_tasks,
                    ppl_datasets=tuple(
                        item.strip()
                        for item in args.ppl_datasets.split(",")
                        if item.strip()
                    ),
                    ppl_seqlen=args.ppl_seqlen,
                    ppl_chunks=args.ppl_chunks,
                    zeroshot_tasks=tuple(
                        item.strip()
                        for item in args.lm_eval_tasks.split(",")
                        if item.strip()
                    ),
                    batch_size=recipe.batch_size,
                    seed=args.eval_seed,
                    allow_sparse=allow_sparse,
                    efficiency_regimes=efficiency_regimes,
                )
            )
        if efficiency_requested:
            metrics.update(
                evaluate_inference(
                    recipe,
                    paths,
                    manifest_path=MANIFEST_PATH,
                    device=DEVICE,
                    batch=recipe.batch_size,
                    prefill_len=args.e2e_seqlen,
                    decode_steps=args.e2e_decode_tokens,
                    warmup=args.e2e_warmup,
                    trials=args.e2e_iters,
                    kv_bits=None,
                    regimes=tuple(
                        item.strip()
                        for item in getattr(args, "e2e_regimes", "dense,sparse,quant,joint").split(",")
                        if item.strip() and item.strip() in efficiency_regimes
                    ),
                    allow_sparse=allow_sparse,
                    cuda_graph=getattr(args, "e2e_cuda_graph", True),
                )
            )
    else:
        metrics = evaluate_regimes(
            paths,
            manifest=MANIFEST_PATH,
            device=DEVICE,
            tasks=tasks,
            ppl_datasets=tuple(
                item.strip() for item in args.ppl_datasets.split(",") if item.strip()
            ),
            ppl_seqlen=args.ppl_seqlen,
            ppl_chunks=args.ppl_chunks,
            zeroshot_tasks=tuple(
                item.strip() for item in args.lm_eval_tasks.split(",") if item.strip()
            ),
            batch_size=recipe.batch_size,
            seed=args.eval_seed,
            e2e_seqlen=args.e2e_seqlen,
            e2e_decode_tokens=args.e2e_decode_tokens,
            e2e_warmup=args.e2e_warmup,
            e2e_iters=args.e2e_iters,
            compile_model=getattr(args, "compile", False),
            allow_sparse=allow_sparse,
            efficiency_regimes=efficiency_regimes,
        )
    aliases = (
        REGIME_ALIASES_16
        if recipe.activation.bits >= 16
        else REGIME_ALIASES_4
    )
    if getattr(args, "engine", "hf") == "inference":
        efficiency = metrics.get("efficiency")
        if isinstance(efficiency, dict):
            metrics["efficiency"] = {
                aliases.get(name, name): entry
                for name, entry in efficiency.items()
            }

    payload: dict[str, object] = {
        "checkpoint": str(joint_path),
        "checkpoint_id": checkpoint_id,
        "model": recipe.model,
        "mechanism": recipe.compensation.mechanism_tag(),
        "recipe": recipe.to_dict(),
        "recipe_hash": recipe.recipe_hash(),
        "provenance": {
            "run_id": run_id,
            "generated_by": "main.py benchmark",
            "tasks": list(tasks),
        },
        "hardware": hardware_fingerprint(),
        "sparse_backend": sparse_backend,
        "kernel_plan": kernel_plan(recipe, DEVICE, allow_sparse=allow_sparse),
        "graph": (
            "cuda_graph"
            if getattr(args, "engine", "hf") == "inference"
            and getattr(args, "e2e_cuda_graph", True)
            else "compiled"
            if getattr(args, "compile", False)
            else "eager"
        ),
        "regime_aliases": aliases,
        "sparse_acceleration": allow_sparse,
        "efficiency_regimes": list(efficiency_regimes),
        **metrics,
    }

    if args.replace:
        store = load_store(output_path)
        store["runs"].pop(checkpoint_id, None)
        write_store(output_path, store)
    upsert_run(output_path, checkpoint_id, payload)
    print(_summary(joint_path, payload), flush=True)
    print(f"benchmarked {checkpoint_id} -> {output_path}", flush=True)
    return 0
