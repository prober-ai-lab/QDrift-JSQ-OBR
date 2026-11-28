# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode/end-to-end timing, throughput, and memory measurement."""

from __future__ import annotations

import statistics

import torch

from benchmark.inference.config import BenchConfig

__all__ = ["measure_e2e", "percentile", "summarize"]


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def _summary_payload(
    cfg: BenchConfig,
    model,
    cache,
    prefill_times: list[float],
    decode_step_times: list[float],
    e2e_times: list[float],
    baseline_allocated: int,
    used_graph: bool,
) -> dict:
    prefill_summary = summarize(prefill_times)
    decode_summary = summarize(decode_step_times)
    e2e_summary = summarize(e2e_times)
    weight_bytes = model.weight_bytes()
    kv_bytes = cache.nbytes()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    prefill_tokens_s = (
        cfg.batch * cfg.prefill_len / (prefill_summary["p50_ms"] * 1e-3)
    )
    decode_tokens_s = cfg.batch / (decode_summary["p50_ms"] * 1e-3)
    e2e_tokens_s = (
        cfg.batch * (cfg.prefill_len + cfg.decode_steps)
        / (e2e_summary["p50_ms"] * 1e-3)
    )

    return {
        "timing_mode": "cuda_graph" if used_graph else "eager",
        "prefill": {
            **prefill_summary,
            "tokens_per_s": prefill_tokens_s,
        },
        "decode": {
            **decode_summary,
            "tokens_per_s": decode_tokens_s,
            "ms_per_token": decode_summary["p50_ms"],
            "tokens_per_s_per_sequence": decode_tokens_s / cfg.batch,
        },
        "e2e": {
            **e2e_summary,
            "ttft_ms": prefill_summary["p50_ms"],
            "tpot_ms": decode_summary["p50_ms"],
            "tokens_per_s": e2e_tokens_s,
            "requests_per_s": cfg.batch / (e2e_summary["p50_ms"] * 1e-3),
        },
        "memory": {
            "weight_bytes": weight_bytes,
            "kv_bytes": kv_bytes,
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    }


@torch.inference_mode()
def measure_e2e(model, cfg: BenchConfig) -> dict:
    max_len = cfg.prefill_len + cfg.decode_steps + 1
    input_ids = torch.randint(
        0,
        model.spec.vocab_size,
        (cfg.batch, cfg.prefill_len),
        device=cfg.device,
    )
    next_ids = torch.zeros(
        (cfg.batch, 1), dtype=torch.long, device=cfg.device
    )

    def new_cache():
        return model.new_cache(cfg.batch, max_len, bits=cfg.kv_bits)

    def run_prefill(cache):
        return model.prefill(input_ids, cache)

    def run_decode(cache):
        for _ in range(cfg.decode_steps):
            model.decode(next_ids, cache)

    # Eager warmup: initialize cuBLAS, CUTLASS, Marlin and SDPA handles.  Each cache
    # is dropped before the next is allocated, so two never coexist (a batch-16 FP16
    # KV cache is several GB).
    cache = None
    for _ in range(cfg.warmup):
        cache = None
        cache = new_cache()
        run_prefill(cache)
        run_decode(cache)
    torch.cuda.synchronize()

    used_graph = False
    graph_cache = None
    g_prefill = g_decode = None
    if cfg.cuda_graph:
        try:
            torch.cuda.synchronize()
            cache = None
            graph_cache = new_cache()
            g_prefill = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_prefill):
                run_prefill(graph_cache)
            g_prefill.replay()
            torch.cuda.synchronize()

            g_decode = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_decode):
                run_decode(graph_cache)
            g_decode.replay()
            torch.cuda.synchronize()
            used_graph = True
        except Exception as error:  # graph capture is best-effort
            print(f"[warn] CUDA graph capture failed: {error!r}")
            print("[warn] falling back to eager timing")
            used_graph = False

    # Avoid empty_cache() here: if graph capture failed, the allocator may
    # still have an unfinished capture and empty_cache() asserts.
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    prefill_times: list[float] = []
    decode_step_times: list[float] = []
    e2e_times: list[float] = []
    cache = graph_cache

    if used_graph:
        assert g_prefill is not None and g_decode is not None
        for _ in range(cfg.trials):
            start, end = _event_pair()
            start.record()
            g_prefill.replay()
            end.record()
            end.synchronize()
            prefill_ms = start.elapsed_time(end)

            start, end = _event_pair()
            start.record()
            g_decode.replay()
            end.record()
            end.synchronize()
            decode_ms = start.elapsed_time(end)

            prefill_times.append(prefill_ms)
            decode_step_times.append(decode_ms / cfg.decode_steps)
            e2e_times.append(prefill_ms + decode_ms)
    else:
        for _ in range(cfg.trials):
            cache = None
            cache = new_cache()
            start, end = _event_pair()
            start.record()
            run_prefill(cache)
            end.record()
            end.synchronize()
            prefill_ms = start.elapsed_time(end)

            step_times = []
            for _ in range(cfg.decode_steps):
                start, end = _event_pair()
                start.record()
                model.decode(next_ids, cache)
                end.record()
                end.synchronize()
                step_times.append(start.elapsed_time(end))

            prefill_times.append(prefill_ms)
            decode_step_times.extend(step_times)
            e2e_times.append(prefill_ms + sum(step_times))

    assert cache is not None
    return _summary_payload(
        cfg,
        model,
        cache,
        prefill_times,
        decode_step_times,
        e2e_times,
        baseline_allocated,
        used_graph,
    )
