# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motivation figure: the quality/efficiency trade-off across the four compression
regimes on LLaMA-2-7B — no compression, sparsification only, quantization only, and
the joint regime QDrift targets. Two panels: WikiText-2 perplexity (degradation,
higher is worse) and weight memory (efficiency, lower is better).

Numbers are read live from the inference-runtime benchmark (`main benchmark`): PPL from
the `ppl` task (`results/benchmarks/*__ppl.json`), weight and peak memory from the `e2e`
task (`*__e2e.json`).  The sparsify-only PPL is the one measured constant (see
SPARSE_ONLY_PPL); the benchmark's own `sparse` entry is the dense model.

    python visualize/plots/regime_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import style as S  # noqa: E402

MODEL = "llama2-7b"
BENCH = Path(__file__).resolve().parents[2] / "results" / "benchmarks"
# One-off measurement, not saved as a record: fp16 LLaMA-2-7B with a 50% unstructured
# SparseGPT mask and no quantization, WikiText-2 test at 2048.
SPARSE_ONLY_PPL = 6.39


def _bench(task: str) -> dict:
    """The single LLaMA-2-7B W4A4 s0.5 OBR run of one benchmark task."""
    paths = sorted(BENCH.glob(f"{MODEL}__0.5w4a4kv4__*__obr__*__{task}.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected one {task} benchmark for {MODEL}, found {len(paths)}")
    runs = json.loads(paths[0].read_text(encoding="utf-8"))["runs"]
    return next(iter(runs.values()))


def regimes() -> tuple[list[tuple[str, float, float]], dict[str, float]]:
    """``[(label, test PPL, weight bytes)]`` for the four regimes, and the memory
    record (``fp16``/``w4a4`` weight bytes, ``*_peak`` peak allocated bytes)."""
    ppl = _bench("ppl")["accuracy"]["ppl"]["wikitext2"]
    eff = _bench("e2e")["efficiency"]
    mem = {
        "fp16": eff["dense"]["memory"]["weight_bytes"],
        "w4a4": eff["w4a4"]["memory"]["weight_bytes"],
        "fp16_peak": eff["dense"]["memory"]["peak_allocated_bytes"],
        "w4a4_peak": eff["w4a4"]["memory"]["peak_allocated_bytes"],
    }
    rows = [
        ("Dense", ppl["dense"], mem["fp16"]),
        ("Sparse\n50%", SPARSE_ONLY_PPL, mem["fp16"]),
        ("W4A4", ppl["quant"], mem["w4a4"]),
        ("Sparse\n+ W4A4", ppl["joint"], mem["w4a4"]),
    ]
    return rows, mem


def main() -> None:
    S.setup()
    rows, _ = regimes()
    labels = [r[0] for r in rows]
    ppl = [r[1] for r in rows]
    mem = [r[2] / 1e9 for r in rows]
    x = list(range(len(rows)))

    # Two side-by-side panels sharing one paper column.
    fig, (axp, axm) = S.plt.subplots(1, 2, figsize=(3.4, 1.9))

    # Panel A: perplexity (degradation).
    cppl = ["#8a8f98", "#8a8f98", "#d98c5f", "#c0504d"]
    axp.bar(x, ppl, color=cppl, width=0.64, zorder=3)
    for xi, v in zip(x, ppl):
        axp.text(xi, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    axp.axhline(ppl[0], ls="--", lw=0.9, color="#8a8f98", zorder=1)
    axp.set_xticks(x, labels, fontsize=5.5)
    axp.set_ylabel("WikiText-2 perplexity", fontsize=6.5)
    axp.set_ylim(0, max(ppl) * 1.18)
    axp.set_title("Quality", fontsize=7.5)
    axp.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    # Panel B: weight memory (efficiency).
    cmem = ["#8a8f98", "#8a8f98", "#5b8c6e", "#5b8c6e"]
    axm.bar(x, mem, color=cmem, width=0.64, zorder=3)
    for xi, v in zip(x, mem):
        axm.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    axm.set_xticks(x, labels, fontsize=5.5)
    axm.set_ylabel("GB", fontsize=6.5)
    axm.set_ylim(0, max(mem) * 1.18)
    axm.set_title("Weight memory", fontsize=7.5)
    axm.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    axm.annotate(rf"${mem[0] / mem[2]:.1f}\times$", xy=(2, mem[2]), xytext=(2.1, 8.5), fontsize=7,
                 color="#5b8c6e", ha="center")

    for ax in (axp, axm):
        ax.tick_params(axis="y", labelsize=5.5)
    fig.tight_layout(w_pad=0.6)
    S.save(fig, f"results__regime_tradeoff__{MODEL}__ppl-and-memory")


if __name__ == "__main__":
    main()
