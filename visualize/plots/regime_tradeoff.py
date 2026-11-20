# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motivation figure: the quality/efficiency trade-off across the four compression
regimes on LLaMA-2-7B — no compression, sparsification only, quantization only, and
the joint regime QDrift targets. Two panels: WikiText-2 perplexity (degradation,
higher is worse) and weight memory (efficiency, lower is better).

Numbers are measured by the inference-runtime benchmark (`main benchmark`): PPL from
the `ppl` task (`results/benchmarks/*__ppl.json`), weight memory from the `e2e` task
(`*__e2e.json`). They are one-off and hard-coded here rather than read live.

    python visualize/plots/regime_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style as S  # noqa: E402

MODEL = "llama2-7b"

# (label, WikiText-2 test PPL, weight memory GB = e2e weight_bytes / 1e9).  7B W4A4 unstructured s0.5, OBR.
# PPL: dense/quant/joint measured; sparse = fp16 + 50% unstructured sparsegpt mask.
REGIMES = [
    ("No\ncompression", 5.47, 13.5),
    ("Sparsify\nonly (50%)", 6.39, 13.5),   # measured: fp16 + 50% unstructured sparsegpt mask
    ("Quantize\nonly (W4A4)", 6.15, 3.8),
    ("Sparsify +\nquantize", 7.74, 3.8),
]


def main() -> None:
    S.setup()
    labels = [r[0] for r in REGIMES]
    ppl = [r[1] for r in REGIMES]
    mem = [r[2] for r in REGIMES]
    x = list(range(len(REGIMES)))

    fig, (axp, axm) = S.plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Panel A: perplexity (degradation).
    cppl = ["#8a8f98", "#8a8f98", "#d98c5f", "#c0504d"]
    axp.bar(x, ppl, color=cppl, width=0.64, zorder=3)
    for xi, v in zip(x, ppl):
        axp.text(xi, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axp.axhline(ppl[0], ls="--", lw=0.9, color="#8a8f98", zorder=1)
    axp.set_xticks(x, labels, fontsize=8)
    axp.set_ylabel("WikiText-2 perplexity")
    axp.set_ylim(0, max(ppl) * 1.18)
    axp.set_title("Quality degradation", fontsize=9)
    axp.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    # Panel B: weight memory (efficiency).
    cmem = ["#8a8f98", "#8a8f98", "#5b8c6e", "#5b8c6e"]
    axm.bar(x, mem, color=cmem, width=0.64, zorder=3)
    for xi, v in zip(x, mem):
        axm.text(xi, v + 0.2, f"{v:.1f} GB", ha="center", va="bottom", fontsize=8)
    axm.set_xticks(x, labels, fontsize=8)
    axm.set_ylabel("Weight memory (GB)")
    axm.set_ylim(0, max(mem) * 1.18)
    axm.set_title("Efficiency (memory)", fontsize=9)
    axm.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    axm.annotate(rf"${mem[0] / mem[2]:.1f}\times$", xy=(2, mem[2]), xytext=(2.1, 8.5), fontsize=9,
                 color="#5b8c6e", ha="center")

    fig.tight_layout()
    S.save(fig, f"results__regime_tradeoff__{MODEL}__ppl-and-memory")


if __name__ == "__main__":
    main()
