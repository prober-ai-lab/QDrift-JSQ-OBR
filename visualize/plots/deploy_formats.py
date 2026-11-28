# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment formats on an RTX 4090 (CUDA graphs), read from the
``results/benchmarks/formats__*`` records of ``benchmark/inference/formats.py``.

Three panels, one bar group per model and one bar per format: weight memory,
decode latency per token at batch 1, and decode throughput at batch 16.  FP16
LLaMA-2-13B does not fit on the 24 GB card and is marked instead of drawn.
OBR and QDrift share every format, so the figure holds for both.

    python visualize/plots/deploy_formats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tables"))

import style as S  # noqa: E402
import paper_tables as P  # noqa: E402

PANELS = (
    ("weights (GB)", 1, lambda r: r["memory"]["weight_bytes"] / 1e9),
    ("decode, batch 1 (ms/token)", 1, lambda r: r["decode"]["ms_per_token"]),
    ("decode, batch 16 (tokens/s)", 16, lambda r: r["decode"]["tokens_per_s"]),
)
COLORS = (S.MUTED, S.SERIES[0], S.SERIES[2], S.SERIES[1])


def main() -> None:
    S.setup()
    models = list(P.MODEL_DISP)
    if not any(P.load_format(m, f, 1) for m in models for f, _ in P.FORMATS):
        print("deploy_formats: no formats__* records, skipping")
        return
    fig, axes = S.plt.subplots(1, len(PANELS), figsize=(6.9, 2.2))
    width = 0.8 / len(P.FORMATS)
    for ax, (title, batch, value) in zip(axes, PANELS):
        top = 0.0
        for j, ((fmt, label), color) in enumerate(zip(P.FORMATS, COLORS)):
            for i, model in enumerate(models):
                x = i + (j - (len(P.FORMATS) - 1) / 2) * width
                record = P.load_format(model, fmt, batch)
                if record is None:
                    if fmt == "fp16" and P.load_format(model, "w4a4-s50", batch):
                        ax.annotate("OOM", (x, 0), xytext=(0, 2), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=5, rotation=90,
                                    color=S.INK_2)
                    continue
                v = value(record)
                top = max(top, v)
                ax.bar(x, v, width=width * 0.92, color=color,
                       label=label if i == 0 else None)
        ax.set_xticks(range(len(models)), [P.MODEL_DISP[m] for m in models], fontsize=5.5,
                      rotation=15, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, top * 1.12)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="x", visible=False)
        S.panel_label(ax, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(P.FORMATS), frameon=False)
    S.save(fig, "results__deploy_formats__all__memory-latency-throughput")


if __name__ == "__main__":
    main()
