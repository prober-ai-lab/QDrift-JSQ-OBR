# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the group-absmax / prune-vs-quant exchange minitest.

``absmax-pruned-vs-sparsity``
    Share of quantization groups whose largest-magnitude weight is pruned
    (so that a scale recomputed after pruning would shrink), per criterion and
    group granularity; mean over layers, with the worst layer as a dotted line.

``better-pruned-than-quantized-g128``
    Share of weights with eps^2 > w^2 (cheaper to prune than to quantize) per
    bit width (mid-rise, group 128), and the share of the retained activation-weighted
    quantization loss carried by such weights under a WANDA mask.

    python visualize/plots/prune_scale_invariance.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import style as S

STEM = "prune_scale_invariance"
GRAN_ORDER = ("g64", "g128", "ch")
GRAN_LABEL = {"g64": "group 64", "g128": "group 128", "ch": "per channel"}
CRITERION_LABEL = {
    "wanda": "WANDA",
    "magnitude": "magnitude",
    "sparsegpt": "SparseGPT",
    "qap": "QAP (W2 step)",
}


def absmax_figure(rows: list[dict], model: str) -> None:
    """Plot the share of groups whose absmax is pruned."""
    criteria = sorted({r["criterion"] for r in rows}, key=lambda c: (c != "wanda", c))
    grans = [g for g in GRAN_ORDER if any(r["gran"] == g for r in rows)]
    fig, axes = S.plt.subplots(
        1, len(criteria), figsize=(2.5 * len(criteria), 2.2), squeeze=False, sharey=True
    )
    for ax, criterion in zip(axes[0], criteria):
        for color, gran in zip(S.SERIES, grans):
            by_s = defaultdict(list)
            for r in rows:
                if (r["criterion"], r["gran"]) == (criterion, gran):
                    by_s[r["sparsity"]].append(1.0 - r["absmax_kept"])
            xs = sorted(by_s)
            mean = [100 * np.mean(by_s[s]) for s in xs]
            worst = [100 * np.max(by_s[s]) for s in xs]
            ax.plot(xs, mean, color=color, marker="o", label=GRAN_LABEL[gran])
            ax.plot(xs, worst, color=color, linestyle=":", linewidth=1.0)
        ax.set_xlabel("sparsity $s$")
        S.panel_label(ax, CRITERION_LABEL.get(criterion, criterion))
    axes[0][0].set_ylabel("groups with absmax pruned (%)")
    axes[0][-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        title="solid: mean, dotted: worst layer",
    )
    S.save(fig, S.figure_name(STEM, model, "absmax-pruned-vs-sparsity"))


def exchange_figure(rows: list[dict], model: str, gran: str = "g128") -> None:
    """Plot the better-pruned-than-quantized statistics (mid-rise)."""
    rows = [r for r in rows if r["gran"] == gran and r["grid"] == "midrise"]
    bits = sorted({r["bits"] for r in rows})
    share_keys = sorted(
        (k for k in rows[0] if k.startswith("loss_share_kept_s")),
        key=lambda k: float(k.rsplit("s", 1)[1]),
    )
    fig, axes = S.plt.subplots(1, 2, figsize=(5.6, 2.3))
    ax = axes[0]
    means = [
        100 * np.mean([r["better_pruned_all"] for r in rows if r["bits"] == b])
        for b in bits
    ]
    ax.bar(range(len(bits)), means, width=0.6, color=S.GRID_COLOR["midrise"])
    for x, m in enumerate(means):
        ax.text(x, m, f"{m:.0f}", ha="center", va="bottom", fontsize=6, color=S.INK_2)
    ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
    ax.set_ylabel("weights (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    S.panel_label(ax, rf"weights with $\epsilon^2>w^2$, {gran}")
    ax = axes[1]
    for color, key in zip(S.BLUE_RAMP[1:], share_keys):
        xs, ys = [], []
        for b in bits:
            values = [r[key] for r in rows if r["bits"] == b]
            if values:
                xs.append(b)
                ys.append(100 * np.mean(values))
        ax.plot(xs, ys, color=color, marker="o", label=f"s = {key.rsplit('s', 1)[1]}")
    ax.set_xticks(bits, [f"W{b}" for b in bits])
    ax.set_ylabel("share of retained quant. loss (%)")
    S.panel_label(ax, r"from WANDA-kept weights with $\epsilon^2>w^2$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"better-pruned-than-quantized-{gran}"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    absmax = S.table(payloads, "absmax", ("layer", "gran", "criterion", "sparsity"))
    exchange = S.table(payloads, "exchange", ("layer", "gran", "grid", "bits"))
    absmax_figure(absmax, args.model)
    exchange_figure(exchange, args.model, "g128")


if __name__ == "__main__":
    main()
