# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the group-absmax / prune-vs-quant exchange minitest.

``absmax-pruned-vs-sparsity``
    Share of quantization groups whose largest-magnitude weight is pruned
    (so that a scale recomputed after pruning would shrink), per criterion and
    group granularity; mean over layers, with the worst layer as a dotted line.

``better-pruned-than-quantized``
    Share of weights with eps^2 > w^2 (cheaper to prune than to quantize) per
    bit width and grid, and the share of the retained activation-weighted
    quantization loss carried by such weights under a WANDA mask.

``qap-mask-vs-wanda``
    Share of the QAP pruned set shared with WANDA's, and the share of weights
    QAP scores negative (expected rounding error above their magnitude), per
    bit width and grid.

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
    """Plot the better-pruned-than-quantized statistics.

    Mid-tread never has eps^2 > w^2: its error is at most s/2, and a weight
    rounded to 0 has eps^2 == w^2, i.e. the quantizer prunes it itself.  The
    mid-tread bar therefore shows that rounded-to-zero share (hatched).
    """
    rows = [r for r in rows if r["gran"] == gran]
    bits = sorted({r["bits"] for r in rows})
    share_keys = sorted(
        (k for k in rows[0] if k.startswith("loss_share_kept_s")),
        key=lambda k: float(k.rsplit("s", 1)[1]),
    )
    bars = (
        ("midrise", "better_pruned_all", r"mid-rise: $\epsilon^2>w^2$", None),
        ("midtread", "rounded_to_zero", "mid-tread: rounded to 0", "////"),
    )
    fig, axes = S.plt.subplots(1, 2, figsize=(5.6, 2.3))
    ax = axes[0]
    width = 0.4
    for k, (grid, key, label, hatch) in enumerate(bars):
        xs, means = [], []
        for x, b in enumerate(bits):
            values = [r[key] for r in rows if (r["grid"], r["bits"]) == (grid, b)]
            if values:
                xs.append(x + (k - 0.5) * width)
                means.append(100 * np.mean(values))
        ax.bar(
            xs,
            means,
            width=width * 0.92,
            color="white" if hatch else S.GRID_COLOR[grid],
            edgecolor=S.GRID_COLOR[grid],
            hatch=hatch,
            linewidth=0.8,
            label=label,
        )
        for x, m in zip(xs, means):
            ax.text(
                x, m, f"{m:.0f}", ha="center", va="bottom", fontsize=6, color=S.INK_2
            )
    ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
    ax.set_ylabel("weights (%)")
    ax.set_ylim(0, 118)  # headroom for the legend above the bars
    ax.legend(loc="upper right")
    S.panel_label(ax, f"cheaper to prune than to quantize, {gran}")
    ax = axes[1]
    for color, key in zip(S.BLUE_RAMP[1:], share_keys):
        xs, ys = [], []
        for b in bits:
            values = [r[key] for r in rows if (r["grid"], r["bits"]) == ("midrise", b)]
            if values:
                xs.append(b)
                ys.append(100 * np.mean(values))
        ax.plot(xs, ys, color=color, marker="o", label=f"s = {key.rsplit('s', 1)[1]}")
    ax.set_xticks(bits, [f"W{b}" for b in bits])
    ax.set_ylabel("share of retained quant. loss (%)")
    S.panel_label(ax, r"from WANDA-kept weights with $\epsilon^2>w^2$ (mid-rise)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"better-pruned-than-quantized-{gran}"))


def overlap_figure(rows: list[dict], model: str) -> None:
    """Plot how far QAP moves the mask from WANDA at each bit width."""
    grids = [g for g in ("midrise", "midtread") if any(r["grid"] == g for r in rows)]
    bits = sorted({r["bits"] for r in rows})
    sparsity = rows[0]["sparsity"]
    fig, axes = S.plt.subplots(1, 2, figsize=(5.6, 2.2))
    width = 0.8 / len(grids)
    panels = (
        ("pruned_shared_with_wanda", f"pruned set shared with WANDA, s = {sparsity:g}"),
        ("qap_negative", r"weights with $w^2<\Delta^2/12$"),
    )
    for ax, (key, title) in zip(axes, panels):
        for k, grid in enumerate(grids):
            xs, ys = [], []
            for x, b in enumerate(bits):
                values = [r[key] for r in rows if (r["grid"], r["bits"]) == (grid, b)]
                if values:
                    xs.append(x + (k - (len(grids) - 1) / 2) * width)
                    ys.append(100 * np.mean(values))
            ax.bar(
                xs,
                ys,
                width=width * 0.92,
                color=S.GRID_COLOR[grid],
                label=S.GRID_LABEL[grid],
            )
        ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
        ax.set_ylabel("%")
        ax.grid(axis="x", visible=False)
        S.panel_label(ax, title)
    axes[0].set_ylim(0, 100)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, "qap-mask-vs-wanda"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    absmax = S.table(payloads, "absmax", ("layer", "gran", "criterion", "sparsity"))
    exchange = S.table(payloads, "exchange", ("layer", "gran", "grid", "bits"))
    overlap = S.table(payloads, "overlap", ("layer", "gran", "grid", "bits"))
    absmax_figure(absmax, args.model)
    for gran in sorted({r["gran"] for r in exchange}):
        exchange_figure(exchange, args.model, gran)
    if overlap:
        overlap_figure(overlap, args.model)


if __name__ == "__main__":
    main()
