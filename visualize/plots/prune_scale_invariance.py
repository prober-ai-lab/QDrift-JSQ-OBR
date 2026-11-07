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

``qwanda-mask-shift-vs-lambda``
    Overlap of the Q-WANDA pruned set with WANDA's, and the diagonal joint-loss
    proxy relative to WANDA (lambda = 1), against lambda.

    python visualize/plots/prune_scale_invariance.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import style as S

STEM = "prune_scale_invariance"
GRAN_ORDER = ("g64", "g128", "ch")
GRAN_LABEL = {"g64": "group 64", "g128": "group 128", "ch": "per channel"}


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
        name = criterion.replace("qwanda", "Q-WANDA $\\lambda$=").replace(
            "wanda", "WANDA"
        )
        S.panel_label(ax, name.replace("magnitude", "magnitude"))
    axes[0][0].set_ylabel("groups with absmax pruned (%)")
    axes[0][-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        title="solid: mean, dotted: worst layer",
    )
    S.save(fig, S.figure_name(STEM, model, "absmax-pruned-vs-sparsity"))


def exchange_figure(rows: list[dict], model: str, gran: str = "g128") -> None:
    """Plot the better-pruned-than-quantized statistics."""
    rows = [r for r in rows if r["gran"] == gran]
    bits = sorted({r["bits"] for r in rows})
    grids = [g for g in ("midrise", "midtread") if any(r["grid"] == g for r in rows)]
    share_keys = sorted(
        (k for k in rows[0] if k.startswith("loss_share_kept_s")),
        key=lambda k: float(k.rsplit("s", 1)[1]),
    )
    fig, axes = S.plt.subplots(1, 1 + len(grids), figsize=(2.6 * (1 + len(grids)), 2.3))
    width = 0.8 / len(grids)
    ax = axes[0]
    for k, grid in enumerate(grids):
        means = []
        for b in bits:
            values = [
                r["better_pruned_all"]
                for r in rows
                if (r["grid"], r["bits"]) == (grid, b)
            ]
            means.append(100 * np.mean(values) if values else np.nan)
        xs = np.arange(len(bits)) + (k - (len(grids) - 1) / 2) * width
        ax.bar(
            xs,
            means,
            width=width * 0.92,
            color=S.GRID_COLOR[grid],
            label=S.GRID_LABEL[grid],
        )
        for x, m in zip(xs, means):
            if not np.isnan(m):
                ax.text(
                    x,
                    m,
                    f"{m:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=S.INK_2,
                )
    ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
    ax.set_ylabel(r"weights with $\epsilon^2>w^2$ (%)")
    ax.legend(loc="upper right")
    S.panel_label(ax, f"all weights, {gran}")
    for ax, grid in zip(axes[1:], grids):
        for color, key in zip(S.BLUE_RAMP[1:], share_keys):
            xs, ys = [], []
            for b in bits:
                values = [r[key] for r in rows if (r["grid"], r["bits"]) == (grid, b)]
                if values:
                    xs.append(b)
                    ys.append(100 * np.mean(values))
            ax.plot(
                xs, ys, color=color, marker="o", label=f"s = {key.rsplit('s', 1)[1]}"
            )
        ax.set_xticks(bits, [f"W{b}" for b in bits])
        ax.set_ylabel("share of retained quant. loss (%)")
        S.panel_label(ax, f"kept by WANDA, {S.GRID_LABEL[grid]}")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"better-pruned-than-quantized-{gran}"))


def overlap_figure(rows: list[dict], model: str, grid: str = "midrise") -> None:
    """Plot the Q-WANDA mask shift and loss proxy against lambda."""
    rows = [r for r in rows if r["grid"] == grid]
    if not rows:
        return
    bits = sorted({r["bits"] for r in rows})
    proxy = defaultdict(dict)
    for r in rows:
        proxy[(r["layer"], r["bits"])][r["lambda"]] = (
            r["proxy_prune"] + r["proxy_quant"]
        )
    fig, axes = S.plt.subplots(1, 2, figsize=(5.4, 2.2))
    for b in bits:
        color = S.bit_color(b, bits)
        by_lam = defaultdict(list)
        ratio = defaultdict(list)
        for r in rows:
            if r["bits"] != b:
                continue
            by_lam[r["lambda"]].append(r["pruned_shared_with_wanda"])
            base = proxy[(r["layer"], b)].get(1.0)
            if base:
                ratio[r["lambda"]].append(proxy[(r["layer"], b)][r["lambda"]] / base)
        xs = sorted(by_lam)
        axes[0].plot(
            xs,
            [100 * np.mean(by_lam[x]) for x in xs],
            color=color,
            marker="o",
            label=f"W{b}",
        )
        xs = sorted(ratio)
        axes[1].plot(xs, [S.geomean(ratio[x]) for x in xs], color=color, marker="o")
    sparsity = rows[0]["sparsity"]
    axes[0].set_ylabel("pruned set shared with WANDA (%)")
    axes[1].set_ylabel("diag. joint-loss proxy / WANDA")
    axes[1].axhline(1, color=S.AXIS, linewidth=0.8)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel(r"Q-WANDA $\lambda$")
    S.panel_label(axes[0], f"mask overlap, s = {sparsity:g}, {S.GRID_LABEL[grid]}")
    S.panel_label(axes[1], r"$\sum_P w^2x^2+\sum_R\epsilon^2x^2$ (geo. mean)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"qwanda-mask-shift-vs-lambda-{grid}"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    absmax = S.table(payloads, "absmax", ("layer", "gran", "criterion", "sparsity"))
    exchange = S.table(payloads, "exchange", ("layer", "gran", "grid", "bits"))
    overlap = S.table(payloads, "overlap", ("layer", "gran", "grid", "bits", "lambda"))
    absmax_figure(absmax, args.model)
    for gran in sorted({r["gran"] for r in exchange}):
        exchange_figure(exchange, args.model, gran)
    for grid in ("midrise", "midtread"):
        overlap_figure(overlap, args.model, grid)


if __name__ == "__main__":
    main()
