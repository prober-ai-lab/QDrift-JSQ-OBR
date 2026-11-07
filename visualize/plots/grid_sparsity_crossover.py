# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the mid-rise vs mid-tread crossover minitest.

``error-ratio-vs-sparsity``
    Mid-rise / mid-tread layer-output error, geometric mean over layers, per
    bit width against sparsity: retained-weight quantization error, joint
    error without compensation, and joint error with OBR.  Below 1, mid-rise
    is better; the first sparsity where a curve crosses 1 is marked.

``clip-ratio-distribution``
    Share of quantization groups that select each clip ratio rho*.

    python visualize/plots/grid_sparsity_crossover.py
"""

from __future__ import annotations

from collections import defaultdict

import style as S

STEM = "grid_sparsity_crossover"
METRICS = (
    ("quant_err", "retained weights only"),
    ("joint_err", "joint, no compensation"),
    ("joint_err_obr", "joint, OBR"),
)


def ratio_figure(rows: list[dict], model: str) -> None:
    """Plot the mid-rise / mid-tread error ratio against sparsity."""
    metrics = [(k, label) for k, label in METRICS if any(k in r for r in rows)]
    bits = sorted({r["bits"] for r in rows if r["grid"] == "midtread"})
    index = {(r["layer"], r["grid"], r["bits"], r["sparsity"]): r for r in rows}
    fig, axes = S.plt.subplots(
        1,
        len(metrics),
        figsize=(2.6 * len(metrics) + 0.6, 2.3),
        squeeze=False,
        sharey=True,
    )
    for ax, (metric, label) in zip(axes[0], metrics):
        for b in bits:
            by_s = defaultdict(list)
            for (layer, grid, bb, s), r in index.items():
                if grid != "midrise" or bb != b or metric not in r:
                    continue
                other = index.get((layer, "midtread", b, s))
                if other and other.get(metric):
                    by_s[s].append(r[metric] / other[metric])
            xs = sorted(by_s)
            ys = [S.geomean(by_s[s]) for s in xs]
            color = S.bit_color(b, bits)
            ax.plot(xs, ys, color=color, marker="o", label=f"W{b}")
            crossing = next((x for x, y in zip(xs, ys) if y < 1), None)
            if crossing is not None:
                ax.axvline(crossing, color=color, linewidth=0.8, linestyle=":")
        ax.axhline(1, color=S.AXIS, linewidth=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("sparsity $s$ (WANDA mask)")
        S.panel_label(ax, label)
    axes[0][0].set_ylabel("mid-rise / mid-tread error")
    axes[0][-1].legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="< 1: mid-rise better"
    )
    S.save(fig, S.figure_name(STEM, model, "error-ratio-vs-sparsity"))


def rho_figure(rows: list[dict], model: str) -> None:
    """Plot the share of groups selecting each clip ratio."""
    totals: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for rho, count in r["rho"].items():
            totals[(r["grid"], r["bits"])][float(rho)] += count
    keys = sorted(totals, key=lambda k: (k[0] != "midrise", k[1]))
    rhos = sorted({rho for counts in totals.values() for rho in counts})
    colors = dict(zip(rhos, S.BLUE_RAMP[-len(rhos) :] if len(rhos) <= 4 else S.SERIES))
    fig, ax = S.plt.subplots(figsize=(3.6, 2.2))
    for x, key in enumerate(keys):
        total = sum(totals[key].values())
        bottom = 0.0
        for rho in rhos:
            share = 100 * totals[key].get(rho, 0) / total
            ax.bar(
                x,
                share,
                bottom=bottom,
                color=colors[rho],
                width=0.7,
                edgecolor="white",
                linewidth=1.0,
                label=f"{rho:g}" if x == 0 else None,
            )
            bottom += share
    ax.set_xticks(
        range(len(keys)), [f"{'MR' if g == 'midrise' else 'MT'} W{b}" for g, b in keys]
    )
    ax.set_ylabel("groups (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title=r"clip $\rho^\star$")
    S.panel_label(ax, "selected clip ratio per group")
    S.save(fig, S.figure_name(STEM, model, "clip-ratio-distribution"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    errors = S.table(payloads, "errors", ("layer", "grid", "bits", "sparsity"))
    rho = S.table(payloads, "rho", ("layer", "grid", "bits"))
    ratio_figure(errors, args.model)
    rho_figure(rho, args.model)


if __name__ == "__main__":
    main()
