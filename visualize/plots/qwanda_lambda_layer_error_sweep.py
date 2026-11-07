# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the Q-WANDA lambda sweep minitest.

``error-vs-lambda``
    Joint layer-output error relative to WANDA (lambda = 1), geometric mean
    over layers, one curve per bit width; panels per compensation and
    sparsity.  The lambda with the smallest error is marked.

``best-lambda``
    Per-layer best lambda for each bit width, as a dot strip with the median.

    python visualize/plots/qwanda_lambda_layer_error_sweep.py
"""

from __future__ import annotations

from collections import defaultdict

import _style as S
import numpy as np

STEM = "qwanda_lambda_layer_error_sweep"
KEYS = ("layer", "grid", "bits", "sparsity", "lambda", "comp")


def relative(rows: list[dict]) -> dict:
    """{(comp, grid, s, bits): {layer: {lambda: err / err(lambda=1)}}}."""
    base = {
        (r["layer"], r["comp"], r["grid"], r["sparsity"], r["bits"]): r["rel_err"]
        for r in rows
        if r["lambda"] == 1.0
    }
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        ref = base.get((r["layer"], r["comp"], r["grid"], r["sparsity"], r["bits"]))
        if ref:
            key = (r["comp"], r["grid"], r["sparsity"], r["bits"])
            out[key][r["layer"]][r["lambda"]] = r["rel_err"] / ref
    return out


def error_figure(rows: list[dict], model: str, grid: str) -> None:
    """Plot the layer error relative to WANDA against lambda."""
    rel = relative([r for r in rows if r["grid"] == grid])
    comps = [c for c in ("none", "obr") if any(k[0] == c for k in rel)]
    sparsities = sorted({k[2] for k in rel})
    bits = sorted({k[3] for k in rel})
    fig, axes = S.plt.subplots(
        len(comps),
        len(sparsities),
        figsize=(2.6 * len(sparsities) + 0.7, 2.1 * len(comps)),
        squeeze=False,
        sharex=True,
    )
    for i, comp in enumerate(comps):
        for j, sparsity in enumerate(sparsities):
            ax = axes[i][j]
            for b in bits:
                layers = rel.get((comp, grid, sparsity, b))
                if not layers:
                    continue
                lams = sorted({lam for values in layers.values() for lam in values})
                curve = [
                    S.geomean(values.get(lam) for values in layers.values())
                    for lam in lams
                ]
                color = S.bit_color(b, bits)
                ax.plot(
                    lams, curve, color=color, marker="o", markersize=3, label=f"W{b}"
                )
                best = int(np.nanargmin(curve))
                ax.plot(
                    lams[best],
                    curve[best],
                    marker="*",
                    markersize=9,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=5,
                )
            ax.axhline(1, color=S.AXIS, linewidth=0.8)
            ax.set_yscale("log")
            S.panel_label(ax, f"{S.COMP_LABEL[comp]}, s = {sparsity:g}")
            if i == len(comps) - 1:
                ax.set_xlabel(r"Q-WANDA $\lambda$  ($\lambda$=1: WANDA)")
        axes[i][0].set_ylabel("layer error / WANDA")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="star: best")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"error-vs-lambda-{grid}"))


def best_lambda_figure(rows: list[dict], model: str, grid: str) -> None:
    """Plot the best lambda of every layer."""
    rel = relative([r for r in rows if r["grid"] == grid])
    comps = [c for c in ("none", "obr") if any(k[0] == c for k in rel)]
    sparsities = sorted({k[2] for k in rel})
    bits = sorted({k[3] for k in rel})
    fig, axes = S.plt.subplots(
        1,
        len(comps) * len(sparsities),
        figsize=(2.0 * len(comps) * len(sparsities), 2.1),
        squeeze=False,
        sharey=True,
    )
    rng = np.random.default_rng(0)
    for ax, (comp, sparsity) in zip(
        axes[0], [(c, s) for c in comps for s in sparsities]
    ):
        for x, b in enumerate(bits):
            layers = rel.get((comp, grid, sparsity, b), {})
            best = [min(values, key=values.get) for values in layers.values() if values]
            if not best:
                continue
            jitter = rng.uniform(-0.18, 0.18, len(best))
            ax.scatter(
                x + jitter,
                best,
                s=9,
                color=S.bit_color(b, bits),
                alpha=0.75,
                linewidths=0,
            )
            ax.hlines(np.median(best), x - 0.3, x + 0.3, color=S.INK, linewidth=1.4)
        ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
        ax.set_ylim(-0.05, 1.05)
        S.panel_label(ax, f"{S.COMP_LABEL[comp]}, s = {sparsity:g}")
    axes[0][0].set_ylabel(r"best $\lambda$ per layer (bar: median)")
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"best-lambda-per-layer-{grid}"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    rows = S.table(S.load(STEM, args), "rows", KEYS)
    for grid in sorted({r["grid"] for r in rows}):
        error_figure(rows, args.model, grid)
        best_lambda_figure(rows, args.model, grid)


if __name__ == "__main__":
    main()
