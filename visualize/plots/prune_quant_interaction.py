# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the single-layer prune/quant interaction minitest.

``delta-nll-heatmap``
    Mean interaction Delta = dNLL_joint - dNLL_quant - dNLL_prune over the
    tested layers, per sparsity s and bit width b (mid-rise grid), one panel
    per compensation.  Blue cells: the joint error is smaller than the sum
    of the individual errors, i.e. pruning absorbs quantization error.  Each
    cell also shows the share of layers with Delta <= 0.

``delta-by-layer-type-w2``
    Delta at 2 bits for each layer type, against sparsity.

    python visualize/plots/prune_quant_interaction.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import style as S
from matplotlib.colors import SymLogNorm

STEM = "prune_quant_interaction"
KEYS = ("layer", "comp", "variant", "grid", "bits", "sparsity")
GRID = "midrise"


def interactions(rows: list[dict]) -> list[dict]:
    """One record per (layer, comp, grid, bits, s) with the interaction."""
    quant, prune = {}, {}
    for row in rows:
        if row["variant"] == "quant":
            quant[(row["layer"], row["comp"], row["grid"], row["bits"])] = row
        elif row["variant"] == "prune":
            prune[(row["layer"], row["comp"], row["sparsity"])] = row
    out = []
    for row in rows:
        if row["variant"] != "joint":
            continue
        q = quant.get((row["layer"], row["comp"], row["grid"], row["bits"]))
        p = prune.get((row["layer"], row["comp"], row["sparsity"]))
        if q is None or p is None:
            continue
        out.append(
            {
                **{key: row[key] for key in ("layer", "type", "block", "comp", "grid")},
                "bits": row["bits"],
                "sparsity": row["sparsity"],
                "joint": row["dnll"],
                "quant": q["dnll"],
                "prune": p["dnll"],
                "delta": row["dnll"] - q["dnll"] - p["dnll"],
                "delta_err": row["rel_err"] - q["rel_err"] - p["rel_err"],
            }
        )
    return out


def panels(records: list[dict]) -> list[str]:
    """Compensation panels present, in a fixed order."""
    present = {r["comp"] for r in records}
    return [comp for comp in ("none", "obr") if comp in present]


def heatmap(records: list[dict], model: str) -> None:
    """Plot the mean interaction over sparsity and bit width."""
    keys = panels(records)
    sparsities = sorted({r["sparsity"] for r in records})
    bits = sorted({r["bits"] for r in records})
    cells = defaultdict(list)
    for r in records:
        cells[(r["comp"], r["bits"], r["sparsity"])].append(r["delta"])
    means = [np.mean(v) for v in cells.values()]
    limit = max(abs(min(means)), abs(max(means)), 1e-6)
    norm = SymLogNorm(linthresh=1e-3, vmin=-limit, vmax=limit, base=10)
    fig, axes = S.plt.subplots(
        1, len(keys), figsize=(3.2 * len(keys) + 0.6, 2.4), squeeze=False, sharey=True
    )
    image = None
    for ax, comp in zip(axes[0], keys):
        grid_values = np.full((len(bits), len(sparsities)), np.nan)
        share = np.full_like(grid_values, np.nan)
        for i, b in enumerate(bits):
            for j, s in enumerate(sparsities):
                values = cells.get((comp, b, s))
                if values:
                    grid_values[i, j] = np.mean(values)
                    share[i, j] = np.mean(np.asarray(values) <= 0)
        image = ax.imshow(
            grid_values, cmap=S.DIVERGING, norm=norm, aspect="auto", origin="lower"
        )
        for i in range(len(bits)):
            for j in range(len(sparsities)):
                if np.isnan(grid_values[i, j]):
                    ax.text(
                        j, i, "n/a", ha="center", va="center", color=S.MUTED, fontsize=6
                    )
                    continue
                dark = abs(norm(grid_values[i, j]) - 0.5) > 0.32
                ax.text(
                    j,
                    i,
                    f"{share[i, j] * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if dark else S.INK_2,
                )
        ax.set_xticks(range(len(sparsities)), [f"{s:g}" for s in sparsities])
        ax.set_yticks(range(len(bits)), [str(b) for b in bits])
        ax.set_xlabel("sparsity $s$")
        ax.grid(False)
        S.panel_label(ax, S.COMP_LABEL[comp])
    axes[0][0].set_ylabel("weight bits $b$")
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    bar.set_label(r"mean $\Delta$ (nats)")
    bar.outline.set_visible(False)
    fig.suptitle(
        r"$\Delta=\Delta\mathrm{NLL}_{joint}-\Delta\mathrm{NLL}_{quant}-\Delta\mathrm{NLL}_{prune}$"
        "  (cell: % of layers with $\\Delta\\leq 0$)",
        fontsize=8,
        x=0.02,
        ha="left",
        y=1.04,
    )
    S.save(fig, S.figure_name(STEM, model, "delta-nll-heatmap"))


def by_layer_type(records: list[dict], model: str, bits: int = 2) -> None:
    """Plot the interaction per layer type at one bit width."""
    keys = panels(records)
    types = [t for t in S.TYPE_ORDER if any(r["type"] == t for r in records)]
    fig, axes = S.plt.subplots(
        1, len(keys), figsize=(2.6 * len(keys), 2.2), squeeze=False, sharey=True
    )
    palette = S.SERIES + ("#6250d6",)
    for ax, comp in zip(axes[0], keys):
        for color, layer_type in zip(palette, types):
            by_s = defaultdict(list)
            for r in records:
                if (r["comp"], r["bits"], r["type"]) == (comp, bits, layer_type):
                    by_s[r["sparsity"]].append(r["delta"])
            if by_s:
                xs = sorted(by_s)
                ax.plot(
                    xs,
                    [np.mean(by_s[s]) for s in xs],
                    color=color,
                    marker="o",
                    label=layer_type,
                )
        ax.axhline(0, color=S.AXIS, linewidth=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("sparsity $s$")
        S.panel_label(ax, f"W{bits}, {S.COMP_LABEL[comp]}")
    axes[0][0].set_ylabel(r"mean $\Delta$ (nats)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="layer")
    S.save(fig, S.figure_name(STEM, model, f"delta-by-layer-type-w{bits}"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    rows = S.table(S.load(STEM, args), "rows", KEYS)
    records = [r for r in interactions(rows) if r["grid"] == GRID]
    if not records:
        raise SystemExit("no joint rows with matching quant/prune rows")
    heatmap(records, args.model)
    by_layer_type(records, args.model, 2)


if __name__ == "__main__":
    main()
