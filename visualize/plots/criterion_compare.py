# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the pruning-criterion comparison minitest.

``error-vs-wanda``
    Joint layer-output error of every criterion relative to WANDA, geometric
    mean over layers, grouped by bit width; one panel per compensation and
    sparsity structure.  Below 1 the criterion beats WANDA.

    python visualize/plots/criterion_compare.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import style as S

STEM = "criterion_compare"
KEYS = ("layer", "grid", "bits", "structure", "criterion", "comp")
ORDER = ("magnitude", "sparsegpt", "qap")
LABEL = {
    "magnitude": "magnitude",
    "sparsegpt": r"SparseGPT $w^2/[H^{-1}]_{jj}$",
    "qap": r"QAP $(w^2-\Delta^2/12)/[H^{-1}]_{jj}$",
}
STRUCT_LABEL = {"u0.5": "unstructured 50%", "u0.7": "unstructured 70%", "2to4": "2:4"}
COLORS = S.SERIES


def ratios(rows: list[dict]) -> dict:
    """{(comp, structure, bits, criterion): [err / err_wanda per layer]}."""
    base = {
        (r["layer"], r["comp"], r["structure"], r["bits"]): r["rel_err"]
        for r in rows
        if r["criterion"] == "wanda"
    }
    out: dict = defaultdict(list)
    for r in rows:
        ref = base.get((r["layer"], r["comp"], r["structure"], r["bits"]))
        if ref and r["criterion"] != "wanda":
            key = (r["comp"], r["structure"], r["bits"], r["criterion"])
            out[key].append(r["rel_err"] / ref)
    return out


def error_figure(rows: list[dict], model: str, grid: str) -> None:
    """Plot every criterion's error relative to WANDA."""
    rel = ratios([r for r in rows if r["grid"] == grid])
    comps = [c for c in ("none", "obr") if any(k[0] == c for k in rel)]
    structures = [s for s in STRUCT_LABEL if any(k[1] == s for k in rel)]
    bits = sorted({k[2] for k in rel})
    criteria = [c for c in ORDER if any(k[3] == c for k in rel)]
    fig, axes = S.plt.subplots(
        len(comps),
        len(structures),
        figsize=(2.7 * len(structures) + 1.6, 2.1 * len(comps)),
        squeeze=False,
        sharey=True,
    )
    width = 0.8 / len(criteria)
    for i, comp in enumerate(comps):
        for j, structure in enumerate(structures):
            ax = axes[i][j]
            for k, criterion in enumerate(criteria):
                xs, ys = [], []
                for x, b in enumerate(bits):
                    values = rel.get((comp, structure, b, criterion))
                    if values:
                        xs.append(x + (k - (len(criteria) - 1) / 2) * width)
                        ys.append(S.geomean(values))
                ax.bar(
                    xs,
                    np.asarray(ys) - 1.0,
                    bottom=1.0,
                    width=width * 0.9,
                    color=COLORS[k],
                    label=LABEL[criterion],
                )
            ax.axhline(1, color=S.INK_2, linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xticks(range(len(bits)), [f"W{b}" for b in bits])
            ax.grid(axis="x", visible=False)
            S.panel_label(ax, f"{S.COMP_LABEL[comp]}, {STRUCT_LABEL[structure]}")
        axes[i][0].set_ylabel("layer error / WANDA")
    axes[0][-1].legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="< 1: better than WANDA"
    )
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, f"error-vs-wanda-{grid}"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    rows = S.table(S.load(STEM, args), "rows", KEYS)
    for grid in sorted({r["grid"] for r in rows}):
        error_figure(rows, args.model, grid)


if __name__ == "__main__":
    main()
