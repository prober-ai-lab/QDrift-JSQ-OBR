# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the noise-amplification minitest.

``error-vs-sparsity``
    Geometric mean over layers, against sparsity, for OBR and QIR: (a) the
    deployment error E||V q - W x||^2, (b) its activation-noise part
    E||V n||^2, both over E||W x||^2, and (c) the noise gain
    E||V n||^2 / E||W n||^2 of the compressed weights over the dense ones.

    python visualize/plots/noise_amplification.py
"""

from __future__ import annotations

from collections import defaultdict

import style as S

STEM = "noise_amplification"
PANELS = (
    ("err", "deployment error / $E\\|Wx\\|^2$", True),
    ("err_a", "activation-noise error / $E\\|Wx\\|^2$", True),
    ("gain_a", "noise gain $E\\|Vn\\|^2 / E\\|Wn\\|^2$", False),
)


def error_figure(rows: list[dict], model: str) -> None:
    """Plot the three panels against sparsity."""
    fig, axes = S.plt.subplots(1, len(PANELS), figsize=(8.4, 2.4), squeeze=False)
    for ax, (metric, label, log) in zip(axes[0], PANELS):
        for mechanism in ("obr", "qir"):
            by_s = defaultdict(list)
            for r in rows:
                if r["mechanism"] == mechanism:
                    by_s[r["sparsity"]].append(r[metric])
            xs = sorted(by_s)
            ax.plot(
                xs,
                [S.geomean(by_s[s]) for s in xs],
                color=S.COMP_COLOR[mechanism],
                marker="o",
                label=S.COMP_LABEL[mechanism],
            )
        if log:
            ax.set_yscale("log")
        else:
            ax.axhline(1, color=S.AXIS, linewidth=0.9)
        ax.set_xlabel("sparsity $s$")
        S.panel_label(ax, label)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    S.save(fig, S.figure_name(STEM, model, "error-vs-sparsity"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    rows = S.table(payloads, "errors", ("layer", "sparsity", "mechanism"))
    error_figure(rows, args.model)


if __name__ == "__main__":
    main()
