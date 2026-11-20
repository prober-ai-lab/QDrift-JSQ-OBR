# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method overview (paper Fig. 5): how QDrift compresses one linear layer.

Two forward passes give the clean input X and the drifted input X_hat; their
statistics merge into the mixture pair (H_mix, T); the mask, OBR's two stages and
GPTQ then run unchanged on that pair, and the compressed layer feeds the prefix of
the next one.  Orange marks what QDrift changes, blue what OBR already does.

    python visualize/plots/qdrift_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import style as S  # noqa: E402

NEW = S.SERIES[1]    # changed by QDrift
OBR = S.SERIES[0]    # OBR, unchanged
NEW_FILL = "#fdeee6"
OBR_FILL = "#e9f1fb"
IN_FILL = S.NEUTRAL


def box(ax, x0, y0, x1, y1, title, body, edge, fill):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2))
    xc = (x0 + x1) / 2
    ax.text(xc, y1 - 1.6, title, ha="center", va="top", fontsize=7.5,
            fontweight="bold", color=S.INK, zorder=3)
    ax.text(xc, (y0 + y1) / 2 - 1.6, body, ha="center", va="center", fontsize=7,
            color=S.INK_2, linespacing=1.5, zorder=3)


def arrow(ax, start, end, color=S.INK_2, style="-|>", rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9, linewidth=1.0, color=color,
        connectionstyle=f"arc3,rad={rad}", linestyle="--" if dashed else "-", zorder=1))


def main() -> None:
    S.setup()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 46)
    ax.axis("off")

    # Row 1: two forward passes -> statistics -> mixture.
    box(ax, 0, 32, 25, 42, "Full-precision prefix",
        r"clean input $X$", S.MUTED, IN_FILL)
    box(ax, 0, 18, 25, 28, "Compressed prefix",
        r"drifted input $\hat X$", S.MUTED, IN_FILL)
    box(ax, 31, 18, 56, 42, "Statistics  (Eq. 4)",
        r"$H_X=\mathrm{E}[XX^{\mathsf{T}}]$" "\n"
        r"$H_{\hat X}=\mathrm{E}[\hat X\hat X^{\mathsf{T}}]$" "\n"
        r"$D=\mathrm{E}[(\hat X-X)\hat X^{\mathsf{T}}]$", NEW, NEW_FILL)
    box(ax, 62, 18, 99, 42, r"Mixture $(\alpha,\gamma)$  (Eq. 5–6)",
        r"$H_{\mathrm{mix}}=(1-\gamma)H_{\hat X}+\gamma H_X$" "\n"
        r"$T=w-\alpha\,wDH_{\mathrm{mix}}^{-1}$" "\n"
        r"$(\alpha,\gamma)=(0,0)$ is OBR;" "\n"
        r"chosen on validation", NEW, NEW_FILL)
    arrow(ax, (25, 37), (31, 33))
    arrow(ax, (25, 23), (31, 27))
    arrow(ax, (56, 30), (62, 30))

    # Row 2 (right to left): mask -> stage 1 -> stage 2 + GPTQ -> compressed layer.
    box(ax, 75, -3, 99, 12, "Mask  (Sec. 4.4)",
        "SparseGPT score, or\nexact pairwise 4:8\non $T$ (Alg. 1)", NEW, NEW_FILL)
    box(ax, 50, -3, 70, 12, "OBR stage 1",
        "compensate pruning\non retained set\n(Eq. 7)", OBR, OBR_FILL)
    box(ax, 25, -3, 45, 12, "OBR stage 2",
        "round early columns,\nre-solve late ones;\nGPTQ on late half", OBR, OBR_FILL)
    box(ax, 0, -3, 22, 12, "Compressed layer",
        r"sparse, low-bit $\hat W$", S.MUTED, IN_FILL)
    arrow(ax, (87, 18), (87, 12))
    arrow(ax, (75, 4.5), (70, 4.5))
    arrow(ax, (50, 4.5), (45, 4.5))
    arrow(ax, (25, 4.5), (22, 4.5))
    arrow(ax, (10.5, 12), (10.5, 18), dashed=True)
    ax.text(11.8, 15, "next layer", fontsize=6.5, color=S.MUTED, va="center")

    # Legend.
    for i, (edge, fill, label) in enumerate(((NEW, NEW_FILL, "changed by QDrift"),
                                             (OBR, OBR_FILL, "OBR, unchanged"))):
        x = 31 + i * 24
        ax.add_patch(FancyBboxPatch((x, 43.2), 2.2, 1.6, boxstyle="round,pad=0,rounding_size=0.4",
                                    linewidth=0.9, edgecolor=edge, facecolor=fill))
        ax.text(x + 3, 44, label, fontsize=6.5, color=S.INK_2, va="center")

    S.save(fig, "method__qdrift_pipeline__all__overview")


if __name__ == "__main__":
    main()
