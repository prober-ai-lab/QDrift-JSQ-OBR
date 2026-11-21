# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schematic of the pairwise 4:8 sparsity layout (paper Sec. 1 and 4.4).

``patterns`` (method figure, no data)
    Left: one weight row over 16 input channels under element-wise 2:4 and
    under pairwise 4:8, the only sparse layout of the INT4 sparse tensor-core
    instructions; brackets mark the groups each constraint acts on.  Right: the
    six legal pairwise patterns of one block of eight channels, two of the four
    aligned pairs retained.

    python visualize/plots/pairwise_layout.py
"""

from __future__ import annotations

from itertools import combinations

import style as S
from matplotlib.patches import Rectangle

KEPT = S.SERIES[0]
PRUNED = "white"
# Example rows: 1 = retained weight.
ELEMENTWISE_24 = (1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
PAIRWISE_48 = (1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0)


def draw_row(ax, y: float, row: tuple[int, ...], group: int, cell: float = 1.0) -> None:
    """One row of channel cells; a gap after every ``group`` channels."""
    for j, kept in enumerate(row):
        x = j * cell + (j // group) * 0.5
        ax.add_patch(Rectangle((x, y), 0.9 * cell, 0.9, facecolor=KEPT if kept else PRUNED,
                               edgecolor=S.INK_2, linewidth=0.6))


def patterns() -> None:
    fig, (left, right) = S.plt.subplots(
        1, 2, figsize=(6.9, 1.15), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    rows = (
        ("element-wise 2:4", ELEMENTWISE_24, 4),
        ("pairwise 4:8", PAIRWISE_48, 2),
    )
    for i, (label, row, group) in enumerate(rows):
        y = 2.2 * (len(rows) - 1 - i)
        draw_row(left, y, row, 8)
        left.text(-0.6, y + 0.45, label, ha="right", va="center", fontsize=7.5, color=S.INK)
        for k in range(0, 16, group):  # bracket the groups the constraint acts on
            x = k + (k // 8) * 0.5
            left.plot([x + 0.05, x + group - 0.15], [y - 0.3, y - 0.3],
                      color=S.MUTED, linewidth=0.8)
    left.set_xlim(-6.4, 16.8)
    left.set_ylim(-0.6, 3.3)
    S.panel_label(left, "(a) one weight row, 16 input channels")

    for p, pairs in enumerate(combinations(range(4), 2)):
        row = tuple(int(j // 2 in pairs) for j in range(8))
        x0 = (p % 2) * 9.5
        y0 = 1.25 * (2 - p // 2)
        for j, kept in enumerate(row):
            right.add_patch(Rectangle((x0 + j, y0), 0.9, 0.9,
                                      facecolor=KEPT if kept else PRUNED,
                                      edgecolor=S.INK_2, linewidth=0.6))
    right.set_xlim(-0.5, 18)
    right.set_ylim(-0.4, 3.5)
    S.panel_label(right, "(b) legal pairwise 4:8 patterns per block")
    for ax in (left, right):
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, facecolor=KEPT, edgecolor=S.INK_2, linewidth=0.6),
               Rectangle((0, 0), 1, 1, facecolor=PRUNED, edgecolor=S.INK_2, linewidth=0.6)]
    right.legend(handles, ["retained", "pruned"], loc="center left", ncol=1,
                 bbox_to_anchor=(1.0, 0.45), frameon=False, fontsize=7)
    S.save(fig, "method__pairwise_layout__all__patterns")


def main() -> None:
    S.setup()
    patterns()


if __name__ == "__main__":
    main()
