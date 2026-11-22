# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework figure (paper Fig. 2): where the drift comes from and how QDrift
compresses one linear layer.

(a) The full-precision model and the compressed model run on the same tokens.
    Blocks 1..l-1 of the compressed model are already compressed, so layer l
    receives the drifted input X_hat instead of the clean input X; the deployed
    model should still produce W X.
(b) QDrift for layer l: three reconstruction targets, weighted (alpha, beta,
    gamma), merge into one quadratic with the pair (H_mix, T); the mask and OBR's
    two stages with GPTQ then run unchanged on that pair, and the compressed layer
    becomes part of the prefix of layer l+1.  Orange marks what QDrift changes,
    blue what OBR already does.

Every label is typeset by LaTeX (Times, as the paper), so the figure is included
at its natural width, about the 6.75 in text width of the ICML template.

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
FP_FILL = S.NEUTRAL
DONE_FILL = "#d9d7d0"   # an already compressed block
DRIFT = "#c0392b"

TEX = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{amsmath}",
    "font.size": 8,
}


def rbox(ax, x0, y0, w, h, edge, fill, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=2))


def arrow(ax, start, end, color=S.INK_2, rad=0.0, dashed=False, lw=0.9, both=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", linestyle="--" if dashed else "-",
        shrinkA=0, shrinkB=0, zorder=3))


def text(ax, x, y, s, size=8, color=S.INK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, color=color, zorder=4, **kw)


def network(ax) -> None:
    """Panel (a): the two models side by side, block l highlighted."""
    text(ax, 0, 43.6, r"\textbf{(a)} The compressed prefix hands layer $\ell$ a drifted input",
         size=8.5, ha="left")
    y_fp, y_cq, h, w = 35.0, 25.6, 5.0, 8.0
    # (label, x centre, kind): kind decides the fill of the compressed row.
    blocks = (("1", 15, "done"), ("2", 25, "done"), ("dots", 32.8, "done"),
              (r"\ell-1", 40.6, "done"), (r"\ell", 55, "now"), ("dots", 63.8, "todo"),
              ("L", 71.2, "todo"))
    text(ax, 0.3, y_fp + h / 2, r"full precision", size=7.5, ha="left", color=S.INK_2)
    text(ax, 0.3, y_cq + h / 2, r"compressed", size=7.5, ha="left", color=S.INK_2)
    edges = []  # (left edge, right edge) of each element, for the arrows between them
    for label, xc, kind in blocks:
        bw = 0 if label == "dots" else w + (3 if kind == "now" else 0)
        half = 1.8 if label == "dots" else bw / 2
        edges.append((xc - half, xc + half))
        for row, y in (("fp", y_fp), ("cq", y_cq)):
            if label == "dots":
                text(ax, xc, y + h / 2, r"$\cdots$", size=9, color=S.INK_2)
                continue
            if row == "fp":
                edge, fill = S.MUTED, FP_FILL
            elif kind == "done":
                edge, fill = S.MUTED, DONE_FILL
            elif kind == "now":
                edge, fill = NEW, NEW_FILL
            else:
                edge, fill = S.AXIS, "white"
            rbox(ax, xc - bw / 2, y, bw, h, edge, fill,
                 lw=1.3 if kind == "now" else 0.9,
                 ls="--" if kind == "todo" and row == "cq" else "-")
            text(ax, xc, y + h / 2, rf"block ${label}$", size=7.5)
    for (_, right), (left, _) in zip(edges, edges[1:]):
        for y in (y_fp, y_cq):
            arrow(ax, (right + 0.3, y + h / 2), (left - 0.3, y + h / 2))
    arrow(ax, (edges[-1][1] + 0.3, y_fp + h / 2), (edges[-1][1] + 2.6, y_fp + h / 2))
    arrow(ax, (edges[-1][1] + 0.3, y_cq + h / 2), (edges[-1][1] + 2.6, y_cq + h / 2))
    # Inputs of block l on both rows, and the drift between them.
    x_in = (edges[3][1] + edges[4][0]) / 2
    text(ax, x_in, y_fp + h / 2 + 2.1, r"$X$", size=8)
    text(ax, x_in, y_cq + h / 2 - 2.2, r"$\hat X$", size=8)
    arrow(ax, (x_in, y_cq + h + 0.3), (x_in, y_fp - 0.3), color=DRIFT, both=True, lw=1.0)
    text(ax, x_in + 0.9, (y_cq + h + y_fp) / 2, r"$\Delta=\hat X-X$", size=7.5,
         color=DRIFT, ha="left")
    # Right column: what layer l should produce, and what OBR fits instead.
    xr = edges[-1][1] + 3.6
    text(ax, xr, y_fp + h / 2 + 1.2, r"wanted: $WX$", size=7.5, ha="left")
    text(ax, xr, y_fp + h / 2 - 1.8, r"clean input, original $W$", size=7, ha="left",
         color=S.INK_2)
    text(ax, xr, y_cq + h / 2 + 1.2, r"OBR fits $\hat W\hat X\approx W\hat X$", size=7.5,
         ha="left")
    text(ax, xr, y_cq + h / 2 - 1.8, r"$\Delta$ is never compensated", size=7,
         ha="left", color=DRIFT)
    text(ax, (edges[0][0] + edges[3][1]) / 2, y_cq - 2.4,
         r"already compressed: $\Delta$ grows with depth (Fig.~3)", size=7, color=S.INK_2)
    text(ax, (edges[5][0] + edges[6][1]) / 2, y_cq - 2.4, r"not yet compressed", size=7,
         color=S.INK_2)


def solver(ax) -> None:
    """Panel (b): the three targets, the merged problem and OBR's unchanged solver."""
    text(ax, 0, 17.6, r"\textbf{(b)} QDrift for layer $\ell$: one weighted target, OBR's solver",
         size=8.5, ha="left")
    y0, h = 0.0, 15.0
    # Three targets.
    rbox(ax, 0, y0, 30, h, NEW, NEW_FILL)
    text(ax, 15, y0 + h - 2.0, r"\textbf{three targets} (Eq.~5)", size=7.5)
    rows = ((r"$\alpha$", r"$\|\hat w\hat X-wX\|^2$", r"asymmetric"),
            (r"$\beta$", r"$\|\hat w\hat X-w\hat X\|^2$", r"symmetric (OBR)"),
            (r"$\gamma$", r"$\|\hat wX-wX\|^2$", r"clean input"))
    for i, (wt, expr, name) in enumerate(rows):
        y = y0 + h - 5.4 - 3.6 * i
        text(ax, 2.2, y, wt, size=8, ha="left", color=NEW)
        text(ax, 5.2, y, expr, size=8, ha="left")
        text(ax, 29.0, y, name, size=7, ha="right", color=S.INK_2)
    arrow(ax, (30.4, y0 + h / 2), (33.6, y0 + h / 2))
    # Merged problem.
    rbox(ax, 34, y0, 27, h, NEW, NEW_FILL)
    text(ax, 47.5, y0 + h - 2.0, r"\textbf{one quadratic} (Eq.~6)", size=7.5)
    text(ax, 47.5, y0 + 8.7, r"$\min\;(\hat w-T)\,H_{\mathrm{mix}}\,(\hat w-T)^{\top}$", size=8)
    text(ax, 47.5, y0 + 5.3, r"$H_{\mathrm{mix}}=(1-\gamma)H_{\hat X}+\gamma H_X$", size=8)
    text(ax, 47.5, y0 + 2.0, r"$T=w-\alpha\,wDH_{\mathrm{mix}}^{-1}$", size=8)
    arrow(ax, (61.4, y0 + h / 2), (64.6, y0 + h / 2))
    # Mask.
    rbox(ax, 65, y0, 13.5, h, NEW, NEW_FILL)
    text(ax, 71.75, y0 + h - 2.0, r"\textbf{mask on} $\boldsymbol{T}$", size=7.5)
    text(ax, 71.75, y0 + 6.6, "SparseGPT\nscore, or exact\npairwise 4:8\n(Alg.~1)",
         size=7, color=S.INK_2, linespacing=1.25)
    arrow(ax, (78.9, y0 + h / 2), (82.1, y0 + h / 2))
    # OBR stages.
    rbox(ax, 82.5, y0, 17.5, h, OBR, OBR_FILL)
    text(ax, 91.25, y0 + h - 2.0, r"\textbf{OBR, unchanged}", size=7.5)
    text(ax, 91.25, y0 + 6.6,
         "stage 1: compensate\npruning (Eq.~7)\nstage 2: round, re-solve;\nGPTQ on late half",
         size=7, color=S.INK_2, linespacing=1.25)


def main() -> None:
    S.setup()
    with plt.rc_context(TEX):
        fig, ax = plt.subplots(figsize=(6.6, 2.72))
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 45.0)
        ax.axis("off")
        fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
        network(ax)
        solver(ax)
        # The compressed layer becomes part of the prefix of layer l+1.
        # Elbow path under panel (a): up from OBR's box, left, up into block l.
        yk = 20.2
        ax.plot([91.25, 91.25, 55], [15.2, yk, yk], color=S.INK_2, linewidth=0.9,
                linestyle="--", zorder=3)
        arrow(ax, (55, yk), (55, 25.3), dashed=True, color=S.INK_2)
        text(ax, 57, yk - 1.7, r"$\hat W$ replaces layer $\ell$ and joins the prefix of $\ell+1$",
             size=7, color=S.INK_2, ha="left")
        # Legend.
        for i, (edge, fill, label) in enumerate(((NEW, NEW_FILL, "changed by QDrift"),
                                                 (OBR, OBR_FILL, "OBR, unchanged"))):
            x = 64 + i * 19
            rbox(ax, x, 42.7, 2.4, 1.8, edge, fill, lw=0.9)
            text(ax, x + 3.2, 43.6, label, size=7, ha="left", color=S.INK_2)
        S.save(fig, "method__qdrift_pipeline__all__overview")


if __name__ == "__main__":
    main()
