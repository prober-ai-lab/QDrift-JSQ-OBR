# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the drift-accumulation minitest.

``input-drift-vs-depth``
    Relative input drift ||dX|| / ||X|| at each block (attention input), one
    curve per compensation mode; the legend carries the end-to-end PPL.

``error-split-vs-depth``
    Per compensation mode, the inherited drift E||W dx||^2, the injected error
    E||dW x_hat||^2, and the total asymmetric error E||W_hat x_hat - W x||^2,
    each normalized by E||W x||^2 and averaged over the block's layers.

``cancellable-drift-vs-depth``
    Share of the inherited drift a free dense update of W could cancel, per
    layer type (OBR run), and the share actually removed by each mode,
    1 - total / (injected + inherited).

    python visualize/plots/drift_accumulation.py
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import style as S
from matplotlib.ticker import MaxNLocator

STEM = "drift_accumulation"
KEYS = ("comp", "layer")
# The run the paper reports (Fig. 2, Appendix E): W2A16 g128, 50% SparseGPT, GPTQ,
# 128x1024 calibration, evaluated on 16 WikiText-2 test sequences of 2048 tokens.
# Older runs with other recipes share the stem, and merging them would mix recipes,
# so this one is the default input.
PAPER_INPUT = S.RESULTS / (
    "drift_accumulation__llama2-7b__rot-hadamard_128x1024"
    "__w2_g128_mr_clip_gptq_u0.5_sparsegpt__none+obr+drift.json"
)


def per_block(
    rows: list[dict], comp: str, key: str, types: set[str] | None = None
) -> tuple[list, list]:
    """Block-wise mean of ``key`` for one compensation mode."""
    values = defaultdict(list)
    for r in rows:
        if r["comp"] == comp and (types is None or r["type"] in types):
            values[r["block"]].append(r[key])
    blocks = sorted(values)
    return blocks, [float(np.mean(values[b])) for b in blocks]


def comps_of(rows: list[dict]) -> list[str]:
    """Compensation modes present, in a fixed order."""
    return [c for c in ("none", "obr", "drift") if any(r["comp"] == c for r in rows)]


def drift_figure(rows: list[dict], nll: dict, model: str) -> None:
    """Plot the relative input drift against depth."""
    fig, ax = S.plt.subplots(figsize=(3.6, 2.3))
    for comp in comps_of(rows):
        blocks, drift = per_block(rows, comp, "drift", types={"q"})
        label = S.COMP_LABEL[comp]
        if nll.get(comp) is not None:
            label += f" (PPL {math.exp(nll[comp]):.2f})"
        ax.plot(
            blocks,
            drift,
            color=S.COMP_COLOR[comp],
            marker="o",
            markersize=2.5,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("block")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(r"$\|\Delta X\|/\|X\|$ at block input")
    if nll.get("dense") is not None:
        S.panel_label(ax, f"input drift (dense PPL {math.exp(nll['dense']):.2f})")
    ax.legend(loc="lower right")
    S.save(fig, S.figure_name(STEM, model, "input-drift-vs-depth"))


def split_figure(rows: list[dict], model: str) -> None:
    """Plot the inherited, injected, and total error against depth."""
    comps = comps_of(rows)
    parts = (
        ("inherited", r"inherited $\|W\Delta x\|^2$", S.SERIES[1]),
        ("injected", r"injected $\|\Delta W\hat x\|^2$", S.SERIES[0]),
        ("total", r"total $\|\hat W\hat x-Wx\|^2$", S.INK),
    )
    fig, axes = S.plt.subplots(
        1, len(comps), figsize=(2.5 * len(comps) + 0.8, 2.3), squeeze=False, sharey=True
    )
    for ax, comp in zip(axes[0], comps):
        for key, label, color in parts:
            blocks, values = per_block(rows, comp, key)
            style = "--" if key == "total" else "-"
            ax.plot(blocks, values, color=color, linestyle=style, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("block")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        S.panel_label(ax, S.COMP_LABEL[comp])
    axes[0][0].set_ylabel(r"error / $\|Wx\|^2$ (block mean)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    S.save(fig, S.figure_name(STEM, model, "error-split-vs-depth"))


def cancel_figure(rows: list[dict], model: str) -> None:
    """Plot the cancellable and the removed drift against depth."""
    comps = comps_of(rows)
    source = "obr" if "obr" in comps else comps[0]
    types = [t for t in S.TYPE_ORDER if any(r["type"] == t for r in rows)]
    palette = S.SERIES + ("#6250d6",)
    fig, axes = S.plt.subplots(1, 2, figsize=(6.2, 2.3))
    for color, layer_type in zip(palette, types):
        blocks, values = per_block(rows, source, "cancel", types={layer_type})
        axes[0].plot(blocks, [100 * v for v in values], color=color, label=layer_type)
    axes[0].set_ylabel("cancellable share of $\\|W\\Delta x\\|^2$ (%)")
    S.panel_label(
        axes[0], f"linearly cancellable drift ({S.COMP_LABEL[source]} prefix)"
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="layer")
    for comp in comps:
        removed = defaultdict(list)
        for r in rows:
            if r["comp"] == comp:
                removed[r["block"]].append(
                    1 - r["total"] / (r["injected"] + r["inherited"])
                )
        blocks = sorted(removed)
        axes[1].plot(
            blocks,
            [100 * np.mean(removed[b]) for b in blocks],
            color=S.COMP_COLOR[comp],
            label=S.COMP_LABEL[comp],
        )
    axes[1].axhline(0, color=S.AXIS, linewidth=0.8)
    axes[1].set_ylabel("error removed vs. inj. + inh. (%)")
    S.panel_label(axes[1], r"$1-\mathrm{total}/(\mathrm{inj.}+\mathrm{inh.})$")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes:
        ax.set_xlabel("block")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, "cancellable-drift-vs-depth"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    if not args.inputs and args.model == "llama2-7b":
        args.inputs = [str(PAPER_INPUT)]
    payloads = S.load(STEM, args)
    rows = S.table(payloads, "rows", KEYS)
    nll: dict = {}
    for payload in payloads:
        nll.update({k: v for k, v in payload.get("nll", {}).items() if v is not None})
    drift_figure(rows, nll, args.model)
    split_figure(rows, args.model)
    cancel_figure(rows, args.model)


if __name__ == "__main__":
    main()
