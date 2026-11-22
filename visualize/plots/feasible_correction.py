# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figure for the feasible-correction minitest (paper Appendix B).

``feasible-vs-depth``
    Left: the feasible fraction rho of each block, the share of the dense
    drift correction's energy ||delta_1||_H^2 that an update on the retained
    support keeps, sum over the block's layers.  Right: the held-out optimal
    step along the feasible correction, one step shared by the block's layers
    (the ratio of the summed numerators and denominators of Eq. heldstep);
    the calibration optimum is 1.  One curve per support; the dotted line is
    the alpha that validation selects (Table 3; one line per distinct value).

    python visualize/plots/feasible_correction.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import style as S
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(S.ROOT / "visualize" / "tables"))
import paper_tables as P  # noqa: E402
import tablekit as T  # noqa: E402

STEM = "feasible_correction"
# Support tag in the result name -> (label, colour, main-table pattern).
SUPPORTS = (
    ("u0.5_sparsegpt", "50% unstructured", S.SERIES[0], "s50"),
    ("u0.6_sparsegpt", "60% unstructured", S.SERIES[2], "s60"),
    ("u0.7_sparsegpt", "70% unstructured", S.SERIES[1], "s70"),
    ("4to8p-obs", "pairwise 4:8", S.SERIES[3], "4to8p-obs"),
)


def per_block(rows: list[dict], num: str, den: str) -> tuple[list, list]:
    """Block-wise ratio of summed ``num`` to summed ``den``."""
    sums = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        sums[r["block"]][0] += r[num]
        sums[r["block"]][1] += r[den]
    blocks = sorted(sums)
    return blocks, [sums[b][0] / sums[b][1] for b in blocks]


def selected_alpha(model: str, pattern: str) -> float | None:
    """The validation-selected alpha of the W4A4KV4 main-table row."""
    tags = T.load_runs().get((model, "W4A4", pattern, P.CALIB))
    if not tags:
        return None
    _, _, sel = T.select(tags)
    return (T.mix_of(sel["calibration"]) or (0.0, 0.0))[0]


def feasible_figure(model: str) -> None:
    fig, axes = S.plt.subplots(1, 2, figsize=(6.6, 2.3))
    drawn, alphas = 0, set()
    for tag, label, color, pattern in SUPPORTS:
        paths = sorted(S.RESULTS.glob(f"{STEM}__{model}__*_{tag}__obr-prefix.json"))
        if not paths:
            continue
        rows = json.loads(paths[-1].read_text(encoding="utf-8"))["rows"]
        print(f"read {paths[-1].name}")
        axes[0].plot(*per_block(rows, "feasible", "dense"), color=color, label=label)
        blocks, steps = per_block(rows, "held_num", "held_den")
        axes[1].plot(blocks, steps, color=color, label=label)
        alpha = selected_alpha(model, pattern)
        if alpha is not None:
            alphas.add(alpha)
        drawn += 1
    if not drawn:
        raise SystemExit(f"no {STEM} results for {model} under {S.RESULTS}")
    axes[0].set_ylabel(r"$\rho$: kept share of $\|\delta_1\|_H^2$")
    axes[0].set_ylim(0, 1)
    S.panel_label(axes[0], "(a) feasible fraction")
    axes[1].axhline(1.0, color=S.INK_2, linewidth=0.8)
    # The validation-selected alpha (one line per distinct value; on LLaMA-2-7B every
    # support selects the same 0.25).
    for alpha in sorted(alphas):
        axes[1].axhline(alpha, color=S.INK_2, linestyle=":", linewidth=1.0)
        axes[1].annotate(rf"selected $\alpha={alpha:g}$", (0.02, alpha),
                         xycoords=("axes fraction", "data"), xytext=(0, 3),
                         textcoords="offset points", fontsize=6, color=S.INK_2)
    axes[1].set_ylabel(r"held-out optimal step $\alpha^h$")
    S.panel_label(axes[1], "(b) held-out step (calibration: 1)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes:
        ax.set_xlabel("block")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    S.save(fig, S.figure_name(STEM, model, "feasible-vs-depth"))


def main() -> None:
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    feasible_figure(args.model)


if __name__ == "__main__":
    main()
