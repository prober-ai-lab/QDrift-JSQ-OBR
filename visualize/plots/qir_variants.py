# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the QIR-variants minitest.

``ratio-vs-depth``
    Held-out deployment error of each QIR variant over OBR's, geometric mean
    over the layer types of a block, against the block index; one panel per
    sparsity.  Below 1, the variant beats OBR.
``noise-vs-depth``
    Per block: activation noise E||W n||^2 / E||W x||^2 and the share of it
    carried by outlier tokens, geometric mean over layer types.

    python visualize/plots/qir_variants.py
"""

from __future__ import annotations

from collections import defaultdict

import style as S

STEM = "qir_variants"
VARIANTS = ("qir", "qir_half", "qir_diag", "qir_outlier", "qir_both")


def ratio_figure(rows: list[dict], model: str) -> None:
    """Plot the variant / OBR error ratio against depth."""
    sparsities = sorted({r["sparsity"] for r in rows})
    index = {(r["layer"], r["sparsity"], r["mechanism"]): r for r in rows}
    fig, axes = S.plt.subplots(
        1, len(sparsities), figsize=(3.4 * len(sparsities) + 0.8, 2.4),
        squeeze=False, sharey=True,
    )
    for ax, sparsity in zip(axes[0], sparsities):
        for color, variant in zip(S.SERIES, VARIANTS):
            by_block = defaultdict(list)
            for (layer, s, mech), r in index.items():
                if s != sparsity or mech != variant:
                    continue
                base = index[(layer, s, "obr")]["err"]
                by_block[r["block"]].append(r["err"] / base)
            xs = sorted(by_block)
            ax.plot(
                xs, [S.geomean(by_block[b]) for b in xs],
                color=color, marker="o", label=variant,
            )
        ax.axhline(1, color=S.AXIS, linewidth=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("block")
        S.panel_label(ax, f"$s={sparsity:g}$")
    axes[0][0].set_ylabel("held-out error / OBR")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    S.save(fig, S.figure_name(STEM, model, "ratio-vs-depth"))


def noise_figure(rows: list[dict], model: str) -> None:
    """Plot the activation noise and its outlier share against depth."""
    layers = {r["layer"]: r for r in rows}.values()
    fig, axes = S.plt.subplots(1, 2, figsize=(6.8, 2.4), squeeze=False)
    for ax, (key, label, log) in zip(
        axes[0],
        (
            ("noise_over_signal", "$E\\|Wn\\|^2 / E\\|Wx\\|^2$", True),
            ("outlier_noise", "outlier-token share of the noise", False),
        ),
    ):
        by_block = defaultdict(list)
        for r in layers:
            by_block[r["block"]].append(r[key])
        xs = sorted(by_block)
        ax.plot(xs, [S.geomean(by_block[b]) for b in xs], color=S.SERIES[0], marker="o")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("block")
        S.panel_label(ax, label)
    S.save(fig, S.figure_name(STEM, model, "noise-vs-depth"))


def main() -> None:
    """Render every figure of this minitest."""
    args = S.arguments(__doc__.split("\n\n")[0])
    S.setup()
    payloads = S.load(STEM, args)
    rows = S.table(payloads, "errors", ("layer", "sparsity", "mechanism"))
    ratio_figure(rows, args.model)
    noise_figure(rows, args.model)


if __name__ == "__main__":
    main()
