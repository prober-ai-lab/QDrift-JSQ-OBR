# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the QDrift mixture grid, read from the main campaign results.

Unlike the minitest figures, this script reads ``results/v2/`` run records
(one per (alpha, gamma) point) rather than ``results/minitests/``.

``alpha-curve``
    WikiText-2 perplexity against the asymmetric weight alpha at gamma = 0,
    test and validation, for one model and compression setting.  This is the
    single-minimum curve of the alpha-sweep ablation (Table 4) as a figure.

``alpha-gamma-heatmap``
    WikiText-2 (validation) perplexity over the two-dimensional mixture grid
    alpha in rows, gamma in columns, with OBR at (0, 0).  Shows which target
    dominates under a given sparsity and bit-width (Sec. 6, D1).

    python visualize/plots/mixture_grid.py --setting w4a4_s70 --model llama2-7b
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import style as S

RESULTS_V2 = S.ROOT / "results" / "v2"


def _mix_of(name: str) -> tuple[float, float] | None:
    """Parse (alpha, gamma) from a run file name; None if not a grid point.

    ``..._seq_seed42``       -> OBR baseline (0, 0)
    ``..._qdrift-a0.25c0.5`` -> the explicit mixture point (0.25, 0.5)
    ``..._qdrift_seed42``    -> the default mixture, which is pure asymmetric (1, 0)
    """
    if "_seq_seed42" in name:
        return 0.0, 0.0
    match = re.search(r"_qdrift-a([0-9.]+)c([0-9.]+)_seed42", name)
    if match:
        return float(match.group(1)), float(match.group(2))
    if "_qdrift_seed42" in name:
        return 1.0, 0.0
    return None


def load_points(model: str, setting: str, calib: str = "c128x1024") -> list[dict]:
    """Every (alpha, gamma) point for one model and setting.

    Returns dicts with ``alpha``, ``gamma``, ``test`` and ``val`` perplexity.
    """
    pattern = f"{model}_*_{setting}_{calib}_*seed42.json"
    points: dict[tuple[float, float], dict] = {}
    for path in sorted(RESULTS_V2.glob(pattern)):
        mix = _mix_of(path.name)
        if mix is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        ppl = payload.get("ppl", {})
        test = ppl.get("wikitext2")
        val = ppl.get("wikitext2_val")
        if test is None or val is None:
            continue
        alpha, gamma = mix
        points[(alpha, gamma)] = {
            "alpha": alpha,
            "gamma": gamma,
            "test": float(test),
            "val": float(val),
        }
        print(f"read {path.name}: (a={alpha}, g={gamma}) test={test:.2f} val={val:.2f}")
    return list(points.values())


def alpha_curve(points: list[dict], model: str, setting: str) -> None:
    """Perplexity against alpha at gamma = 0."""
    line = sorted((p for p in points if p["gamma"] == 0.0), key=lambda p: p["alpha"])
    if len(line) < 3:
        print(f"alpha-curve: only {len(line)} gamma=0 points, skipping")
        return
    alphas = [p["alpha"] for p in line]
    fig, ax = S.plt.subplots(figsize=(3.6, 2.4))
    for key, color, label in (
        ("test", S.SERIES[1], "test"),
        ("val", S.SERIES[0], "validation"),
    ):
        ax.plot(alphas, [p[key] for p in line], color=color, marker="o", label=label)
    best = min(line, key=lambda p: p["val"])
    ax.axvline(best["alpha"], color=S.MUTED, linewidth=0.8, linestyle=":")
    ax.scatter([best["alpha"]], [best["test"]], color=S.INK, s=22, zorder=5)
    ax.annotate(
        f"selected $\\alpha={best['alpha']:g}$\n{best['test']:.2f}",
        (best["alpha"], best["test"]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=7,
        color=S.INK,
    )
    ax.set_xlabel(r"asymmetric weight $\alpha$ ($\gamma=0$)")
    ax.set_ylabel("WikiText-2 perplexity")
    S.panel_label(ax, f"{_pretty(setting)} ($\\alpha$-sweep, OBR at $\\alpha=0$)")
    ax.legend(loc="upper center")
    S.save(fig, _name(model, "alpha-curve", setting))


def alpha_gamma_heatmap(points: list[dict], model: str, setting: str) -> None:
    """Validation perplexity over the common 3x3 (alpha, gamma) grid."""
    axis = [0.0, 0.25, 0.5]
    grid = np.full((len(axis), len(axis)), np.nan)
    for p in points:
        if p["alpha"] in axis and p["gamma"] in axis:
            grid[axis.index(p["alpha"]), axis.index(p["gamma"])] = p["val"]
    if np.isnan(grid).all():
        print("heatmap: no points on the 3x3 grid, skipping")
        return
    fig, ax = S.plt.subplots(figsize=(3.3, 2.8))
    finite = grid[np.isfinite(grid)]
    im = ax.imshow(
        grid, cmap=S.DIVERGING.reversed(), aspect="auto",
        vmin=float(finite.min()), vmax=float(np.percentile(finite, 90)),
    )
    for i in range(len(axis)):
        for j in range(len(axis)):
            if np.isfinite(grid[i, j]):
                best = grid[i, j] == np.nanmin(grid)
                ax.text(
                    j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    fontsize=7.5, color=S.INK, fontweight="bold" if best else "normal",
                )
                if i == 0 and j == 0:
                    ax.text(j, i + 0.33, "OBR", ha="center", va="center",
                            fontsize=6, color=S.INK_2)
    ax.set_xticks(range(len(axis)), [f"{g:g}" for g in axis])
    ax.set_yticks(range(len(axis)), [f"{a:g}" for a in axis])
    ax.set_xlabel(r"clean weight $\gamma$")
    ax.set_ylabel(r"asymmetric weight $\alpha$")
    ax.grid(False)
    S.panel_label(ax, f"{_pretty(setting)} validation PPL")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="PPL (lower is better)")
    S.save(fig, _name(model, "alpha-gamma-heatmap", setting))


def ppl_vs_sparsity(model: str, quant: str) -> None:
    """Test perplexity against sparsity for one bit-width.

    Three curves: OBR ``(0, 0)``, the validation-selected mixture (the
    minimum-validation point of the gamma = 0 line at each sparsity), and the
    pure asymmetric target ``(1, 0)``.  The pure target crossing above OBR at
    high sparsity is the overfitting story (Sec. 5.2).
    """
    pattern = re.compile(rf"_{quant}_s(\d+)_c128x1024_")
    sparsities = sorted(
        {
            int(match.group(1))
            for path in RESULTS_V2.glob(f"{model}_*_{quant}_s*_c128x1024_*seed42.json")
            if (match := pattern.search(path.name))
        }
    )
    rows = []
    for s in sparsities:
        points = load_points(model, f"{quant}_s{s}")
        by_mix = {(p["alpha"], p["gamma"]): p for p in points}
        obr = by_mix.get((0.0, 0.0))
        line = [p for p in points if p["gamma"] == 0.0]
        selected = min(line, key=lambda p: p["val"]) if line else None
        if obr is None or selected is None:
            continue
        pure = by_mix.get((1.0, 0.0))
        rows.append(
            {
                "s": s / 100,
                "obr": obr["test"],
                "selected": selected["test"],
                "pure": pure["test"] if pure else None,
                "alpha": selected["alpha"],
            }
        )
    if len(rows) < 2:
        print(f"ppl-vs-sparsity: only {len(rows)} sparsities for {quant}, skipping")
        return
    xs = [r["s"] for r in rows]
    fig, ax = S.plt.subplots(figsize=(3.6, 2.4))
    ax.plot(xs, [r["obr"] for r in rows], color=S.COMP_COLOR["obr"],
            marker="s", label="OBR")
    if all(r["pure"] is not None for r in rows):
        ax.plot(xs, [r["pure"] for r in rows], color=S.MUTED, marker="^",
                linestyle="--", label=r"pure asymmetric $(1,0)$")
    ax.plot(xs, [r["selected"] for r in rows], color=S.COMP_COLOR["drift"],
            marker="o", label="QDrift (selected)")
    for r in rows:
        ax.annotate(f"$\\alpha{{=}}{r['alpha']:g}$", (r["s"], r["selected"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=6,
                    color=S.COMP_COLOR["drift"])
    ax.set_xlabel("sparsity $s$")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_xticks(xs)
    S.panel_label(ax, f"{_pretty(quant)}, perplexity vs. sparsity")
    ax.legend(loc="upper left")
    S.save(fig, _name(model, "ppl-vs-sparsity", quant))


def _pretty(setting: str) -> str:
    """``w4a4_s70`` -> ``W4A4KV4, 70% sparsity``."""
    quant, _, sp = setting.partition("_s")
    bits = {"w4a4": "W4A4KV4", "w4a16": "W4A16", "w2a16": "W2A16"}.get(quant, quant)
    return f"{bits}, {int(sp)}% sparsity" if sp.isdigit() else bits


def _name(model: str, content: str, setting: str) -> str:
    return f"results__mixture_grid__{model}__{content}-{setting.replace('_', '-')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument("--setting", default="w4a4_s70")
    parser.add_argument(
        "--sweep",
        default=None,
        help="bit-width tag (e.g. w4a4) -> PPL-vs-sparsity figure, then exit",
    )
    args = parser.parse_args()
    S.setup()
    if args.sweep:
        ppl_vs_sparsity(args.model, args.sweep)
        return
    points = load_points(args.model, args.setting)
    if not points:
        raise SystemExit(f"no results/v2 points for {args.model} / {args.setting}")
    alpha_curve(points, args.model, args.setting)
    alpha_gamma_heatmap(points, args.model, args.setting)


if __name__ == "__main__":
    main()
