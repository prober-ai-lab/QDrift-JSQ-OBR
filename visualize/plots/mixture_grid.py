# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures for the QDrift mixture grid, read from the main campaign results.

Unlike the minitest figures, this script reads ``results/v2/`` run records
(one per (alpha, gamma) point) rather than ``results/minitests/``.

``alpha-curve``
    WikiText-2 perplexity against the asymmetric weight alpha at gamma = 0,
    test and validation, for one model and compression setting.  This is the
    single-minimum curve of the alpha-sweep ablation (Table 4) as a figure.

``delta-vs-obr`` (all models)
    Test perplexity of the selected QDrift and of the pure asymmetric target
    (1, 0), relative to OBR, for every row of the main table (Table 3).

``localization`` (LLaMA-2-7B, W4A4KV4 70%)
    Test-perplexity change against OBR when the pure asymmetric target is
    applied to a subset of the layers only (Table 2 as a figure).

    python visualize/plots/mixture_grid.py --setting w4a4_s70 --model llama2-7b
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import style as S
from matplotlib.ticker import FormatStrFormatter, NullFormatter

sys.path.insert(0, str(S.ROOT / "visualize" / "tables"))
import paper_tables as P  # noqa: E402
import tablekit as T  # noqa: E402

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


PROFILE_MODELS = ("llama2-7b", "llama3.1-8b")
PROFILE_SETTINGS = (
    "w4a16_s50", "w4a16_s70", "w4a4_s50", "w4a4_s60", "w4a4_s70", "w2a16_s50",
)
PROFILE_COLOR = {"50": S.SERIES[0], "60": S.SERIES[2], "70": S.SERIES[1]}
PROFILE_STYLE = {"w4a4": ("-", "o"), "w4a16": ("--", "s"), "w2a16": (":", "^")}


def alpha_profiles() -> None:
    """Validation perplexity relative to OBR against alpha at gamma = 0.

    One panel per model, one curve per setting: colour is the sparsity, line
    style the bit-width.  The filled marker is each curve's minimum.  Shows how
    the best alpha and the range of alpha that beats OBR move with the setting
    (Sec. 6).
    """
    fig, axes = S.plt.subplots(
        1, len(PROFILE_MODELS), figsize=(6.9, 2.5), sharey=True, squeeze=False
    )
    handles = {}
    for ax, model in zip(axes[0], PROFILE_MODELS):
        for setting in PROFILE_SETTINGS:
            line = sorted(
                (p for p in load_points(model, setting) if p["gamma"] == 0.0),
                key=lambda p: p["alpha"],
            )
            obr = next((p for p in line if p["alpha"] == 0.0), None)
            if obr is None or len(line) < 3:
                continue
            quant, _, sparsity = setting.partition("_s")
            style, marker = PROFILE_STYLE[quant]
            color = PROFILE_COLOR[sparsity]
            alphas = [p["alpha"] for p in line]
            ratios = [p["val"] / obr["val"] for p in line]
            (handle,) = ax.plot(
                alphas, ratios, color=color, linestyle=style, marker=marker,
                markersize=3, markerfacecolor="white",
            )
            best = min(range(len(line)), key=lambda i: ratios[i])
            ax.plot(alphas[best], ratios[best], color=color, marker=marker,
                    markersize=4.5, linestyle="none")
            handles.setdefault(_pretty(setting), handle)
        ax.axhline(1.0, color=S.INK_2, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"asymmetric weight $\alpha$ ($\gamma=0$)")
        S.panel_label(ax, MODEL_TITLE[model])
    axes[0][0].set_ylabel("validation PPL / OBR")
    fig.legend(
        handles.values(), handles.keys(), loc="upper center",
        bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
    )
    S.save(fig, "results__mixture_grid__all__alpha-profiles")


MODEL_TITLE = {"llama2-7b": "LLaMA-2-7B", "llama3.1-8b": "LLaMA-3.1-8B"}
WORSE = "#e34948"


def _sparsity_tick(pattern: str) -> str:
    """``s70`` -> ``70%``; 4:8 patterns -> ``4:8p`` (every layer) / ``4:8m`` (mixed)."""
    if pattern.startswith("4to8"):
        return P._nm_label(pattern, "4:8p", "4:8m")
    return f"{pattern[1:]}%"


def delta_vs_obr() -> None:
    """Selected QDrift and the pure target (1, 0) against OBR, every main-table row.

    One panel per model on a shared log axis of test PPL / OBR; OBR is the line
    at 1.  The filled circle is the validation-selected QDrift, labelled with its
    test-perplexity change; the hollow triangle is the pure asymmetric target.
    """
    runs = T.load_runs()
    models = list(dict.fromkeys(m for m, _, _ in P.MAIN_SPEC))
    widths = [sum(m == model for m, _, _ in P.MAIN_SPEC) for model in models]
    fig, axes = S.plt.subplots(
        1, len(models), figsize=(6.9, 2.4), sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": widths},
    )
    for ax, model in zip(axes[0], models):
        ticks, groups = [], {}
        for x, (_, bits, pattern) in enumerate(
            spec for spec in P.MAIN_SPEC if spec[0] == model
        ):
            ticks.append(_sparsity_tick(pattern))
            groups.setdefault(bits, []).append(x)
            tags = runs.get((model, bits, pattern, P.CALIB))
            obr, pure, sel = T.select(tags) if tags else (None, None, None)
            if obr is None:
                continue
            base = obr["ppl"]["wikitext2"]
            if pure is not None:
                ratio = pure["ppl"]["wikitext2"] / base
                ax.plot(x, ratio, marker="^", markersize=5, linestyle="none",
                        markerfacecolor="white", color=S.MUTED)
            ratio = sel["ppl"]["wikitext2"] / base
            ax.plot(x, ratio, marker="o", markersize=5, linestyle="none",
                    color=S.COMP_COLOR["drift"])
            ax.annotate(f"{100 * (ratio - 1):+.0f}%", (x, ratio),
                        textcoords="offset points", xytext=(0, -10), ha="center",
                        fontsize=6, color=S.INK_2)
        ax.axhline(1.0, color=S.COMP_COLOR["obr"], linewidth=1.0)
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_xlim(-0.6, len(ticks) - 0.4)
        ax.tick_params(axis="x", labelsize=6)
        for label, xs in groups.items():
            ax.annotate(label, ((xs[0] + xs[-1]) / 2, -0.17), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6, color=S.INK_2)
            ax.annotate("", (xs[0] - 0.3, -0.15), (xs[-1] + 0.3, -0.15),
                        xycoords=("data", "axes fraction"),
                        arrowprops={"arrowstyle": "-", "color": S.AXIS, "linewidth": 0.8})
        S.panel_label(ax, P.MODEL_DISP[model])
    ax = axes[0][0]
    ax.set_yscale("log")
    ax.set_ylim(0.25, 2.5)
    ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("test PPL / OBR")
    handles = [
        S.plt.Line2D([], [], color=S.COMP_COLOR["obr"], label="OBR"),
        S.plt.Line2D([], [], color=S.COMP_COLOR["drift"], marker="o", linestyle="none",
                     label="QDrift (selected on validation)"),
        S.plt.Line2D([], [], color=S.MUTED, marker="^", markerfacecolor="white",
                     linestyle="none", label=r"pure asymmetric $(1,0)$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False)
    S.save(fig, "results__mixture_grid__all__delta-vs-obr")


LOCALIZATION_ORDER = (
    "blocks 0--3", "blocks 0--7", "blocks 0--11", "blocks 0--15", "all blocks",
    "blocks 8--31", "attention only", "MLP gate/up-proj only", "MLP down-proj only",
)


def localization() -> None:
    """Change of test PPL against OBR with the pure target on a subset of layers."""
    tags = dict(P.LOCALIZATION_ROWS)
    records = {}
    for label in ("none (OBR)",) + LOCALIZATION_ORDER:
        path = T.RESULTS_V2 / f"{P.LOCALIZATION_BASE}_{tags[label]}_seed42.json"
        if path.exists():
            records[label] = json.loads(path.read_text(encoding="utf-8"))
    obr = records.get("none (OBR)")
    labels = [label for label in LOCALIZATION_ORDER if label in records]
    if obr is None or len(labels) < 3:
        print("localization: the localized runs are missing, skipping")
        return
    deltas = [T.delta_vs_obr(records[label], obr) for label in labels]
    fig, ax = S.plt.subplots(figsize=(3.4, 2.5))
    ys = list(range(len(labels)))[::-1]
    colors = [S.SERIES[0] if d < 0 else WORSE for d in deltas]
    ax.barh(ys, deltas, color=colors, height=0.62)
    for y, d in zip(ys, deltas):
        ax.annotate(f"{d:+.0f}%", (d, y), textcoords="offset points",
                    xytext=(3 if d >= 0 else -3, 0), ha="left" if d >= 0 else "right",
                    va="center", fontsize=6, color=S.INK_2)
    ax.axvline(0, color=S.INK_2, linewidth=0.8)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(-150, 1e5)
    ax.set_xticks([-100, -50, 0, 50, 100, 1000, 10000])
    ax.set_xticklabels(["$-$100", "$-$50", "0", "50", "100", "10$^3$", "10$^4$"])
    ax.set_yticks(ys, [label.replace("--", "–") for label in labels])
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test PPL change vs. OBR (%)")
    S.panel_label(ax, r"pure target $(1,0)$ on these layers, OBR elsewhere")
    S.save(fig, "results__mixture_grid__llama2-7b__localization-w4a4-s70")


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


if __name__ == "__main__":
    main()
