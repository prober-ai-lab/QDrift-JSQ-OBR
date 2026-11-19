# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared style, loading, and saving for the figure scripts.

Every figure is written twice under one name: ``figs/<name>.png`` and
``figs/pdf/<name>.pdf``.  Names are flat and descriptive,
``<origin>__<producer>__<model>__<content>``, e.g.
``minitest__drift_accumulation__llama2-7b__input-drift-vs-depth``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "minitests"
FIGS = ROOT / "figs"

# Reference palette, light surface.  Categorical slots in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
# Ordinal blue ramp (steps 250..700) for ordered series such as bit widths.
BLUE_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# Diverging: blue = negative (absorbed), gray midpoint, red = positive.
DIVERGING = LinearSegmentedColormap.from_list(
    "qdrift_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL, "#f2a3a2", "#e34948", "#8f1f1e"],
)

COMP_LABEL = {"none": "no compensation", "obr": "OBR", "drift": "QDrift"}
COMP_COLOR = {"none": MUTED, "obr": SERIES[0], "drift": SERIES[1]}
GRID_LABEL = {"midrise": "mid-rise", "midtread": "mid-tread"}
GRID_COLOR = {"midrise": SERIES[0], "midtread": SERIES[1]}
TYPE_ORDER = ("q", "k", "v", "o", "gate", "up", "down")


def bit_color(bits: int, all_bits: Sequence[int]) -> str:
    """Darker for more bits; spread over the ordinal ramp."""
    ordered = sorted(all_bits)
    if len(ordered) == 1:
        return BLUE_RAMP[2]
    position = ordered.index(bits) / (len(ordered) - 1)
    return BLUE_RAMP[round(position * (len(BLUE_RAMP) - 1))]


def setup() -> None:
    """Apply the paper rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def arguments(description: str) -> argparse.Namespace:
    """Parse the arguments shared by every figure script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="explicit result JSON files (default: all for the model)",
    )
    parser.add_argument(
        "--with-smoke", action="store_true", help="also read *__smoke.json results"
    )
    return parser.parse_args()


def load(stem: str, args: argparse.Namespace) -> list[dict]:
    """All result payloads of one minitest for ``args.model``."""
    if args.inputs:
        paths = [Path(path) for path in args.inputs]
    else:
        paths = sorted(RESULTS.glob(f"{stem}__{args.model}__*.json"))
        if not args.with_smoke:
            paths = [path for path in paths if not path.stem.endswith("__smoke")]
    if not paths:
        raise SystemExit(f"no results for {stem} / {args.model} under {RESULTS}")
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        payloads.append(payload)
        print(f"read {path.name}")
    return payloads


def table(payloads: Iterable[dict], name: str, keys: Sequence[str]) -> list[dict]:
    """Concatenate one table; later files win on duplicate ``keys``."""
    merged: dict[tuple, dict] = {}
    for payload in payloads:
        for row in payload.get(name, []):
            merged[tuple(row.get(key) for key in keys)] = row
    return list(merged.values())


def figure_name(stem: str, model: str, content: str) -> str:
    """Flat figure name: ``minitest__<stem>__<model>__<content>``."""
    return f"minitest__{stem}__{model}__{content}"


def save(fig: plt.Figure, name: str) -> None:
    """Write ``figs/<name>.png`` and ``figs/pdf/<name>.pdf``."""
    (FIGS / "pdf").mkdir(parents=True, exist_ok=True)
    png = FIGS / f"{name}.png"
    pdf = FIGS / "pdf" / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


def geomean(values: Iterable[float]) -> float:
    """Geometric mean of the positive values."""
    array = np.asarray([v for v in values if v is not None and v > 0], dtype=float)
    return float(np.exp(np.log(array).mean())) if array.size else float("nan")


def panel_label(ax: plt.Axes, text: str) -> None:
    """Left-aligned panel title."""
    ax.set_title(text, loc="left")
