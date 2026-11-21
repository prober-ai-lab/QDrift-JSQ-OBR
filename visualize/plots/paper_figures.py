# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate every figure the paper uses, into ``figs/`` and ``figs/pdf/``.

A thin driver over the figure modules in this directory so the whole figure set
can be rebuilt with one command (the table counterpart is
``visualize/tables/paper_tables.py``).  Each figure is written twice,
``figs/png/<name>.png`` and ``figs/pdf/<name>.pdf``, by ``style.save``.

    python visualize/plots/paper_figures.py           # build all
    python visualize/plots/paper_figures.py --only drift

Figure -> paper:
    regime (Fig. 1) · drift (Figs. 2-4) · pipeline (Fig. 5) · alpha_curve (Fig. 6)
    · ppl_vs_sparsity (Fig. 7) · alpha_profiles (Sec. 5.3) · delta_vs_obr (Sec. 5.2)
    · localization (Sec. 3.3) · pairwise (Sec. 4.4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style as S  # noqa: E402
import drift_accumulation as drift  # noqa: E402
import mixture_grid as grid  # noqa: E402
import pairwise_layout as pairwise  # noqa: E402
import qdrift_pipeline as pipeline  # noqa: E402
import regime_tradeoff as regime  # noqa: E402


def build_drift() -> None:
    """Figures 2-4: the drift-accumulation figures (LLaMA-2-7B)."""
    argv = sys.argv
    sys.argv = ["paper_figures", "--model", "llama2-7b"]
    try:
        drift.main()
    except SystemExit as err:  # no drift minitest JSON yet
        print(f"skip drift figure: {err}")
    finally:
        sys.argv = argv


def build_alpha_curve() -> None:
    """Figure 6: the alpha-sweep U-curve at W4A4KV4 70% (LLaMA-2-7B)."""
    points = grid.load_points("llama2-7b", "w4a4_s70")
    if points:
        grid.alpha_curve(points, "llama2-7b", "w4a4_s70")


def build_ppl_vs_sparsity() -> None:
    """Figure 7: perplexity against sparsity for W4A4 (LLaMA-2-7B)."""
    grid.ppl_vs_sparsity("llama2-7b", "w4a4")


BUILDERS = {
    "regime": regime.main,
    "drift": build_drift,
    "pipeline": pipeline.main,
    "pairwise": pairwise.main,
    "alpha_curve": build_alpha_curve,
    "ppl_vs_sparsity": build_ppl_vs_sparsity,
    "alpha_profiles": grid.alpha_profiles,
    "delta_vs_obr": grid.delta_vs_obr,
    "localization": grid.localization,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", choices=list(BUILDERS), help="build a subset")
    args = parser.parse_args()
    S.setup()
    for name in args.only or list(BUILDERS):
        print(f"== {name} ==")
        BUILDERS[name]()


if __name__ == "__main__":
    main()
