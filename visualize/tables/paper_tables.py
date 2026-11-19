# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate every paper table as a booktabs ``.tex`` fragment and a compiled PDF.

Each table is written to ``tables/<name>.tex`` (``\\input`` it in the paper) and
``tables/pdf/<name>.pdf`` (standalone, for preview or direct inclusion).  Data
tables are pulled live from ``results/v2`` so they track the campaign; the
measured-constant tables (the pre-campaign asymmetric-overfit and localization
studies, and the GEMM microbenchmark) carry their numbers inline.

    python visualize/tables/paper_tables.py            # build all
    python visualize/tables/paper_tables.py --only main_results seed_robustness
    python visualize/tables/paper_tables.py --list

Name -> paper table:
    main_results (3) · seed_robustness (3b) · alpha_sweep (4) · mask_selection (5)
    · gemm_efficiency (6) · alpha_gamma_grid (7) · compression_time
    · asymmetric_overfit (1) · localization (2)
"""

from __future__ import annotations

import argparse

import tablekit as T

MODEL_DISP = {"llama2-7b": "7B", "llama3.1-8b": "8B", "llama2-13b": "13B"}
SPARSITY_DISP = {"s50": r"50\%", "s60": r"60\%", "s70": r"70\%"}
CALIB = "128x1024"


def _sel_label(obr: dict, selected: dict) -> str:
    if selected is obr:
        return "OBR"
    alpha, gamma = T.mix_of(selected["calibration"])
    return f"$({alpha:g}, {gamma:g})$"


# --------------------------------------------------------------------------- #
# Table 3 -- main results
# --------------------------------------------------------------------------- #
MAIN_SPEC = [
    ("llama2-7b", "W4A4", "s50"), ("llama2-7b", "W4A4", "s60"),
    ("llama2-7b", "W4A4", "s70"), ("llama2-7b", "W4A4", "4to8p-obs"),
    ("llama2-7b", "W4A16", "s50"), ("llama2-7b", "W4A16", "s70"),
    ("llama2-7b", "W2A16", "s50"),
    ("llama3.1-8b", "W4A4", "s50"), ("llama3.1-8b", "W4A4", "s60"),
    ("llama3.1-8b", "W4A4", "s70"), ("llama3.1-8b", "W4A4", "4to8p40-obs"),
    ("llama3.1-8b", "W4A16", "s50"), ("llama3.1-8b", "W4A16", "s70"),
    ("llama3.1-8b", "W2A16", "s50"),
    ("llama2-13b", "W4A4", "s50"), ("llama2-13b", "W4A4", "s60"),
    ("llama2-13b", "W4A4", "s70"), ("llama2-13b", "W4A4", "4to8p40-obs"),
    ("llama2-13b", "W2A16", "s50"),
]


def main_results(runs) -> tuple[str, str]:
    blocks: list[list[str]] = [[], []]
    for model, bits, pattern in MAIN_SPEC:
        tags = runs.get((model, bits, pattern, CALIB))
        if not tags:
            print(f"  skip main row {model} {bits} {pattern}: no data")
            continue
        obr, _, sel = T.select(tags)
        if obr is None:
            continue
        pairwise = pattern.startswith("4to8")
        setting = T.BITS_LABEL[bits] + (" pairwise 4:8" if pairwise else "")
        sparsity = "4:8" if pairwise else SPARSITY_DISP[pattern]
        qcell = T.qdrift_cell(sel, obr) if sel is not obr else "--"
        row = " & ".join([
            MODEL_DISP[model], setting, sparsity,
            T.ppl_cell(obr), qcell, _sel_label(obr, sel),
        ])
        blocks[0 if model == "llama2-7b" else 1].append(row)
    header = r"Model & Setting & Sparsity & OBR & QDrift & $(\alpha,\gamma)$"
    body = _render("lllccc", header, blocks)
    return "main_results", body


# --------------------------------------------------------------------------- #
# Table 3b -- seed robustness
# --------------------------------------------------------------------------- #
SEED_SPEC = [
    ("llama2-7b", "W2A16", "s50"), ("llama2-7b", "W4A4", "s70"),
    ("llama3.1-8b", "W2A16", "s50"), ("llama3.1-8b", "W4A4", "s70"),
]


def seed_robustness(runs) -> tuple[str, str]:
    rows = []
    for model, bits, pattern in SEED_SPEC:
        tags = runs.get((model, bits, pattern, CALIB))
        if not tags:
            print(f"  skip seed row {model} {bits} {pattern}: no data")
            continue
        obr, _, sel = T.select(tags)
        obr_seeds = T.seeds_of(tags, "seq") or T.seeds_of(tags, "qdrift-a0c0")
        sel_seeds = T.seeds_of(tags, sel["calibration"])
        if len(obr_seeds) < 2 or len(sel_seeds) < 2:
            print(f"  skip seed row {model} {bits} {pattern}: <2 seeds")
            continue
        om, osd = T.mean_sd(obr_seeds)
        sm, ssd = T.mean_sd(sel_seeds)
        alpha, gamma = T.mix_of(sel["calibration"])
        setting = f"{T.BITS_LABEL[bits]} {SPARSITY_DISP[pattern]} $({alpha:g},{gamma:g})$"
        rows.append(" & ".join([
            MODEL_DISP[model], setting,
            f"${om:.2f}\\pm{osd:.2f}$",
            f"$\\mathbf{{{sm:.2f}\\pm{ssd:.2f}}}$",
            f"${100 * (sm - om) / om:+.1f}\\%$",
        ]))
    header = r"Model & Setting & OBR & QDrift & vs OBR"
    body = _render("ll" + "c" * 3, header, [rows])
    return "seed_robustness", body


# --------------------------------------------------------------------------- #
# Table 4 -- alpha sweep (7B W4A4 s70, gamma = 0)
# --------------------------------------------------------------------------- #
def alpha_sweep(runs) -> tuple[str, str]:
    tags = runs.get(("llama2-7b", "W4A4", "s70", CALIB))
    if not tags:
        raise SystemExit("alpha_sweep: no 7B W4A4 s70 data")
    line = []
    for tag, record in tags.items():
        if "@" in tag:
            continue
        mix = T.mix_of(tag)
        if mix and mix[1] == 0.0:
            line.append((mix[0], record))
    line.sort(key=lambda p: p[0])
    best = min(line, key=lambda p: T._val(p[1]))
    rows = []
    for alpha, record in line:
        label = r"$(0, 0)$ OBR" if alpha == 0 else f"$({alpha:g}, 0)$"
        cell = T.ppl_cell(record)
        rows.append(f"{label} & " + (f"\\textbf{{{cell}}}" if record is best[1] else cell))
    header = r"$(\alpha,\gamma)$ & WikiText-2 test (val)"
    return "alpha_sweep", _render("lc", header, [rows])


# --------------------------------------------------------------------------- #
# Table 5 -- mask selection (7B pairwise 4:8)
# --------------------------------------------------------------------------- #
def mask_selection(runs) -> tuple[str, str]:
    score = runs.get(("llama2-7b", "W4A4", "4to8p", CALIB), {})
    obs = runs.get(("llama2-7b", "W4A4", "4to8p-obs", CALIB), {})
    if not score or not obs:
        raise SystemExit("mask_selection: need both 4to8p (score) and 4to8p-obs grids")
    obr_obs, _, sel = T.select(obs)
    alpha, gamma = T.mix_of(sel["calibration"])
    rows = [
        "OBR & SparseGPT pair score & " + T.ppl_cell(score.get("seq")),
        "OBR & Exact block-sequential & " + T.ppl_cell(obr_obs),
        f"QDrift $({alpha:g},{gamma:g})$ & Exact block-sequential & "
        + f"\\textbf{{{T.ppl_cell(sel)}}}",
    ]
    header = r"Compensation & Mask & WikiText-2 test (val)"
    return "mask_selection", _render("llc", header, [rows])


# --------------------------------------------------------------------------- #
# Table 7 -- (alpha, gamma) grid (8B W4A4, s50 and s70)
# --------------------------------------------------------------------------- #
def alpha_gamma_grid(runs) -> tuple[str, str]:
    axis = [0.0, 0.25, 0.5]

    def grid(pattern):
        tags = runs.get(("llama3.1-8b", "W4A4", pattern, CALIB))
        if not tags:
            raise SystemExit(f"alpha_gamma_grid: no 8B W4A4 {pattern} data")
        cells = {}
        for tag, record in tags.items():
            if "@" in tag:
                continue
            mix = T.mix_of(tag)
            if mix and mix[0] in axis and mix[1] in axis:
                cells[mix] = T._val(record)
        return cells

    g50, g70 = grid("s50"), grid("s70")
    best50, best70 = min(g50.values()), min(g70.values())

    def fmt(cells, a, g, best):
        v = cells.get((a, g))
        if v is None:
            return "--"
        return f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}"

    rows = []
    for a in axis:
        left = " & ".join(fmt(g50, a, g, best50) for g in axis)
        right = " & ".join(fmt(g70, a, g, best70) for g in axis)
        rows.append(f"$\\alpha{{=}}{a:g}$ & {left} & {right}")

    header = (
        r" & \multicolumn{3}{c}{50\% sparsity} & \multicolumn{3}{c}{70\% sparsity} \\"
        + "\n"
        + r"$\alpha$ & $\gamma{=}0$ & $\gamma{=}0.25$ & $\gamma{=}0.5$ "
        + r"& $\gamma{=}0$ & $\gamma{=}0.25$ & $\gamma{=}0.5$"
    )
    body = _render("l" + "c" * 6, header, [rows])
    return "alpha_gamma_grid", body


# --------------------------------------------------------------------------- #
# compression time -- OBR vs selected QDrift wall-clock, from logged compress_s
# --------------------------------------------------------------------------- #
COMPRESS_SPEC = [
    ("llama2-7b", "W4A4", "s70"), ("llama2-7b", "W2A16", "s50"),
    ("llama3.1-8b", "W4A4", "s70"), ("llama3.1-8b", "W2A16", "s50"),
]


def compression_time(runs) -> tuple[str, str]:
    rows = []
    for model, bits, pattern in COMPRESS_SPEC:
        tags = runs.get((model, bits, pattern, CALIB))
        if not tags:
            continue
        obr, _, sel = T.select(tags)
        if obr is None or sel is obr:
            continue
        o, q = obr.get("compress_s"), sel.get("compress_s")
        if not o or not q:
            print(f"  skip compress row {model} {bits} {pattern}: no compress_s")
            continue
        alpha, gamma = T.mix_of(sel["calibration"])
        rows.append(" & ".join([
            MODEL_DISP[model], f"{T.BITS_LABEL[bits]} {SPARSITY_DISP[pattern]}",
            f"{o / 60:.0f}", f"{q / 60:.0f}", f"${100 * (q - o) / o:+.0f}\\%$",
        ]))
    if not rows:
        raise SystemExit("compression_time: no paired compress_s found")
    header = r"Model & Setting & OBR (min) & QDrift (min) & overhead"
    return "compression_time", _render("ll" + "c" * 3, header, [rows])


# --------------------------------------------------------------------------- #
# Measured-constant tables (not in results/v2)
# --------------------------------------------------------------------------- #
def gemm_efficiency(_runs) -> tuple[str, str]:
    """Table 6 -- dense vs pairwise-4:8 sparse INT4 GEMM on LLaMA-2-7B linear
    shapes, SM 8.9 (benchmark/kernels/checks/w4a4_check.py::bench_shape)."""
    rows = [
        r"q/o-proj & $4096\times4096$ & 73.5 & 60.8 & 1.21$\times$ & 31.5 & 16.8 & 1.87$\times$",
        r"gate/up-proj & $11008\times4096$ & 226.9 & 169.1 & 1.34$\times$ & 31.4 & 12.0 & 2.61$\times$",
        r"down-proj & $4096\times11008$ & 163.1 & 143.6 & 1.14$\times$ & 75.9 & 24.8 & \textbf{3.05$\times$}",
    ]
    header = (
        r" & & \multicolumn{2}{c}{Prefill ($R{=}2048$)} & & \multicolumn{2}{c}{Decode ($R{=}32$)} & \\"
        + "\n"
        + r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}"
        + "\n"
        + r"Layer & Shape ($c_{out}{\times}c_{in}$) & Dense & Sparse & Speedup & Dense & Sparse & Speedup"
    )
    body = _render("llcccccc", header, [rows])
    return "gemm_efficiency", body


def asymmetric_overfit(_runs) -> tuple[str, str]:
    """Table 1 -- OBR vs the pure asymmetric target (pre-campaign calibration)."""
    rows = [
        r"W2A16 g128 & $32\times1024$ & 50\% & 26.46 & 12.41",
        r"W4A4KV4 & $32\times1024$ & 50\% & 7.74 & 7.94",
        r"W4A4KV4 & $32\times1024$ & 70\% & 38.38 & 52.96",
        r"W4A4KV4 & $128\times2048$ & 50\% & 7.73 & 7.38",
        r"W4A4KV4 & $128\times2048$ & 70\% & 53.0 & 46.1",
    ]
    header = r"Setting & Calibration & Sparsity & OBR & Asymmetric"
    return "asymmetric_overfit", _render("lll" + "cc", header, [rows])


def localization(_runs) -> tuple[str, str]:
    """Table 2 -- asymmetric target on a block prefix only (pre-campaign)."""
    rows = [
        r"none (OBR) & 38.38 & 39.88", r"blocks 0--3 & 31.71 & 31.38",
        r"blocks 0--7 & 32.43 & 31.16", r"blocks 0--11 & 32.91 & 30.93",
        r"blocks 0--15 & 34.96 & 32.54", r"blocks 8--31 & 94.32 & 89.12",
        r"attention only & 42.58 & 39.47", r"MLP gate/up only & 4227.7 & 4076.3",
        r"MLP down\_proj only & 515.7 & 496.7", r"all blocks & 52.96 & 47.67",
    ]
    header = r"Layers with asymmetric target & Test & Val"
    return "localization", _render("lcc", header, [rows])


# --------------------------------------------------------------------------- #
# Rendering + driver
# --------------------------------------------------------------------------- #
def _render(colspec: str, header: str, blocks: list[list[str]]) -> str:
    """A booktabs tabular whose row blocks are separated by ``\\midrule``."""
    lines = [r"\begin{tabular}{%s}" % colspec, r"\toprule", header + r" \\", r"\midrule"]
    for i, block in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        lines.extend(row + r" \\" for row in block)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table 8 -- zero-shot accuracy (mean over the lm-eval task set), OBR vs selected
# --------------------------------------------------------------------------- #
def zeroshot(runs) -> tuple[str, str]:
    """Average zero-shot accuracy, OBR vs the val-selected QDrift mixture. Mirrors
    MAIN_SPEC; rows without a ``zeroshot`` field (not yet evaluated) are skipped."""
    rows: list[str] = []
    for model, bits, pattern in MAIN_SPEC:
        tags = runs.get((model, bits, pattern, CALIB))
        if not tags:
            continue
        obr, _, sel = T.select(tags)
        if obr is None or sel is None:
            continue
        oz, sz = obr.get("zeroshot"), sel.get("zeroshot")
        if not oz or not sz:
            print(f"  skip zeroshot row {model} {bits} {pattern}: no zeroshot")
            continue
        pairwise = pattern.startswith("4to8")
        setting = T.BITS_LABEL[bits] + (" pairwise 4:8" if pairwise else "")
        sparsity = "4:8" if pairwise else SPARSITY_DISP[pattern]
        oa, sa = float(oz["avg"]), float(sz["avg"])
        rows.append(" & ".join([
            MODEL_DISP[model], setting, sparsity,
            f"{oa:.1f}", f"$\\mathbf{{{sa:.1f}}}$", f"${sa - oa:+.1f}$",
        ]))
    if not rows:
        raise SystemExit("zeroshot: no results carry a zeroshot field yet")
    header = r"Model & Setting & Sparsity & OBR & QDrift & $\Delta$"
    body = _render("lll" + "c" * 3, header, [rows])
    return "zeroshot", body


BUILDERS = {
    "main_results": main_results,
    "seed_robustness": seed_robustness,
    "alpha_sweep": alpha_sweep,
    "mask_selection": mask_selection,
    "gemm_efficiency": gemm_efficiency,
    "alpha_gamma_grid": alpha_gamma_grid,
    "compression_time": compression_time,
    "asymmetric_overfit": asymmetric_overfit,
    "localization": localization,
    "zeroshot": zeroshot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", choices=list(BUILDERS), help="build a subset")
    parser.add_argument("--folder", default=str(T.RESULTS_V2), help="v2 results folder")
    parser.add_argument("--list", action="store_true", help="list table names and exit")
    args = parser.parse_args()
    if args.list:
        for name in BUILDERS:
            print(name)
        return
    runs = T.load_runs(args.folder)
    for name in args.only or list(BUILDERS):
        try:
            table_name, body = BUILDERS[name](runs)
        except SystemExit as err:
            print(f"skip {name}: {err}")
            continue
        T.save_table(table_name, body)


if __name__ == "__main__":
    main()
