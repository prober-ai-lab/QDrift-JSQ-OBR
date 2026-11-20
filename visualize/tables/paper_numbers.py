# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every number the paper text cites, as LaTeX macros: ``tables/tex/numbers.tex``.

The paper never types a result by hand; it writes ``\\qd{<key>}``.  An unknown key
is a compile error, so a typo or a renamed result cannot slip through.  Values are
computed with the same loading and selection as ``paper_tables.py``, so the text
and the tables always agree.

Keys are ``/``-separated; ``<m>`` is a model id (``llama2-7b``, ``llama3.1-8b``,
``llama2-13b``) and ``<s>`` a setting (``w4a4-s70``, ``w4a16-s50``, ``w2a16-s50``,
``w4a4-4to8p`` fully pairwise, ``w4a4-4to8m`` mixed 40% pairwise).

    <m>/<s>/{obr,obr-val,qd,qd-val,pure,pure-val}     test / val perplexity (2 dp)
    <m>/<s>/{qd-delta,pure-delta}                     signed % vs OBR, test (1 dp)
    <m>/<s>/{qd-red}                                  |qd-delta| rounded to an integer
    <m>/<s>/{qd-abs,pure-abs}                         |qd-delta|, |pure-delta| (1 dp)
    <m>/qd-delta-{min,max}                            signed range of qd-delta over a model's rows
    overfit/w4a4-s50/max-abs                          largest |pure-delta| at W4A4KV4 50%
    <m>/<s>/{alpha,gamma,mix}                         selected mixture
    <m>/<s>/seed/{obr-mean,obr-sd,qd-mean,qd-sd,delta,gap,gap-ratio}
    seed/min-gap-ratio                                smallest gap / larger sd over the seed rows
    <m>/<s>/zs/<task>/{obr,qd,diff}                   zero-shot accuracy (1 dp), diff signed
    <m>/<s>/zs/{wins,min-diff,max-diff,max-abs,min-abs}  over the six tasks (avg excluded)
    zs/<g>/{ppl-red-min,ppl-red-max,wins,pairs,max-abs-llama2}  per setting group <g>
                                                      (w4a4-s50, w2a16-s50, 4to8 = 4:8p and 4:8m)
    <m>/<s>/zs/retain                                 best avg / FP16 avg, %
    <m>/fp16/zs/<task>                                FP16 dense reference
    <m>/<s>/time/{obr,qd,overhead}                    compression minutes, % overhead
    zs/{obr-avg-wins,settings}                        settings where OBR has the higher avg
    zs/obr-avg-wins-lt1                               ... of which by less than one point
    zs/chance                                         chance level of the six-task average
    time/overhead-{min,max}                           range of the compression-time overhead
    loc/<row>/{ppl,ppl-val,delta}, loc/late-ratio     Table 2 (localization)
    sweep/a<alpha>/{ppl,ppl-val}                      Table 4 (gamma = 0)
    mask/{score,exact,qd,mask-gain,qd-delta}          Table 5
    grid/<s>/a<a>g<g>, grid/s50/spread, grid/s70/worst-ratio, grid/s70/opt{,-mix}
    llama2-7b/w4a4-s70/clean{,-val}                   pure clean target (0, 1)
    calib/<size>/<s>/obr                              OBR at other calibration sizes
    drift/<comp>/{ppl,first-above-1,rel-min,rel-max,ratio-min,ratio-max}, drift/dense/ppl
    drift/<comp>/removed-{min,max}                    Fig. 4 right: % of inj.+inh. removed, blocks >= 4
    drift/cancel/{min,max}                            Fig. 4 left: % linearly cancellable (OBR prefix),
                                                      per-layer-type block means, blocks >= 8
    grids/b<bits>/{best-ratio,best-s,below,above}     Appendix C: mid-rise / mid-tread joint error
    regime/{dense,sparse,quant,joint}, mem/{fp16,w4a4,ratio,fp16-peak,w4a4-peak,peak-ratio}
    gemm/{prefill,decode}-{min,max}

Signed values carry ``\\ensuremath{-}`` / ``\\ensuremath{+}`` so they work in text and
math; no value contains ``\\%``.

    python visualize/tables/paper_numbers.py            # write tables/tex/numbers.tex
    python visualize/tables/paper_numbers.py --list     # print key = value
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

import paper_tables as P  # noqa: E402
import tablekit as T  # noqa: E402
from drift_accumulation import PAPER_INPUT as MINITEST  # noqa: E402
from regime_tradeoff import regimes  # noqa: E402

OUT = T.TABLES / "tex" / "numbers.tex"
TASKS = [key for key, _ in P.ZEROSHOT_TASKS if key != "avg"]


def setting_id(bits: str, pattern: str) -> str:
    """``w4a4-s70``; 4:8 patterns become ``4to8p`` (all pairwise) or ``4to8m`` (mixed)."""
    if pattern.startswith("4to8"):
        pattern = "4to8p" if pattern.split("-")[0] == "4to8p" else "4to8m"
    return f"{bits.lower()}-{pattern}"


def signed(value: float, digits: int = 1) -> str:
    sign = "-" if round(value, digits) < 0 else "+"
    return rf"\ensuremath{{{sign}}}{abs(value):.{digits}f}"


class Numbers:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def __setitem__(self, key: str, value) -> None:
        if key in self.values:
            raise SystemExit(f"duplicate number key {key}")
        self.values[key] = value if isinstance(value, str) else f"{value}"

    def ppl(self, key: str, record: dict) -> None:
        self[key] = f"{record['ppl']['wikitext2']:.2f}"
        self[f"{key}-val"] = f"{T._val(record):.2f}"


def main_numbers(n: Numbers, runs) -> None:
    model_deltas, overfit_s50 = defaultdict(list), []
    for model, bits, pattern in P.MAIN_SPEC + [s for s in P.OVERFIT_SPEC if s not in P.MAIN_SPEC]:
        tags = runs.get((model, bits, pattern, P.CALIB))
        if not tags:
            continue
        obr, pure, sel = T.select(tags)
        base = f"{model}/{setting_id(bits, pattern)}"
        n.ppl(f"{base}/obr", obr)
        if pure is not None:
            n.ppl(f"{base}/pure", pure)
            n[f"{base}/pure-delta"] = signed(T.delta_vs_obr(pure, obr))
            n[f"{base}/pure-abs"] = f"{abs(T.delta_vs_obr(pure, obr)):.1f}"
            if (bits, pattern) == ("W4A4", "s50"):
                overfit_s50.append(abs(T.delta_vs_obr(pure, obr)))
        if (model, bits, pattern) not in P.MAIN_SPEC:
            continue
        n.ppl(f"{base}/qd", sel)
        delta = T.delta_vs_obr(sel, obr)
        n[f"{base}/qd-delta"] = signed(delta)
        n[f"{base}/qd-red"] = f"{abs(delta):.0f}"
        n[f"{base}/qd-abs"] = f"{abs(delta):.1f}"
        model_deltas[model].append(round(delta, 1))
        alpha, gamma = T.mix_of(sel["calibration"])
        n[f"{base}/alpha"], n[f"{base}/gamma"] = f"{alpha:g}", f"{gamma:g}"
        n[f"{base}/mix"] = f"$({alpha:g},{gamma:g})$"
    for model, deltas in model_deltas.items():
        n[f"{model}/qd-delta-min"], n[f"{model}/qd-delta-max"] = signed(min(deltas)), signed(max(deltas))
    n["overfit/w4a4-s50/max-abs"] = f"{max(overfit_s50):.1f}"


def seed_numbers(n: Numbers, runs) -> None:
    ratios = []
    for model, bits, pattern in P.SEED_SPEC:
        tags = runs[(model, bits, pattern, P.CALIB)]
        _, _, sel = T.select(tags)
        om, osd = T.mean_sd(T.seeds_of(tags, "seq") or T.seeds_of(tags, "qdrift-a0c0"))
        sm, ssd = T.mean_sd(T.seeds_of(tags, sel["calibration"]))
        base = f"{model}/{setting_id(bits, pattern)}/seed"
        n[f"{base}/obr-mean"], n[f"{base}/obr-sd"] = f"{om:.2f}", f"{osd:.2f}"
        n[f"{base}/qd-mean"], n[f"{base}/qd-sd"] = f"{sm:.2f}", f"{ssd:.2f}"
        n[f"{base}/delta"] = signed(100 * (sm - om) / om)
        n[f"{base}/gap"] = f"{om - sm:.2f}"
        n[f"{base}/gap-ratio"] = f"{(om - sm) / max(osd, ssd):.1f}"
        ratios.append((om - sm) / max(osd, ssd))
    n["seed/min-gap-ratio"] = f"{min(ratios):.1f}"


def zeroshot_numbers(n: Numbers, runs) -> None:
    fp16 = {}
    for model in P.MODEL_DISP:
        path = P.FP16_RESULTS / f"{model}__zeroshot.json"
        if path.exists():
            fp16[model] = json.loads(path.read_text(encoding="utf-8"))["zeroshot"]
            for key in TASKS + ["avg"]:
                n[f"{model}/fp16/zs/{key}"] = f"{fp16[model][key]:.1f}"
    obr_wins = obr_wins_lt1 = settings = 0
    group = defaultdict(lambda: {"red": [], "wins": 0, "pairs": 0, "abs-llama2": []})
    for model, bits, pattern in P.MAIN_SPEC:
        tags = runs.get((model, bits, pattern, P.CALIB))
        if not tags:
            continue
        obr, _, sel = T.select(tags)
        oz, sz = obr.get("zeroshot"), sel.get("zeroshot")
        if not oz or not sz:
            continue
        base = f"{model}/{setting_id(bits, pattern)}/zs"
        diffs = []
        for key in TASKS + ["avg"]:
            oa, sa = round(float(oz[key]), 1), round(float(sz[key]), 1)
            n[f"{base}/{key}/obr"], n[f"{base}/{key}/qd"] = f"{oa:.1f}", f"{sa:.1f}"
            n[f"{base}/{key}/diff"] = signed(sa - oa)
            if key != "avg":
                diffs.append(round(sa - oa, 1))
        n[f"{base}/wins"] = f"{sum(d > 0 for d in diffs)}"
        n[f"{base}/min-diff"], n[f"{base}/max-diff"] = signed(min(diffs)), signed(max(diffs))
        n[f"{base}/max-abs"] = f"{max(abs(d) for d in diffs):.1f}"
        n[f"{base}/min-abs"] = f"{min(abs(d) for d in diffs):.1f}"
        sid = setting_id(bits, pattern)
        g = group["4to8" if "4to8" in sid else sid]
        g["red"].append(abs(T.delta_vs_obr(sel, obr)))
        g["wins"] += sum(d > 0 for d in diffs)
        g["pairs"] += len(diffs)
        if model.startswith("llama2"):
            g["abs-llama2"] += [abs(d) for d in diffs]
        if model in fp16:
            best = max(round(float(oz["avg"]), 1), round(float(sz["avg"]), 1))
            n[f"{base}/retain"] = f"{100 * best / fp16[model]['avg']:.0f}"
        settings += 1
        oa, sa = round(float(oz["avg"]), 1), round(float(sz["avg"]), 1)
        obr_wins += oa > sa
        obr_wins_lt1 += sa < oa < sa + 1.0
    n["zs/obr-avg-wins"], n["zs/settings"] = f"{obr_wins}", f"{settings}"
    n["zs/obr-avg-wins-lt1"] = f"{obr_wins_lt1}"
    for gid in ("w4a4-s50", "w2a16-s50", "4to8"):
        g = group[gid]
        n[f"zs/{gid}/ppl-red-min"] = f"{min(g['red']):.1f}"
        n[f"zs/{gid}/ppl-red-max"] = f"{max(g['red']):.1f}"
        n[f"zs/{gid}/wins"], n[f"zs/{gid}/pairs"] = f"{g['wins']}", f"{g['pairs']}"
        n[f"zs/{gid}/max-abs-llama2"] = f"{max(g['abs-llama2']):.1f}"
    # Chance level of each task (random guessing; LAMBADA is open-vocabulary, so 0).
    chance = {"piqa": 50.0, "hellaswag": 25.0, "arc_easy": 25.0, "arc_challenge": 25.0,
              "winogrande": 50.0, "lambada_openai": 0.0}
    n["zs/chance"] = f"{sum(chance[t] for t in TASKS) / len(TASKS):.0f}"


def time_numbers(n: Numbers, runs) -> None:
    overheads = []
    for model, bits, pattern in P.COMPRESS_SPEC:
        obr, _, sel = T.select(runs[(model, bits, pattern, P.CALIB)])
        o, q = obr["compress_s"], sel["compress_s"]
        base = f"{model}/{setting_id(bits, pattern)}/time"
        n[f"{base}/obr"], n[f"{base}/qd"] = f"{o / 60:.0f}", f"{q / 60:.0f}"
        n[f"{base}/overhead"] = f"{100 * (q - o) / o:.0f}"
        overheads.append(round(100 * (q - o) / o))
    n["time/overhead-min"], n["time/overhead-max"] = f"{min(overheads)}", f"{max(overheads)}"


def ablation_numbers(n: Numbers, runs) -> None:
    ids = {"seq": "none", "qdrift": "all", "qdrift-Mattn": "attn",
           "qdrift-Mmlp_in": "mlp-in", "qdrift-Mmlp_out": "mlp-out"}
    loc = {}
    for _, tag in P.LOCALIZATION_ROWS:
        record = json.loads((T.RESULTS_V2 / f"{P.LOCALIZATION_BASE}_{tag}_seed42.json")
                            .read_text(encoding="utf-8"))
        loc[ids.get(tag, "b" + tag.removeprefix("qdrift-L"))] = record
    for row, record in loc.items():
        n.ppl(f"loc/{row}/ppl", record)
        n[f"loc/{row}/delta"] = signed(T.delta_vs_obr(record, loc["none"]))
    n["loc/late-ratio"] = f"{loc['b8-31']['ppl']['wikitext2'] / loc['none']['ppl']['wikitext2']:.1f}"

    s70 = runs[("llama2-7b", "W4A4", "s70", P.CALIB)]
    for tag, record in s70.items():
        mix = T.mix_of(tag)
        if "@" not in tag and mix and mix[1] == 0.0:
            n.ppl(f"sweep/a{mix[0]:g}/ppl", record)
    n.ppl("llama2-7b/w4a4-s70/clean", s70["qdrift-a0c1"])

    score = runs[("llama2-7b", "W4A4", "4to8p", P.CALIB)]["seq"]
    obr, _, sel = T.select(runs[("llama2-7b", "W4A4", "4to8p-obs", P.CALIB)])
    n.ppl("mask/score", score)
    n.ppl("mask/exact", obr)
    n.ppl("mask/qd", sel)
    n["mask/mask-gain"] = f"{-T.delta_vs_obr(obr, score):.0f}"
    n["mask/qd-delta"] = signed(T.delta_vs_obr(sel, obr))

    axis = (0.0, 0.25, 0.5)
    for pattern in ("s50", "s70"):
        tags = runs[("llama3.1-8b", "W4A4", pattern, P.CALIB)]
        cells = {T.mix_of(t): T._val(r) for t, r in tags.items()
                 if "@" not in t and T.mix_of(t) and set(T.mix_of(t)) <= set(axis)}
        for (a, g), v in cells.items():
            n[f"grid/{pattern}/a{a:g}g{g:g}"] = f"{v:.2f}"
        n[f"grid/{pattern}/spread"] = f"{100 * (max(cells.values()) / min(cells.values()) - 1):.1f}"
        if pattern == "s70":
            n["grid/s70/worst-ratio"] = f"{max(cells.values()) / cells[(0.0, 0.0)]:.2f}"
            _, _, best = T.select(tags)
            a, g = T.mix_of(best["calibration"])
            n["grid/s70/opt"], n["grid/s70/opt-mix"] = f"{T._val(best):.2f}", f"$({a:g},{g:g})$"

    for (model, bits, pattern, calib), tags in runs.items():
        if model == "llama2-7b" and calib != P.CALIB and "seq" in tags:
            n[f"calib/{calib}/{setting_id(bits, pattern)}/obr"] = f"{tags['seq']['ppl']['wikitext2']:.2f}"


def drift_numbers(n: Numbers) -> None:
    """Figs. 2 and 3 statistics, computed as the figures compute them (per-block means;
    inherited / injected is the ratio of the block means), over blocks >= 8."""
    record = json.loads(MINITEST.read_text(encoding="utf-8"))
    n["drift/dense/ppl"] = f"{math.exp(record['nll']['dense']):.2f}"
    for comp in ("none", "obr", "drift"):
        n[f"drift/{comp}/ppl"] = f"{math.exp(record['nll'][comp]):.2f}"
        drift, inh, inj = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in record["rows"]:
            if row["comp"] != comp:
                continue
            if row["type"] == "q":
                drift[row["block"]].append(row["drift"])
            inh[row["block"]].append(row["inherited"])
            inj[row["block"]].append(row["injected"])
        mean = {b: sum(v) / len(v) for b, v in drift.items()}
        ratio = {b: (sum(inh[b]) / len(inh[b])) / (sum(inj[b]) / len(inj[b])) for b in inh}
        deep = [b for b in sorted(mean) if b >= 8]
        above = [b for b in sorted(mean) if mean[b] > 1.0]
        n[f"drift/{comp}/first-above-1"] = f"{above[0]}" if above else "none"
        n[f"drift/{comp}/rel-min"] = f"{min(mean[b] for b in deep):.2f}"
        n[f"drift/{comp}/rel-max"] = f"{max(mean[b] for b in deep):.2f}"
        n[f"drift/{comp}/ratio-min"] = f"{min(ratio[b] for b in deep):.0f}"
        n[f"drift/{comp}/ratio-max"] = f"{max(ratio[b] for b in deep):.0f}"
        # Fig. 4 right, as cancel_figure: block mean of 1 - total / (injected + inherited).
        removed = defaultdict(list)
        for row in record["rows"]:
            if row["comp"] == comp:
                removed[row["block"]].append(1 - row["total"] / (row["injected"] + row["inherited"]))
        share = [100 * sum(v) / len(v) for b, v in sorted(removed.items()) if b >= 4]
        n[f"drift/{comp}/removed-min"] = f"{min(share):.1f}"
        n[f"drift/{comp}/removed-max"] = f"{max(share):.1f}"
    # Fig. 4 left, as cancel_figure: per layer type, block mean of the cancellable share on
    # the OBR prefix.
    cancel = defaultdict(list)
    for row in record["rows"]:
        if row["comp"] == "obr" and row["block"] >= 8 and not math.isnan(row["cancel"]):
            cancel[(row["type"], row["block"])].append(row["cancel"])
    means = [100 * sum(v) / len(v) for v in cancel.values()]
    n["drift/cancel/min"], n["drift/cancel/max"] = f"{min(means):.0f}", f"{max(means):.0f}"


GRID_MINITEST = sorted((T.ROOT / "results" / "minitests").glob("grid_sparsity_crossover__*.json"))


def grid_numbers(n: Numbers) -> None:
    """Appendix C: ratio of the OBR-compensated joint layer error of the mid-rise grid to
    that of the mid-tread grid, summed over the measured layers, per bit width and sparsity."""
    record = json.loads(GRID_MINITEST[0].read_text(encoding="utf-8"))
    errors = defaultdict(float)
    for row in record["errors"]:
        if row.get("joint_err_obr") is not None:
            errors[(row["grid"], row["bits"], row["sparsity"])] += row["joint_err_obr"]
    for bits in sorted({b for _, b, _ in errors}):
        sparsities = sorted({s for g, b, s in errors if b == bits and g == "midrise"})
        ratio = {s: errors[("midrise", bits, s)] / errors[("midtread", bits, s)] for s in sparsities}
        best = min(ratio, key=ratio.get)
        n[f"grids/b{bits}/best-ratio"] = f"{ratio[best]:.2f}"
        n[f"grids/b{bits}/best-s"] = f"{100 * best:.0f}"
        above = [s for s in sparsities if ratio[s] >= 1.0]
        if above:
            first = above[0]
            below = [s for s in sparsities if s < first]
            n[f"grids/b{bits}/above"] = f"{100 * first:.0f}"
            n[f"grids/b{bits}/below"] = f"{100 * below[-1]:.0f}" if below else "none"


def efficiency_numbers(n: Numbers) -> None:
    r, mem = regimes()
    for key, (_, ppl, _) in zip(("dense", "sparse", "quant", "joint"), r):
        n[f"regime/{key}"] = f"{ppl:.2f}"
    n["mem/fp16"], n["mem/w4a4"] = f"{mem['fp16'] / 1e9:.1f}", f"{mem['w4a4'] / 1e9:.1f}"
    n["mem/ratio"] = f"{mem['fp16'] / mem['w4a4']:.1f}"
    n["mem/fp16-peak"] = f"{mem['fp16_peak'] / 1e9:.1f}"
    n["mem/w4a4-peak"] = f"{mem['w4a4_peak'] / 1e9:.1f}"
    n["mem/peak-ratio"] = f"{mem['fp16_peak'] / mem['w4a4_peak']:.1f}"
    for phase in ("prefill", "decode"):
        speedups = [row[phase][2] for row in P.GEMM_US]
        n[f"gemm/{phase}-min"], n[f"gemm/{phase}-max"] = f"{min(speedups):.2f}", f"{max(speedups):.2f}"


def collect() -> Numbers:
    runs = T.load_runs()
    n = Numbers()
    main_numbers(n, runs)
    seed_numbers(n, runs)
    zeroshot_numbers(n, runs)
    time_numbers(n, runs)
    ablation_numbers(n, runs)
    drift_numbers(n)
    grid_numbers(n)
    efficiency_numbers(n)
    return n


def render(n: Numbers) -> str:
    lines = [
        "% Autogenerated by visualize/tables/paper_numbers.py -- do not edit by hand.",
        "% Use \\qd{<key>}; an unknown key is a compile error. Key scheme: see the script.",
        "\\makeatletter",
        "\\providecommand\\qd[1]{\\@ifundefined{qd@#1}"
        "{\\PackageError{qdnumbers}{Unknown number key `#1'}{See numbers.tex}}"
        "{\\@nameuse{qd@#1}}}",
    ]
    lines += [f"\\@namedef{{qd@{k}}}{{{v}}}" for k, v in sorted(n.values.items())]
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print key = value and exit")
    args = parser.parse_args()
    n = collect()
    if args.list:
        for key, value in sorted(n.values.items()):
            print(f"{key} = {value}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(n), encoding="utf-8")
    print(f"wrote {OUT.relative_to(T.ROOT)} ({len(n.values)} numbers)")


if __name__ == "__main__":
    main()
