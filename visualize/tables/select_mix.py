# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select one global (alpha, gamma) per setting by WikiText-2 validation PPL.

For each (setting, pattern) it reads every calibration tag found in a v2
results folder -- ``seq`` (OBR), ``qdrift`` (alpha 1, gamma 0) and
``qdrift-a{alpha}c{gamma}`` -- picks the tag with the lowest validation PPL,
and prints a table row with the reported test PPL.  The grid is assumed to
contain OBR; on a tie OBR wins, so the selection is never worse than OBR on
the validation criterion.

    python visualize/tables/select_mix.py results/v2
    python visualize/tables/select_mix.py results/v2 --json selections.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TAG_RE = re.compile(r"^qdrift-a(?P<alpha>[0-9.]+)c(?P<gamma>[0-9.]+)$")


def _setting(record: dict) -> str:
    criterion = record.get("criterion", "sparsegpt")
    quant = record.get("quant", "gptq")
    size = record.get("calib_size", "32x1024")
    return (
        f"W{record['wbits']}A{record['abits']}"
        + ("" if criterion == "sparsegpt" else f" {criterion}")
        + ("" if quant == "gptq" else f" {quant}")
        + ("" if size == "32x1024" else f" {size}")
    )


def _mix(tag: str) -> tuple[float, float] | None:
    """The (alpha, gamma) of a calibration tag, or None for OBR."""
    if tag == "seq":
        return (0.0, 0.0)
    if tag == "qdrift":
        return (1.0, 0.0)
    match = TAG_RE.match(tag)
    if match:
        return (float(match["alpha"]), float(match["gamma"]))
    return None


def _cell(ppl: dict[str, float]) -> str:
    val = ppl.get("wikitext2_val")
    test = ppl.get("wikitext2")
    if test is None:
        return "-"
    return f"{test:.2f}" + (f" ({val:.2f})" if val is not None else "")


def main(argv: list[str] | None = None) -> int:
    """Print the selected table and optionally dump the selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=str(ROOT / "results" / "v2"))
    parser.add_argument("--json", help="write the selected (alpha, gamma) per setting")
    parser.add_argument(
        "--require-val",
        action="store_true",
        help="skip runs without a validation PPL instead of using test",
    )
    args = parser.parse_args(argv)

    runs: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted(Path(args.folder).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        ppl = record.get("ppl")
        tag = record.get("calibration")
        if not ppl or "wikitext2" not in ppl or tag is None:
            continue
        if args.require_val and ppl.get("wikitext2_val") is None:
            continue
        if _mix(tag) is None:
            # Legacy diagnostic tags (per-layer, per-prefix, ridge variants)
            # are not part of the selectable grid.
            continue
        if record.get("mechanism", "obr") != "obr":
            # Retired mechanisms (QIR and friends) must not enter the grid.
            continue
        key = (record.get("model", "llama2-7b"), _setting(record), record["pattern"])
        runs.setdefault(key, {})[tag] = ppl

    selections: dict[str, dict] = {}
    print("| Model | Setting | Pattern | OBR | QDrift (1,0) | selected (alpha, gamma) | selected test (val) | vs OBR |")
    print("|---|---|---|---|---|---|---|---|")
    for (model, setting, pattern), tags in sorted(runs.items()):
        obr_tags = [tag for tag in ("seq", "qdrift-a0c0") if tag in tags]
        if not obr_tags:
            continue
        obr = tags[obr_tags[0]]
        base_key = (
            obr.get("wikitext2_val")
            if obr.get("wikitext2_val") is not None
            else obr["wikitext2"]
        )
        best_tag, best_key = obr_tags[0], base_key
        for tag, ppl in tags.items():
            key = ppl.get("wikitext2_val")
            if key is None:
                key = ppl["wikitext2"]
            if key < best_key:
                best_tag, best_key = tag, key
        mix = _mix(best_tag)
        label = (
            "OBR"
            if best_tag in ("seq", "qdrift-a0c0")
            else f"({mix[0]:g}, {mix[1]:g})"
        )
        delta = 100.0 * (best_key - base_key) / base_key
        one = tags.get("qdrift", {})
        selections[f"{model} | {setting} | {pattern}"] = {
            "tag": best_tag,
            "alpha": mix[0] if mix else 0.0,
            "gamma": mix[1] if mix else 0.0,
            "test": tags[best_tag]["wikitext2"],
            "val": tags[best_tag].get("wikitext2_val"),
            "delta_vs_obr_pct": delta,
        }
        print(
            f"| {model} | {setting} | {pattern} | {_cell(obr)} | {_cell(one) if one else '-'} "
            f"| {label} | {_cell(tags[best_tag])} | {delta:+.2f}% |"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(selections, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
