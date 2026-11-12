# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The v2 results as a markdown table.

Rows are (setting, pattern); columns are the calibration objectives found:
``seq`` (OBR) first, then ``qdrift`` (original DAC) and QDrift's mixtures
``qdrift-a{alpha}c{gamma}``.  Cells are WikiText-2 test perplexity with the
validation perplexity in parentheses when present; missing runs print ``-``.
Runs whose late half is not quantized by GPTQ, whose criterion is not
SparseGPT, or whose calibration size is not 32x1024 get their own rows.

    python visualize/tables/v2_table.py [results/v2]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cell(ppl: dict[str, float] | None) -> str:
    if not ppl or "wikitext2" not in ppl:
        return "-"
    val = ppl.get("wikitext2_val")
    return f"{ppl['wikitext2']:.2f}" + (f" ({val:.2f})" if val is not None else "")


def main() -> None:
    """Print the markdown table."""
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "v2"
    runs: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "ppl" not in record:
            continue
        criterion = record.get("criterion", "sparsegpt")
        quant = record.get("quant", "gptq")
        size = record.get("calib_size", "32x1024")
        setting = (
            f"W{record['wbits']}A{record['abits']}"
            + ("" if criterion == "sparsegpt" else f" {criterion}")
            + ("" if quant == "gptq" else f" {quant}")
            + ("" if size == "32x1024" else f" {size}")
        )
        runs[(setting, record["pattern"], record["calibration"])] = record["ppl"]
    columns = sorted({key[2] for key in runs}, key=lambda tag: (tag != "seq", tag != "qdrift", tag))
    rows = sorted({(key[0], key[1]) for key in runs})
    print("| Setting | Pattern | " + " | ".join(columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for setting, pattern in rows:
        cells = [_cell(runs.get((setting, pattern, tag))) for tag in columns]
        print(f"| {setting} | {pattern} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
