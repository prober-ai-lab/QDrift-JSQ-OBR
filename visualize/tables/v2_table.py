# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table 1 of the paper from the v2 run results.

Rows are (setting, sparsity); columns the four combinations of calibration
(``seq`` / ``qdrift``) and mechanism (``obr`` / ``qir``):
OBR = obr+seq, +DAC = obr+qdrift, +QIR = qir+seq, QDrift = qir+qdrift.
Cells are WikiText-2 perplexity; missing runs print ``-``.  Runs
whose late half is not quantized by GPTQ, or whose calibration size is not
32x1024, get their own rows (``W4A4 rtn``, ``W4A4 128x2048``).

    python visualize/tables/v2_table.py [results/v2]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = (
    ("OBR", "obr", "seq"),
    ("+DAC", "obr", "qdrift"),
    ("+QIR", "qir", "seq"),
    ("QDrift", "qir", "qdrift"),
)


def main() -> None:
    """Print the markdown table."""
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "v2"
    runs = {}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "ppl" not in record:
            continue
        criterion = record.get("criterion", "sparsegpt")
        quant = record.get("quant", "gptq")
        size = record.get("calib_size", "32x1024")
        key = (
            f"W{record['wbits']}A{record['abits']}"
            + ("" if criterion == "sparsegpt" else f" {criterion}")
            + ("" if quant == "gptq" else f" {quant}")
            + ("" if size == "32x1024" else f" {size}"),
            record["sparsity"],
            record["mechanism"],
            record["calibration"],
        )
        runs[key] = record["ppl"]
    settings = sorted({(k[0], k[1]) for k in runs}, key=lambda k: (-int(k[0][1]), k[0], k[1]))
    print("| Setting | Sparsity | " + " | ".join(c[0] for c in COLUMNS) + " |")
    print("|---|---|" + "---|" * len(COLUMNS))
    for setting, sparsity in settings:
        cells = []
        for _, mechanism, calibration in COLUMNS:
            ppl = runs.get((setting, sparsity, mechanism, calibration))
            cells.append(f"{ppl['wikitext2']:.2f}" if ppl else "-")
        print(f"| {setting} | {sparsity:.0%} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
