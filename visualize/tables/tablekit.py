# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared loading and LaTeX/PDF saving for the paper tables.

Every table is written twice under one name: ``tables/tex/<name>.tex`` -- a booktabs
``tabular`` fragment that the paper ``\\input``s inside its own ``table`` float --
and ``tables/pdf/<name>.pdf`` -- the same fragment compiled standalone with
``pdflatex`` for preview and direct inclusion.  This mirrors the figure pipeline
in ``visualize/plots/style.py`` (``figs/png/<name>.png`` + ``figs/pdf/<name>.pdf``).

Numbers are pulled live from the v2 result records so a table never drifts from
the campaign.  The mixture is selected on WikiText-2 *validation* perplexity; the
``vs OBR`` delta is computed on the reported *test* perplexity.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
RESULTS_V2 = ROOT / "results" / "v2"

TAG_RE = re.compile(r"^qdrift-a(?P<alpha>[0-9.]+)c(?P<gamma>[0-9.]+)$")

# Display names for the bit-width tag of a setting.
BITS_LABEL = {"W4A4": "W4A4KV4", "W4A16": "W4A16", "W2A16": "W2A16 g128"}

_STANDALONE = r"""\documentclass[border=6pt]{standalone}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[T1]{fontenc}
\begin{document}
%s
\end{document}
"""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def mix_of(tag: str | None) -> tuple[float, float] | None:
    """The ``(alpha, gamma)`` of a calibration tag, or ``None`` if not a grid tag."""
    if tag == "seq":
        return (0.0, 0.0)
    if tag == "qdrift":
        return (1.0, 0.0)
    match = TAG_RE.match(tag or "")
    if match:
        return (float(match["alpha"]), float(match["gamma"]))
    return None


def load_runs(folder: Path | str = RESULTS_V2) -> dict[tuple, dict[str, dict]]:
    """Group v2 records by ``(model, bits, pattern, calib)`` -> ``{tag: record}``.

    Only records on the selectable mixture grid (``seq``, ``qdrift``,
    ``qdrift-a*c*``) are kept; legacy diagnostic tags are dropped.  When several
    seeds of the same point exist they are stored under ``tag`` with a seed
    suffix (``tag@seed``) so callers can still find a single representative
    (seed 42) while seed studies see them all.
    """
    runs: dict[tuple, dict[str, dict]] = {}
    for path in sorted(Path(folder).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        ppl = record.get("ppl")
        tag = record.get("calibration")
        if not ppl or "wikitext2" not in ppl or mix_of(tag) is None:
            continue
        key = (
            record.get("model"),
            f"W{record['wbits']}A{record['abits']}",
            record["pattern"],
            record.get("calib_size"),
        )
        seed = record.get("seed", 42)
        store_tag = tag if seed == 42 else f"{tag}@{seed}"
        runs.setdefault(key, {})[store_tag] = record
    return runs


def _val(record: dict) -> float:
    """Validation PPL, falling back to test when validation is absent."""
    ppl = record["ppl"]
    return ppl["wikitext2_val"] if ppl.get("wikitext2_val") is not None else ppl["wikitext2"]


def select(tags: dict[str, dict]) -> tuple[dict | None, dict | None, dict | None]:
    """Return ``(obr, pure, selected)`` records for one grid (seed 42 only).

    ``selected`` is the lowest-validation point; ties break toward the smallest
    mixture (then smaller gamma, then smaller alpha), so OBR wins a true tie and
    the simpler mixture wins among equal QDrift points.  This keeps the choice
    reproducible and never worse than OBR on the criterion.  ``pure`` is
    ``(1, 0)`` or ``None``.
    """
    seed42 = {t: r for t, r in tags.items() if "@" not in t}
    obr = seed42.get("seq") or seed42.get("qdrift-a0c0")
    if obr is None:
        return None, None, None

    def rank(record: dict) -> tuple:
        alpha, gamma = mix_of(record["calibration"]) or (0.0, 0.0)
        return (round(_val(record), 6), alpha + gamma, gamma, alpha)

    selected = min(seed42.values(), key=rank)
    return obr, seed42.get("qdrift"), selected


def seeds_of(tags: dict[str, dict], base_tag: str) -> list[float]:
    """Test PPLs of one calibration tag across every available seed."""
    out = []
    for tag, record in tags.items():
        if tag == base_tag or tag.startswith(f"{base_tag}@"):
            out.append(record["ppl"]["wikitext2"])
    return sorted(out)


# --------------------------------------------------------------------------- #
# Cell formatting
# --------------------------------------------------------------------------- #
def ppl_cell(record: dict | None) -> str:
    """``test (val)`` or ``--``."""
    if record is None:
        return "--"
    ppl = record["ppl"]
    val = ppl.get("wikitext2_val")
    return f"{ppl['wikitext2']:.2f} ({val:.2f})" if val is not None else f"{ppl['wikitext2']:.2f}"


def delta_vs_obr(record: dict, obr: dict) -> float:
    """Relative **test**-PPL change of ``record`` against OBR, in percent.

    Test-relative so the subscript is consistent with the headline test PPL in
    the cell; selection still happens on validation (noted in the caption).
    """
    return 100.0 * (record["ppl"]["wikitext2"] - obr["ppl"]["wikitext2"]) / obr["ppl"]["wikitext2"]


def qdrift_cell(record: dict, obr: dict) -> str:
    """Bold ``test (val)`` with the test-relative delta as a subscript.

    The delta is written in parentheses so it reads ``7.37 (7.47)`` with a
    small ``(-4.8%)`` subscript, and never looks detached from the number.
    """
    return r"\textbf{%s}$_{(%+.1f\%%)}$" % (ppl_cell(record), delta_vs_obr(record, obr))


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and population standard deviation."""
    mean = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, sd


# --------------------------------------------------------------------------- #
# LaTeX assembly and compilation
# --------------------------------------------------------------------------- #
def tabular(colspec: str, header: str, rows: list[str], groups: list[str] | None = None) -> str:
    """Assemble a booktabs ``tabular``.

    ``header`` and each ``rows`` entry are LaTeX row bodies without the trailing
    ``\\\\``.  ``groups`` inserts a ``\\midrule`` before each listed row body (by
    exact match) to separate blocks.
    """
    lines = [r"\begin{tabular}{%s}" % colspec, r"\toprule", header + r" \\", r"\midrule"]
    for row in rows:
        if groups and row in groups:
            lines.append(r"\midrule")
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def save_table(name: str, body: str) -> None:
    """Write the table three ways, like the figure pipeline plus the LaTeX source.

    ``tables/tex/<name>.tex`` -- booktabs fragment the paper ``\\input``s;
    ``tables/pdf/<name>.pdf`` -- standalone compile (vector, for inclusion);
    ``tables/png/<name>.png`` -- 300 dpi raster of that PDF, for preview.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    for sub in ("tex", "pdf", "png"):
        (TABLES / sub).mkdir(parents=True, exist_ok=True)
    tex = TABLES / "tex" / f"{name}.tex"
    tex.write_text(body.rstrip() + "\n", encoding="utf-8")
    pdf = TABLES / "pdf" / f"{name}.pdf"
    png = TABLES / "png" / f"{name}.png"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / f"{name}.tex").write_text(_STANDALONE % body, encoding="utf-8")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
            # Fixed dates and IDs, so an unchanged table rebuilds to identical bytes.
            env={**os.environ, "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"},
        )
        built = tmp / f"{name}.pdf"
        if not built.exists():
            log = tmp / f"{name}.log"
            tail = (log.read_text(encoding="utf-8", errors="replace")[-1600:]
                    if log.exists() else proc.stdout[-1600:])
            raise SystemExit(f"pdflatex failed for {name}:\n{tail}")
        shutil.copy(built, pdf)

    _rasterize(pdf, png)
    outputs = f"{tex.relative_to(ROOT)}, {pdf.relative_to(ROOT)}"
    if png.exists():
        outputs += f" and {png.relative_to(ROOT)}"
    print(f"wrote {outputs}")


def _rasterize(pdf: Path, png: Path) -> None:
    """Render a single-page PDF to ``png`` at 300 dpi with pdftoppm.

    No-op with a warning if pdftoppm is not installed, so the .tex/.pdf still land.
    """
    if shutil.which("pdftoppm") is None:
        print(f"  (no pdftoppm; skipped {png.name})")
        return
    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(png.with_suffix(""))],
        check=True,
        capture_output=True,
    )
