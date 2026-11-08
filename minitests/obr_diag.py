# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic protocol: OBR's error budget and the early/late count lever.

Subcommands, run in order:

``collect``
    Rotated weights and input Hessians of six layers (``q/o/up/down`` of the
    middle block, ``down`` of block 1 and of the last block): the held-out C4
    Hessian ``eval`` (the first ``--holdout`` chunks of the validation stream,
    built from documents ``< C4_HELDOUT_DOCS``) and one calibration Hessian per
    ``--seeds`` value, drawn from the later validation documents so the two
    never share text.  Cached under ``--cache``.
``run``
    The tests in ``--tests``, each written to ``results/diag/<test>.json``:

    t0     error budget ``A`` (pruning compensation), ``R_E`` (early rounding
           after its compensation onto the late columns), ``R_L`` (late
           rounding) against the realized total
    t1     1a dither, 1b energy model, 1c low-rank regime
    t2     schedule head-to-head on the magnitude mask
    t3     late/early count sweep against the slot model
``report``
    Compact tables and pass/fail flags of every test written so far.

Errors are ``E||dW x||^2 / 2`` per row, summed over rows, under the undamped
held-out Hessian; the damped calibration Hessian is used only inside the
algorithms.  OBR comes in two forms that share pruning compensation and the
early-rounding compensation: ``rtn`` rounds the late columns (the minitests'
OBR, ``method=rtn``) and ``gptq`` runs sequential GPTQ over them (the
pipeline default, ``method=gptq``).

    python minitests/obr_diag.py collect --cache /path/to/cache
    python minitests/obr_diag.py run --cache /path/to/cache --tests t0,t1,t2
    python minitests/obr_diag.py report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import common as C
import datasets
import torch

from qdrift.obr.obr import PruningCompensationState, compensate_quantization_error
from qdrift.prune.mask import mask_from_score
from qdrift.quant.gptq import _hessian_inverse, gptq_quantize
from qdrift.quant.quantize import dequantize, fake_quantize, qrange

STEM = "obr_diag"
OUT = C.ROOT / "results" / "diag"
GRID = "midrise"
C4_VALIDATION = "en/c4-validation.00000-of-00008.json.gz"
# get_c4(eval_mode=True) builds its stream from these leading documents.
C4_HELDOUT_DOCS = 1100
# Relative jitter that makes the undamped held-out Hessian factorizable.
JITTER = 1e-6
FRACTIONS = (0.25, 0.2, 0.15, 0.1, 0.05)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("collect", "run", "report"))
    C.common_arguments(parser)
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    parser.add_argument("--middle", type=int, default=15)
    parser.add_argument("--layers", type=C.names, default=[], help="subset, e.g. 15.q")
    parser.add_argument("--seeds", type=C.ints, default=C.ints("0,1,2"))
    parser.add_argument("--holdout", type=int, default=64)
    parser.add_argument("--tests", type=C.names, default=["t0", "t1", "t2"])
    parser.add_argument("--bits", type=C.ints, default=C.ints("3,4"))
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--model-rows", type=int, default=64, help="t1b/t1c rows")
    parser.add_argument("--barrier-rows", type=int, default=8)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--late", type=C.floats, default=C.floats("0.55,0.6,0.65"))
    parser.set_defaults(nsamples=128, seqlen=2048)
    return parser.parse_args()


def layer_list(depth: int, middle: int) -> list[tuple[int, str]]:
    """The six diagnostic layers."""
    return [
        (middle, "q"),
        (middle, "o"),
        (middle, "up"),
        (middle, "down"),
        (1, "down"),
        (depth - 1, "down"),
    ]


def tag(index: int, layer_type: str) -> str:
    """Short layer tag, e.g. ``15.down``."""
    return f"{index}.{layer_type}"


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------
def c4_calibration(
    tokenizer, nsamples: int, seqlen: int, seed: int
) -> list[torch.Tensor]:
    """``get_c4`` calibration sampling over the non-held-out validation docs.

    The C4 train shard is not available offline; the validation shard minus
    the documents of the held-out stream is a disjoint stand-in.
    """
    data = datasets.load_dataset(
        "allenai/c4", data_files={"validation": C4_VALIDATION}, split="validation"
    )
    generator = random.Random(seed)
    chunks: list[torch.Tensor] = []
    while len(chunks) < nsamples:
        item = data[generator.randint(C4_HELDOUT_DOCS, len(data) - 1)]["text"]
        ids = tokenizer(item).input_ids
        if len(ids) < seqlen:
            continue
        start = generator.randint(0, len(ids) - seqlen)
        chunks.append(torch.tensor(ids[start : start + seqlen], dtype=torch.long))
    return chunks


@torch.no_grad()
def collect(args: argparse.Namespace) -> None:
    """Cache the weights and Hessians of the diagnostic layers."""
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    layers = layer_list(streamer.depth, args.middle)
    blocks = sorted({index for index, _ in layers})
    streamer.fit_resident(range(streamer.depth))
    streams = {}
    held = C.get_loaders(
        "c4", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
    )
    streams["eval"] = streamer.start(C.batch_chunks(held, args.batch_size))
    for seed in args.seeds:
        chunks = c4_calibration(tokenizer, args.nsamples, args.seqlen, seed)
        streams[f"s{seed}"] = streamer.start(C.batch_chunks(chunks, args.batch_size))
    C.log(f"streams {[(k, len(v[0])) for k, v in streams.items()]}")
    args.cache.mkdir(parents=True, exist_ok=True)
    for index in range(max(blocks) + 1):
        if index not in blocks:
            for key, (hidden, kwargs) in streams.items():
                streams[key] = (streamer.run(index, hidden, kwargs), kwargs)
            continue
        types = [t for i, t in layers if i == index]
        sources = sorted({C.INPUT_SOURCE[t] for t in types})
        grams = {}
        for key, (hidden, kwargs) in streams.items():
            grams[key], hidden = C.block_grams(streamer, index, hidden, kwargs, sources)
            streams[key] = (hidden, kwargs)
        for layer_type in types:
            source = C.INPUT_SOURCE[layer_type]
            entry = {
                "weight": C.block_module(model, index, layer_type).weight.data.float(),
                "count": {key: grams[key][source].count for key in grams},
            }
            for key in grams:
                entry[key] = grams[key][source].hessians(args.damping)[0].cpu()
            torch.save(entry, args.cache / f"{tag(index, layer_type)}.pt")
            C.log(f"  cached {tag(index, layer_type)}")
        del grams
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# OBR stages
# --------------------------------------------------------------------------
def energy(delta: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Per-row ``delta^T H delta / 2``."""
    return 0.5 * (delta @ hessian).mul_(delta).sum(1)


def dense(weight: torch.Tensor, keep: torch.Tensor, values: torch.Tensor):
    """Scatter compact retained values into a dense zero matrix."""
    out = torch.zeros_like(weight)
    out.scatter_(1, keep, values)
    return out


def with_alpha(
    state: PruningCompensationState, cols: int, alpha: float
) -> PruningCompensationState:
    """The same pruning state with the early/late split moved to ``alpha``."""
    cut = torch.full(
        (state.keep.shape[0], 1),
        int(cols * alpha),
        dtype=state.keep.dtype,
        device=state.keep.device,
    )
    return replace(state, boundary=torch.searchsorted(state.keep, cut).squeeze(1))


def gptq_trace(
    work: torch.Tensor,
    system: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
    cut: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``qdrift.quant.gptq.gptq_quantize`` from column ``cut`` on, dequantized.

    Columns before ``cut`` must already be rounded.  Also returns every
    column's value right before it is rounded.
    """
    work = work.clone()
    pre = work.clone()
    hinv = _hessian_inverse(system, 0.0)
    cols = work.shape[1]
    for col in range(cut, cols):
        column = work[:, col]
        pre[:, col] = column
        quantized = fake_quantize(
            column.unsqueeze(1), scale[:, col : col + 1], qparams, grid
        ).squeeze(1)
        target = torch.where(mask[:, col], quantized, torch.zeros_like(quantized))
        error = column - target
        work[:, col] = target
        if col + 1 < cols:
            work[:, col + 1 :] -= torch.outer(
                error / hinv[col, col].clamp_min(1e-12), hinv[col, col + 1 :]
            )
    return work, pre


def obr_stages(
    weight, mask, state, system, scale, bits, alpha, late, grid=GRID
) -> dict[str, torch.Tensor]:
    """Dense OBR iterates.

    ``v1`` pruning-compensated, ``v2`` early columns rounded and their error
    compensated onto the late columns, ``v3`` final; ``pre`` the value of each
    weight right before it is rounded.
    """
    qparams = qrange(bits, grid)
    cols = weight.shape[1]
    cut = int(cols * alpha)
    state = with_alpha(state, cols, alpha)
    v1 = dense(weight, state.keep, state.retained)
    kept_scale = scale.gather(1, state.keep)
    values = compensate_quantization_error(
        weight, state, kept_scale, qparams, grid, alpha=alpha
    )
    v2 = dense(weight, state.keep, values)
    pre = v2.clone()
    v2[:, :cut] = fake_quantize(v2[:, :cut], scale[:, :cut], qparams, grid)
    v2[:, :cut] *= mask[:, :cut]
    if late == "rtn":
        v3 = fake_quantize(v2, scale, qparams, grid) * mask
    else:
        v3, late_pre = gptq_trace(v2, system, mask, scale, qparams, grid, cut)
        pre[:, cut:] = late_pre[:, cut:]
    return {"v1": v1, "v2": v2, "v3": v3, "pre": pre}


def sequential(weight, mask, system, scale, bits, grid=GRID) -> torch.Tensor:
    """GPTQ over every column of ``weight`` with the mask forced (no split)."""
    qparams = qrange(bits, grid)
    return gptq_trace(weight * 1.0, system, mask, scale, qparams, grid, 0)[0]


def magnitude_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-row unstructured magnitude mask (True = keep)."""
    return mask_from_score(weight.abs(), sparsity, "unstructured")


def split_mask_counts(weight: torch.Tensor, late_sparsity: float) -> torch.Tensor:
    """Magnitude mask per half; the late half prunes ``late_sparsity``.

    The early half prunes the rest, so every row stays at 50%.
    """
    rows, cols = weight.shape
    half = cols // 2
    late_pruned = round(late_sparsity * (cols - half))
    early_pruned = cols // 2 - late_pruned
    mask = torch.ones_like(weight, dtype=torch.bool)
    for start, end, count in ((0, half, early_pruned), (half, cols, late_pruned)):
        if count <= 0:
            continue
        index = weight[:, start:end].abs().topk(count, dim=1, largest=False).indices
        mask[:, start:end].scatter_(1, index, False)
    return mask


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def geometry(entry: dict, seed: int, args, device) -> dict:
    """Weights, Hessians, scales, and the magnitude mask of one layer/seed."""
    weight = entry["weight"].to(device)
    rows = C.row_subset(weight.shape[0], args.rows, 0, device)
    if rows is not None:
        weight = weight[rows]
    hessian = entry[f"s{seed}"].to(device)
    system = C.damped(hessian, args.damping)
    evalh = entry["eval"].to(device)
    lam = args.damping * torch.diagonal(hessian).clamp_min(0).mean()
    return {
        "weight": weight,
        "hessian": hessian,
        "system": system,
        "eval": evalh,
        "lam": float(lam),
        "mask": magnitude_mask(weight, args.sparsity),
    }


def scale_of(weight, bits, args, grid=GRID) -> torch.Tensor:
    """Dense-weight scale exactly as the pipeline picks it."""
    clip = args.range == "clip"
    return C.weight_scale(weight, bits, grid, args.group_size, clip).scale


def budget(weight, stages, evalh) -> dict[str, torch.Tensor]:
    """Per-row A, R_E, R_L, and total."""
    return {
        "A": energy(stages["v1"] - weight, evalh),
        "R_E": energy(stages["v2"] - stages["v1"], evalh),
        "R_L": energy(stages["v3"] - stages["v2"], evalh),
        "total": energy(stages["v3"] - weight, evalh),
    }


def t0_row(head: dict, parts: dict[str, torch.Tensor]) -> dict:
    """Summed budget and shares of one layer/bits/seed/variant."""
    total = float(parts["total"].sum())
    sums = {key: float(value.sum()) for key, value in parts.items()}
    late_share = (parts["R_L"] / parts["total"].clamp_min(1e-30)).median()
    return {
        **head,
        **sums,
        "A_share": sums["A"] / total,
        "R_E_share": sums["R_E"] / total,
        "R_L_share": sums["R_L"] / total,
        "R_L_share_row_median": float(late_share),
        "additivity_error": abs(sums["A"] + sums["R_E"] + sums["R_L"] - total) / total,
    }


def dither_rows(head, weight, mask, stages, scale, bits, cut, group_size) -> list:
    """1a: does the pre-rounding value forget the residual of ``w``?"""
    qparams = qrange(bits, GRID)
    eps_w = fake_quantize(weight, scale, qparams, GRID) - weight
    pre = stages["pre"]
    eps_u = fake_quantize(pre, scale, qparams, GRID) - pre
    cols = weight.shape[1]
    columns = torch.arange(cols, device=weight.device)
    rows = []
    for half, sel_cols in (("E", columns < cut), ("L", columns >= cut)):
        sel = mask & sel_cols.unsqueeze(0)
        a = eps_w[sel].pow(2)
        b = eps_u[sel].pow(2)
        corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        size = group_size if group_size > 0 else cols
        groups = cols // size
        shape = (weight.shape[0], groups, size)
        diff = ((pre - weight) * sel).reshape(shape)
        count = sel.reshape(shape).sum(2)
        mean = diff.sum(2) / count.clamp_min(1)
        var = (diff.pow(2).sum(2) / count.clamp_min(1) - mean.pow(2)).clamp_min(0)
        step = scale.reshape(shape)[:, :, 0]
        ratio = (var.sqrt() / step)[count >= 8]
        rows.append(
            {
                **head,
                "half": half,
                "corr_eps2": corr,
                "s_over_step_median": float(ratio.median()),
                "s_over_step_mean": float(ratio.mean()),
                "pass": corr < 0.1 and float(ratio.median()) > 0.5,
            }
        )
    return rows


def factor(matrix: torch.Tensor) -> torch.Tensor:
    """fp64 Cholesky with the relative jitter, escalated if needed."""
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.float64)
    mean = torch.diagonal(matrix).mean().double()
    for jitter in (JITTER, 1e-4, 1e-2):
        chol, info = torch.linalg.cholesky_ex(matrix.double() + jitter * mean * eye)
        if int(info) == 0:
            return chol
    raise RuntimeError("Hessian block is not factorizable")


def schur_diag(evalh, late, early) -> torch.Tensor:
    """``diag(H_EE - H_EL H_LL^-1 H_LE)`` via one Cholesky of ``[L, E]``."""
    order = torch.cat([late, early])
    chol = factor(evalh[order][:, order])
    return chol[len(late) :, len(late) :].pow(2).sum(1)


def energy_model(geo, args) -> list[dict]:
    """1b inputs that do not depend on bits: per-row conditional variances."""
    weight, mask, evalh, system = geo["weight"], geo["mask"], geo["eval"], geo["system"]
    cut = int(weight.shape[1] * 0.5)
    out = []
    for row in range(min(args.model_rows, weight.shape[0])):
        keep = torch.nonzero(mask[row]).squeeze(1)
        early, late = keep[keep < cut], keep[keep >= cut]
        sigma = schur_diag(evalh, late, early)
        # What OBR's compensation (solved with H_sys) leaves under H_eval.
        gain = torch.linalg.solve(
            system[late][:, late].double(), system[late][:, early].double()
        )
        h_ll = evalh[late][:, late].double()
        h_le = evalh[late][:, early].double()
        sigma_sys = (
            torch.diagonal(evalh[early][:, early]).double()
            - 2 * (h_le * gain).sum(0)
            + (gain * (h_ll @ gain)).sum(0)
        )
        reverse = late.flip(0)
        cond = torch.diagonal(factor(evalh[reverse][:, reverse])).pow(2).flip(0)
        out.append(
            {
                "early": early,
                "late": late,
                "sigma": sigma,
                "sigma_sys": sigma_sys,
                "late_diag": torch.diagonal(evalh)[late].double(),
                "late_cond": cond,
            }
        )
    return out


def energy_rows(head, model, scale, parts_by_variant) -> list[dict]:
    """1b: measured over predicted R_E and R_L on the model rows."""
    pred_e = pred_e_sys = pred_rtn = pred_gptq = 0.0
    for row, item in enumerate(model):
        step2 = scale[row].double().pow(2) / 12.0
        pred_e += 0.5 * float((step2[item["early"]] * item["sigma"]).sum())
        pred_e_sys += 0.5 * float((step2[item["early"]] * item["sigma_sys"]).sum())
        pred_rtn += 0.5 * float((step2[item["late"]] * item["late_diag"]).sum())
        pred_gptq += 0.5 * float((step2[item["late"]] * item["late_cond"]).sum())
    count = len(model)
    rows = []
    for variant, parts in parts_by_variant.items():
        meas_e = float(parts["R_E"][:count].sum())
        meas_l = float(parts["R_L"][:count].sum())
        pred_l = pred_rtn if variant == "rtn" else pred_gptq
        ratio_e, ratio_l = meas_e / pred_e, meas_l / pred_l
        rows.append(
            {
                **head,
                "variant": variant,
                "R_E_meas": meas_e,
                "R_E_pred": pred_e,
                "R_E_pred_sys": pred_e_sys,
                "R_L_meas": meas_l,
                "R_L_pred": pred_l,
                "R_L_pred_rtn": pred_rtn,
                "ratio_E": ratio_e,
                "ratio_E_sys": meas_e / pred_e_sys,
                "ratio_L": ratio_l,
                "pass": 0.8 <= ratio_e <= 1.25 and 0.8 <= ratio_l <= 1.25,
            }
        )
    return rows


def lowrank_row(head, geo, model, args) -> dict:
    """1c: effective rank, rho_A, and the barrier curve of one layer."""
    hessian, evalh, weight = geo["hessian"], geo["eval"], geo["weight"]
    lam = geo["lam"]
    eig = torch.linalg.eigvalsh(hessian.double())
    eig_eval = torch.linalg.eigvalsh(evalh.double())
    d = hessian.shape[0]
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(geo["system"].double()))
    rho = []
    for row in range(min(args.model_rows, weight.shape[0])):
        pruned = torch.nonzero(~geo["mask"][row]).squeeze(1)
        w_p = weight[row, pruned].double()
        loss = w_p @ torch.linalg.solve(hinv[pruned][:, pruned], w_p)
        norm = w_p.pow(2).sum()
        rho.append(float((loss - lam * norm) / (lam * norm)))
    generator = torch.Generator().manual_seed(0)
    curve = {f: [] for f in FRACTIONS}
    for item in model[: args.barrier_rows]:
        early, late = item["early"], item["late"]
        base = float(torch.diagonal(evalh)[early].double().sum())
        for fraction in FRACTIONS:
            size = min(round(fraction * d), len(late))
            draws = 1 if size == len(late) else args.draws
            for _ in range(draws):
                pick = torch.randperm(len(late), generator=generator)[:size]
                sub = late[pick.to(late.device)].sort().values
                curve[fraction].append(
                    float(schur_diag(evalh, sub, early).sum()) / base
                )
    barrier = {f"{f:g}": sum(v) / len(v) for f, v in curve.items() if v}
    doubling = None
    for fraction in FRACTIONS if barrier else ():
        if barrier[f"{fraction:g}"] >= 2 * barrier[f"{FRACTIONS[0]:g}"]:
            doubling = fraction
            break
    tail = eig_eval[eig_eval < 2 * lam]
    rho_median = sorted(rho)[len(rho) // 2]
    return {
        **head,
        "lambda": lam,
        "r_over_d": float((eig > 2 * lam).sum()) / d,
        "r_over_d_eval": float((eig_eval > 2 * lam).sum()) / d,
        "lambda_tail_eval": float(tail.mean()) if tail.numel() else 0.0,
        "h_bar_eval": float(torch.diagonal(evalh).mean()),
        "rho_A_median": rho_median,
        "rho_A_p90": sorted(rho)[int(0.9 * (len(rho) - 1))],
        "barrier": barrier,
        "late_over_d": sum(len(m["late"]) for m in model) / max(len(model), 1) / d,
        "doubling_fraction": doubling,
        "pass": float((eig > 2 * lam).sum()) / d < 0.25 and rho_median < 0.1,
    }


def slot_prediction(weight, scale, geo, lowrank) -> dict:
    """T3 model: b, g, m*, and the predicted gain per row."""
    rows, cols = weight.shape
    half = cols // 2
    lam = geo["lam"]
    w2 = weight.pow(2)
    threshold = w2.kthvalue(int(cols * 0.5), dim=1).values
    densities = []
    for part in (w2[:, :half], w2[:, half:]):
        ordered = part.sort(dim=1).values
        pos = torch.searchsorted(ordered, threshold.unsqueeze(1)).squeeze(1)
        k = max(8, part.shape[1] // 50)
        lo = (pos - k).clamp(0, part.shape[1] - 1)
        hi = (pos + k).clamp(0, part.shape[1] - 1)
        span = ordered.gather(1, hi.unsqueeze(1)) - ordered.gather(1, lo.unsqueeze(1))
        densities.append((hi - lo).float() / span.squeeze(1).clamp_min(1e-30))
    g_e, g_l = densities
    g = 2.0 / (1.0 / g_e + 1.0 / g_l)
    b = (
        scale.pow(2).mean(1)
        / 12.0
        * (lowrank["h_bar_eval"] - lowrank["lambda_tail_eval"])
    )
    slots = b * g / (2 * lam)
    gain = b.pow(2) * g / (4 * lam)
    return {"b": b, "g_E": g_e, "g_L": g_l, "g": g, "slots": slots, "gain": gain}


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    """Run the selected tests on the cached layers."""
    device = torch.device(args.device)
    tables: dict[str, list[dict]] = {test: [] for test in args.tests}
    entries = sorted(args.cache.glob("*.pt"))
    wanted = set(args.layers)
    for path in entries:
        layer = path.stem
        if wanted and layer not in wanted:
            continue
        entry = torch.load(path, map_location="cpu")
        lowrank = None
        for seed in args.seeds:
            start = time.time()
            geo = geometry(entry, seed, args, device)
            weight, mask, system, evalh = (
                geo["weight"],
                geo["mask"],
                geo["system"],
                geo["eval"],
            )
            state = C.pruning_state(weight, mask, system, 0.5)
            model = energy_model(geo, args) if "t1" in args.tests else None
            if "t1" in args.tests and seed == args.seeds[0]:
                lowrank = lowrank_row({"layer": layer, "seed": seed}, geo, model, args)
                tables["t1"].append({"test": "1c", **lowrank})
            if "t3" in args.tests and lowrank is None:
                lowrank = lowrank_row({"layer": layer, "seed": seed}, geo, [], args)
            for bits in args.bits:
                scale = scale_of(weight, bits, args)
                head = {"layer": layer, "bits": bits, "seed": seed}
                parts = {}
                for late in ("rtn", "gptq"):
                    stages = obr_stages(
                        weight, mask, state, system, scale, bits, 0.5, late
                    )
                    parts[late] = budget(weight, stages, evalh)
                    if late == "gptq" and path == entries[0] and seed == args.seeds[0]:
                        codes = gptq_quantize(
                            dense(
                                weight,
                                state.keep,
                                compensate_quantization_error(
                                    weight,
                                    with_alpha(state, weight.shape[1], 0.5),
                                    scale.gather(1, state.keep),
                                    qrange(bits, GRID),
                                    GRID,
                                ),
                            ),
                            system,
                            mask,
                            scale,
                            qrange(bits, GRID),
                            GRID,
                            0.5,
                            0.0,
                        )
                        ref = dequantize(codes, scale, qrange(bits, GRID)) * mask
                        C.log(
                            f"  gptq trace vs pipeline: max diff {float((ref - stages['v3']).abs().max()):.3g}"
                        )
                    if "t0" in args.tests:
                        tables["t0"].append(
                            t0_row({**head, "variant": late}, parts[late])
                        )
                    if "t1" in args.tests:
                        tables["t1"] += [
                            {"test": "1a", "variant": late, **row}
                            for row in dither_rows(
                                head,
                                weight,
                                mask,
                                stages,
                                scale,
                                bits,
                                weight.shape[1] // 2,
                                args.group_size,
                            )
                        ]
                    del stages
                if "t1" in args.tests:
                    tables["t1"] += [
                        {"test": "1b", **row}
                        for row in energy_rows(head, model, scale, parts)
                    ]
                if "t2" in args.tests:
                    arms = {
                        "o": parts["gptq"]["total"],
                        "o_rtn": parts["rtn"]["total"],
                    }
                    for alpha in (0.75, 0.9):
                        for late in ("gptq", "rtn"):
                            name = f"a{round(alpha * 100)}" + (
                                "" if late == "gptq" else "_rtn"
                            )
                            stages = obr_stages(
                                weight, mask, state, system, scale, bits, alpha, late
                            )
                            arms[name] = energy(stages["v3"] - weight, evalh)
                            del stages
                    v1 = dense(weight, state.keep, state.retained)
                    arms["s_obr"] = energy(
                        sequential(v1, mask, system, scale, bits) - weight, evalh
                    )
                    arms["s_joint"] = energy(
                        sequential(weight, mask, system, scale, bits) - weight, evalh
                    )
                    for name, value in arms.items():
                        tables["t2"].append(
                            {**head, "arm": name, "total": float(value.sum())}
                        )
                if "t3" in args.tests:
                    prediction = slot_prediction(weight, scale, geo, lowrank)
                    totals = {late: parts[late]["total"] for late in parts}
                    base_two = 2 * totals["rtn"]
                    tables["t3"].append(
                        {
                            **head,
                            "kind": "prediction",
                            "b": float(prediction["b"].mean()),
                            "g_E": float(prediction["g_E"].median()),
                            "g_L": float(prediction["g_L"].median()),
                            "slots_mean": float(prediction["slots"].mean()),
                            "slots_over_half": float(prediction["slots"].mean())
                            / (weight.shape[1] / 2),
                            "pred_late_sparsity": 0.5
                            + float(prediction["slots"].mean()) / (weight.shape[1] / 2),
                            "pred_gain_rel": float(
                                prediction["gain"].sum() / base_two.sum()
                            ),
                        }
                    )
                    for late_sparsity in [0.5, *args.late]:
                        swept = split_mask_counts(weight, late_sparsity)
                        swept_state = C.pruning_state(weight, swept, system, 0.5)
                        for late in ("rtn", "gptq"):
                            stages = obr_stages(
                                weight,
                                swept,
                                swept_state,
                                system,
                                scale,
                                bits,
                                0.5,
                                late,
                            )
                            total = float(energy(stages["v3"] - weight, evalh).sum())
                            tables["t3"].append(
                                {
                                    **head,
                                    "kind": "sweep",
                                    "variant": late,
                                    "late_sparsity": late_sparsity,
                                    "total": total,
                                    "ratio_vs_magnitude": total
                                    / float(totals[late].sum()),
                                }
                            )
                            del stages
                        del swept_state
                torch.cuda.empty_cache()
            del state, model, geo
            torch.cuda.empty_cache()
            for test, rows in tables.items():
                C.save_result(
                    OUT / f"{test}{args.suffix}.json", STEM, vars_json(args), rows=rows
                )
            C.log(f"  {layer} seed {seed} done in {time.time() - start:.0f}s")


def vars_json(args: argparse.Namespace) -> dict:
    """JSON-safe config."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load_rows(test: str) -> list[dict]:
    """All rows of one test, merged over shard files."""
    rows = []
    for path in sorted(OUT.glob(f"{test}*.json")):
        rows += json.loads(path.read_text())["rows"]
    return rows


def geomean(values: list[float]) -> float:
    """Geometric mean."""
    return math.exp(sum(math.log(v) for v in values) / len(values))


def report(args: argparse.Namespace) -> None:
    """Print compact tables of every test written so far."""
    rows = load_rows("t0")
    if rows:
        print("\nT0 error budget (shares of total, summed over rows; mean over seeds)")
        print(
            f"{'layer':9} {'W':>2} {'var':5} {'A':>6} {'R_E':>6} {'R_L':>6} {'R_L~':>6} {'add':>6}"
        )
        keys = sorted({(r["layer"], r["bits"], r["variant"]) for r in rows})
        late = []
        for key in keys:
            sel = [r for r in rows if (r["layer"], r["bits"], r["variant"]) == key]
            mean = {
                k: sum(r[k] for r in sel) / len(sel)
                for k in (
                    "A_share",
                    "R_E_share",
                    "R_L_share",
                    "R_L_share_row_median",
                    "additivity_error",
                )
            }
            late.append((key[2], mean["R_L_share"]))
            print(
                f"{key[0]:9} {key[1]:>2} {key[2]:5} {mean['A_share']:6.3f} {mean['R_E_share']:6.3f} "
                f"{mean['R_L_share']:6.3f} {mean['R_L_share_row_median']:6.3f} {mean['additivity_error']:6.3f}"
            )
        for variant in ("rtn", "gptq"):
            shares = sorted(v for k, v in late if k == variant)
            if shares:
                print(
                    f"median late share ({variant}): {shares[len(shares) // 2]:.3f}"
                    "  (< 0.075 => count lever capped)"
                )
    rows = load_rows("t1")
    if rows:
        print("\nT1a dither (mean over seeds)")
        sel = [r for r in rows if r["test"] == "1a"]
        for key in sorted(
            {(r["layer"], r["bits"], r["variant"], r["half"]) for r in sel}
        ):
            group = [
                r
                for r in sel
                if (r["layer"], r["bits"], r["variant"], r["half"]) == key
            ]
            corr = sum(r["corr_eps2"] for r in group) / len(group)
            ratio = sum(r["s_over_step_median"] for r in group) / len(group)
            print(
                f"  {key[0]:9} W{key[1]} {key[2]:4} {key[3]}  corr {corr:6.3f}  s/D {ratio:6.3f}  "
                f"{'PASS' if corr < 0.1 and ratio > 0.5 else 'FAIL'}"
            )
        print("\nT1b energy model (measured/predicted, mean over seeds)")
        sel = [r for r in rows if r["test"] == "1b"]
        for key in sorted({(r["layer"], r["bits"], r["variant"]) for r in sel}):
            group = [r for r in sel if (r["layer"], r["bits"], r["variant"]) == key]
            e = sum(r["ratio_E"] for r in group) / len(group)
            es = sum(r["ratio_E_sys"] for r in group) / len(group)
            lo = sum(r["ratio_L"] for r in group) / len(group)
            ok = 0.8 <= e <= 1.25 and 0.8 <= lo <= 1.25
            print(
                f"  {key[0]:9} W{key[1]} {key[2]:4}  R_E {e:6.3f} (sys {es:6.3f})  R_L {lo:6.3f}  "
                f"{'PASS' if ok else 'FAIL'}"
            )
        print("\nT1c low-rank regime")
        for r in [r for r in rows if r["test"] == "1c"]:
            curve = " ".join(f"{v:.2f}" for v in r["barrier"].values())
            print(
                f"  {r['layer']:9} r/d {r['r_over_d']:.3f} (eval {r['r_over_d_eval']:.3f})  "
                f"rho_A {r['rho_A_median']:.3f}  |L|/d {r['late_over_d']:.3f}  "
                f"barrier[{curve}] doubles at {r['doubling_fraction']}  {'PASS' if r['pass'] else 'FAIL'}"
            )
    rows = load_rows("t2")
    if rows:
        print("\nT2 schedules: geomean ratio vs o over layers, mean +- std over seeds")
        arms = list(dict.fromkeys(r["arm"] for r in rows))
        for bits in sorted({r["bits"] for r in rows}):
            for arm in arms:
                per_seed = []
                for seed in sorted({r["seed"] for r in rows}):
                    sel = {
                        (r["layer"]): r["total"]
                        for r in rows
                        if r["bits"] == bits and r["seed"] == seed and r["arm"] == arm
                    }
                    base = {
                        (r["layer"]): r["total"]
                        for r in rows
                        if r["bits"] == bits and r["seed"] == seed and r["arm"] == "o"
                    }
                    if sel:
                        per_seed.append(geomean([sel[k] / base[k] for k in sel]))
                mean = sum(per_seed) / len(per_seed)
                std = (
                    sum((v - mean) ** 2 for v in per_seed) / max(len(per_seed) - 1, 1)
                ) ** 0.5
                print(f"  W{bits} {arm:9} {mean:6.3f} +- {std:.3f}")
    rows = load_rows("t3")
    if rows:
        print(
            "\nT3 count sweep (ratio vs magnitude 50/50, geomean over layers and seeds)"
        )
        pred = [r for r in rows if r["kind"] == "prediction"]
        for bits in sorted({r["bits"] for r in pred}):
            sel = [r for r in pred if r["bits"] == bits]
            print(
                f"  W{bits} predicted late sparsity {sum(r['pred_late_sparsity'] for r in sel) / len(sel):.3f}, "
                f"gain {sum(r['pred_gain_rel'] for r in sel) / len(sel):.4f}"
            )
        sweep = [r for r in rows if r["kind"] == "sweep"]
        for key in sorted(
            {(r["bits"], r["variant"], r["late_sparsity"]) for r in sweep}
        ):
            sel = [
                r["ratio_vs_magnitude"]
                for r in sweep
                if (r["bits"], r["variant"], r["late_sparsity"]) == key
            ]
            print(f"  W{key[0]} {key[1]:4} late {key[2]:.2f}: {geomean(sel):.4f}")


def main() -> None:
    """Dispatch the subcommand."""
    args = parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
