# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: pruning keeps each group's absmax, so it cannot move the scales.

Weight statistics over every linear layer of the (rotated) model, using the
dense-path input norms ``||x_j||^2`` for WANDA / Q-WANDA:

``absmax``
    For each quantization group (g64, g128, per channel), is the group's
    largest-magnitude weight retained by the mask?  If it always is, the
    absmax-based scale of the pruned weights equals the dense scale, i.e.
    pruning neither enlarges nor reduces the quantization range.

``exchange``
    Fraction of weights whose quantization error exceeds their own magnitude
    (``eps^2 >= w^2``: cheaper to prune than to quantize), over all weights and
    over the weights a WANDA mask retains, plus the share of the retained
    activation-weighted quantization loss those weights carry.

``overlap``
    How far Q-WANDA moves the mask away from WANDA as lambda decreases, and the
    diagonal proxy of the joint loss ``sum_P w^2 x^2 + sum_R eps^2 x^2``.

    python minitests/prune_scale_invariance.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.prune.mask import mask_from_score

STEM = "prune_scale_invariance"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=list(range(32)))
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument(
        "--group-sizes",
        type=C.ints,
        default=C.ints("64,128,0"),
        help="quantization groups to test; 0 = per channel",
    )
    parser.add_argument(
        "--sparsities",
        type=C.floats,
        default=C.floats("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument("--bits", type=C.ints, default=C.ints("1,2,3,4"))
    parser.add_argument("--grids", type=C.names, default=["midrise", "midtread"])
    parser.add_argument(
        "--lambdas",
        type=C.floats,
        default=C.floats("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"),
    )
    parser.add_argument(
        "--qwanda-lambda",
        type=float,
        default=0.7,
        help="lambda of the Q-WANDA mask in the absmax table",
    )
    parser.add_argument(
        "--qwanda-bits",
        type=int,
        default=2,
        help="mid-rise bit width whose error drives that Q-WANDA mask",
    )
    parser.add_argument(
        "--exchange-sparsities", type=C.floats, default=C.floats("0.3,0.5,0.7")
    )
    parser.add_argument("--overlap-sparsity", type=float, default=0.5)
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the sweep."""
    setup = f"rot-{args.rotation}_{args.range}"
    sweep = "_".join(
        [
            C.span(args.blocks, "blk"),
            "+".join(C.gran_tag(g) for g in args.group_sizes),
            C.span(args.sparsities, "s"),
            C.span(args.bits, "w"),
        ]
    )
    variant = "+".join(args.grids) + f"_qw{args.qwanda_lambda:g}-w{args.qwanda_bits}"
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


def grouped(values: torch.Tensor, group_size: int) -> torch.Tensor:
    """[rows, cols] -> [rows, groups, size]; per channel is one group."""
    rows, cols = values.shape
    size = group_size if 0 < group_size < cols else cols
    if cols % size:
        raise ValueError(f"width {cols} is not a multiple of group size {size}")
    return values.reshape(rows, cols // size, size)


def absmax_rows(
    args: argparse.Namespace,
    weight: torch.Tensor,
    diag: torch.Tensor,
    group_size: int,
    eps: torch.Tensor,
    head: dict,
) -> list[dict]:
    """Absmax retention of one granularity for every criterion and sparsity."""
    rows = []
    groups = grouped(weight.abs(), group_size)
    argmax = groups.argmax(dim=2, keepdim=True)
    amax = groups.gather(2, argmax).squeeze(2).clamp_min(1e-12)
    criteria = {
        "wanda": C.score("wanda", weight, diag),
        "magnitude": C.score("magnitude", weight, diag),
        f"qwanda{args.qwanda_lambda:g}": C.score(
            "qwanda", weight, diag, eps, args.qwanda_lambda
        ),
    }
    for criterion, crit in criteria.items():
        for sparsity in args.sparsities:
            mask = grouped(mask_from_score(crit, sparsity, "unstructured"), group_size)
            kept = mask.gather(2, argmax).squeeze(2)
            shift = 1.0 - (groups * mask).amax(dim=2) / amax
            rows.append(
                {
                    **head,
                    "gran": C.gran_tag(group_size),
                    "criterion": criterion,
                    "sparsity": sparsity,
                    "absmax_kept": float(kept.float().mean()),
                    "amax_shift_mean": float(shift.mean()),
                    "amax_shift_max": float(shift.max()),
                }
            )
    return rows


def exchange_row(
    args: argparse.Namespace,
    weight: torch.Tensor,
    diag: torch.Tensor,
    masks: dict[float, torch.Tensor],
    key: tuple[int, str, int],
    eps: torch.Tensor,
    head: dict,
) -> dict:
    """Prune-vs-quantize exchange statistics of one quantizer."""
    group_size, grid, bits = key
    w2 = weight.pow(2)
    eps2 = eps.pow(2)
    # Strict: a mid-tread weight rounded to zero has eps^2 == w^2 and is
    # already pruned by the quantizer; it is counted separately.
    better = eps2 > w2
    weighted = eps2 * diag.reshape(1, -1)
    row = {
        **head,
        "gran": C.gran_tag(group_size),
        "grid": grid,
        "bits": bits,
        "better_pruned_all": float(better.float().mean()),
        "rounded_to_zero": float((eps == -weight).float().mean()),
        "quant_loss_over_prune_loss": float(weighted.sum() / (w2 * diag).sum()),
    }
    for sparsity, mask in masks.items():
        kept_better = better & mask
        row[f"better_pruned_kept_s{sparsity:g}"] = float(kept_better.sum() / mask.sum())
        row[f"loss_share_kept_s{sparsity:g}"] = float(
            (weighted * kept_better).sum() / (weighted * mask).sum().clamp_min(1e-30)
        )
    return row


def overlap_rows(
    args: argparse.Namespace,
    weight: torch.Tensor,
    diag: torch.Tensor,
    pruned_wanda: torch.Tensor,
    key: tuple[int, str, int],
    eps: torch.Tensor,
    head: dict,
) -> list[dict]:
    """Q-WANDA mask overlap with WANDA and the diagonal proxy, per lambda."""
    group_size, grid, bits = key
    rows = []
    w2x = weight.pow(2) * diag.reshape(1, -1)
    e2x = eps.pow(2) * diag.reshape(1, -1)
    for lam in args.lambdas:
        mask = mask_from_score(
            C.score("qwanda", weight, diag, eps, lam),
            args.overlap_sparsity,
            "unstructured",
        )
        pruned = ~mask
        rows.append(
            {
                **head,
                "gran": C.gran_tag(group_size),
                "grid": grid,
                "bits": bits,
                "lambda": lam,
                "sparsity": args.overlap_sparsity,
                "pruned_shared_with_wanda": float(
                    (pruned & pruned_wanda).sum() / pruned_wanda.sum()
                ),
                "proxy_prune": float((w2x * pruned).sum()),
                "proxy_quant": float((e2x * mask).sum()),
            }
        )
    return rows


def measure_layer(
    args: argparse.Namespace, weight: torch.Tensor, diag: torch.Tensor, head: dict
) -> dict[str, list[dict]]:
    """All three tables for one layer."""
    clip = args.range == "clip"
    tables: dict[str, list[dict]] = {"absmax": [], "exchange": [], "overlap": []}
    wanda = C.score("wanda", weight, diag)
    masks = {
        s: mask_from_score(wanda, s, "unstructured") for s in args.exchange_sparsities
    }
    pruned_wanda = ~mask_from_score(wanda, args.overlap_sparsity, "unstructured")
    del wanda
    for group_size in args.group_sizes:
        for grid in args.grids:
            for bits in args.bits:
                if grid == "midtread" and bits < 2:
                    continue
                key = (group_size, grid, bits)
                scale = C.weight_scale(weight, bits, grid, group_size, clip).scale
                eps = C.quant_error(weight, scale, bits, grid)
                del scale
                tables["exchange"].append(
                    exchange_row(args, weight, diag, masks, key, eps, head)
                )
                if group_size == args.group_size:
                    tables["overlap"] += overlap_rows(
                        args, weight, diag, pruned_wanda, key, eps, head
                    )
                if grid == "midrise" and bits == args.qwanda_bits:
                    tables["absmax"] += absmax_rows(
                        args, weight, diag, group_size, eps, head
                    )
                del eps
                torch.cuda.empty_cache()
    return tables


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    if "midrise" not in args.grids or args.qwanda_bits not in args.bits:
        raise SystemExit(
            "the absmax table needs --grids midrise and --qwanda-bits in --bits"
        )
    path = output_path(args)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(streamer.depth))
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    hidden, kwargs = streamer.start(calib)
    tables: dict[str, list[dict]] = {"absmax": [], "exchange": [], "overlap": []}
    sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
    for index in range(max(args.blocks) + 1):
        if index not in args.blocks:
            hidden = streamer.run(index, hidden, kwargs)
            continue
        start = time.time()
        grams, hidden = C.block_grams(streamer, index, hidden, kwargs, sources)
        diags = {
            source: torch.diagonal(gram.sum / gram.count).clamp_min(0)
            for source, gram in grams.items()
        }
        del grams
        for layer_type in args.types:
            module = C.block_module(model, index, layer_type)
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            head = {
                "block": index,
                "type": layer_type,
                "layer": C.layer_name(index, layer_type),
            }
            layer = measure_layer(args, weight, diags[C.INPUT_SOURCE[layer_type]], head)
            for name, rows in layer.items():
                tables[name] += rows
            del weight
            torch.cuda.empty_cache()
        C.save_result(path, STEM, vars(args), **tables)
        C.log(f"block {index} done in {time.time() - start:.0f}s")
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
