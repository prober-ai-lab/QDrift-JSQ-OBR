# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: why do magnitude, SparseGPT and QAP pick nearly the same mask?

Four per-layer diagnostics on the (rotated) model, ``g_j = 1 / [H^-1]_jj``:

``flat``
    Spread (coefficient of variation, p10/p90) of ``diag(H)``,
    ``diag(H_sys^-1)`` and ``VIF_j = H_sys,jj [H_sys^-1]_jj`` for every
    ``--dampings`` value.  Separates flattening by rotation (run with
    ``--rotation none``) from flattening by damping.

``mask``
    Pruned-set overlap of SparseGPT, QAP and the unweighted QAP score
    ``w^2 - eps^2`` with magnitude, split into a cross-group part (how many
    weights each quantization group loses) and a within-group part (whether a
    group prunes its smallest ``|w|``).  For a mid-rise grid ``w^2 - eps^2``
    increases with ``|w|`` inside a group, so its within-group mismatch must
    be exactly zero.

``residual``
    How much OBR's pruning compensation ``delta`` moves the kept weights
    relative to the step, and whether the rounding residual of ``w`` survives:
    the correlation of ``eps(w)`` with ``eps(w + delta)``.

``ties``
    Share of weights whose QAP score is exactly zero (they round to 0 on a
    mid-tread grid), against the sparsity.

    python minitests/criterion_anatomy.py --blocks 1,15,30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.prune.criteria import hessian_inverse_diag
from qdrift.prune.mask import mask_from_score

STEM = "criterion_anatomy"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=C.ints("1,15,30"))
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument("--bits", type=C.ints, default=C.ints("1,2,3,4"))
    parser.add_argument("--dampings", type=C.floats, default=C.floats("0.01,0.001"))
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--rows", type=int, default=256, help="0 = all rows")
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the sweep."""
    setup = f"rot-{args.rotation}_{C.gran_tag(args.group_size)}_{args.range}"
    sweep = f"{C.span(args.blocks, 'blk')}_u{args.sparsity:g}_{C.span(args.bits, 'w')}"
    return C.result_path(
        STEM, args.model, setup, sweep, f"rows{args.rows}", args.suffix
    )


def spread(values: torch.Tensor) -> dict[str, float]:
    """Coefficient of variation and quantile ratio of a positive vector."""
    values = values.double()
    p10, p50, p90 = torch.quantile(
        values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype, device=values.device)
    ).tolist()
    return {
        "cv": float(values.std() / values.mean()),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p90_over_p10": p90 / max(p10, 1e-30),
    }


def flat_rows(args, hessian: torch.Tensor, head: dict) -> list[dict]:
    """Spread of the Hessian diagonals and VIF for every damping."""
    rows = []
    for damping in args.dampings:
        system = C.damped(hessian, damping)
        hinv = hessian_inverse_diag(system)
        rows.append(
            {
                **head,
                "table": "flat",
                "damping": damping,
                "diag_h": spread(torch.diagonal(hessian).clamp_min(1e-30)),
                "diag_hinv": spread(hinv),
                "vif": spread(torch.diagonal(system) * hinv),
            }
        )
    return rows


def group_ids(cols: int, group_size: int, device: torch.device) -> torch.Tensor:
    """Quantization-group index of every column (one group per row if 0)."""
    size = group_size if group_size > 0 else cols
    return torch.arange(cols, device=device) // size


def mask_compare(
    pruned: torch.Tensor, base: torch.Tensor, weight: torch.Tensor, groups: torch.Tensor
) -> dict[str, float]:
    """Overlap of a pruned set with the magnitude one, cross vs within group."""
    count = groups.max().item() + 1
    index = groups.expand_as(pruned)
    per_group = torch.zeros(pruned.shape[0], count, device=pruned.device)
    per_base = torch.zeros_like(per_group)
    per_group.scatter_add_(1, index, pruned.float())
    per_base.scatter_add_(1, index, base.float())
    # Within-group check: given this criterion's pruned count k per group, is
    # the pruned set the k smallest |w| of the group?
    size = pruned.shape[1] // count
    absw = weight.abs().reshape(weight.shape[0], count, size)
    rank = absw.argsort(dim=2).argsort(dim=2)
    k = per_group.unsqueeze(2)
    smallest = rank < k
    within = (smallest != pruned.reshape_as(smallest)).any(dim=2).float().mean()
    total = base.sum().clamp_min(1)
    return {
        "shared_with_magnitude": float((pruned & base).sum() / total),
        "moved_across_groups": float((per_group - per_base).abs().sum() / 2 / total),
        "groups_not_smallest": float(within),
    }


@torch.no_grad()
def measure_layer(args, weight, hessian, system, head) -> list[dict]:
    """All four diagnostics for one layer."""
    rows = flat_rows(args, hessian, head)
    hinv = hessian_inverse_diag(system)
    subset = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if subset is not None:
        weight = weight[subset]
    groups = group_ids(weight.shape[1], args.group_size, weight.device)
    clip = args.range == "clip"
    base = ~mask_from_score(
        C.score("magnitude", weight, hinv), args.sparsity, "unstructured"
    )
    for grid in ("midrise", "midtread"):
        for bits in args.bits:
            if grid == "midtread" and bits < 2:
                continue
            step = C.weight_scale(weight, bits, grid, args.group_size, clip).scale
            error = C.quant_error(weight, step, bits, grid)
            gain = weight.pow(2) - error.pow(2)
            row = {**head, "grid": grid, "bits": bits}
            rows.append(
                {
                    **row,
                    "table": "ties",
                    "zero_score": float((gain == 0).float().mean()),
                    "sparsity": args.sparsity,
                }
            )
            if grid != "midrise":
                continue
            scores = {
                "sparsegpt": C.score("sparsegpt", weight, hinv, hinv=hinv),
                "qap": C.score("qap", weight, hinv, error=error, hinv=hinv),
                "qap_unweighted": gain,
            }
            for name, crit in scores.items():
                pruned = ~mask_from_score(crit, args.sparsity, "unstructured")
                rows.append(
                    {
                        **row,
                        "table": "mask",
                        "criterion": name,
                        **mask_compare(pruned, base, weight, groups),
                    }
                )
            state = C.pruning_state(weight, ~base, system, args.alpha)
            kept = weight.gather(1, state.keep)
            kept_step = step.gather(1, state.keep)
            moved = C.quant_error(state.retained, kept_step, bits, grid)
            before = error.gather(1, state.keep)
            ratio = (state.delta.abs() / kept_step).flatten()
            rows.append(
                {
                    **row,
                    "table": "residual",
                    "delta_over_step_p50": float(ratio.median()),
                    "delta_over_step_p90": float(ratio.quantile(0.9)),
                    "eps_corr": float(
                        torch.corrcoef(
                            torch.stack([before.flatten(), moved.flatten()])
                        )[0, 1]
                    ),
                    "eps_sign_agree": float(
                        (before.sign() == moved.sign()).float().mean()
                    ),
                    "kept_abs_over_step": float((kept.abs() / kept_step).median()),
                }
            )
            del state
    return rows


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    path = output_path(args)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(max(args.blocks) + 1))
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    hidden, kwargs = streamer.start(calib)
    rows: list[dict] = []
    sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
    for index in range(max(args.blocks) + 1):
        if index not in args.blocks:
            hidden = streamer.run(index, hidden, kwargs)
            continue
        grams, hidden = C.block_grams(streamer, index, hidden, kwargs, sources)
        for layer_type in args.types:
            start = time.time()
            hessian, system = grams[C.INPUT_SOURCE[layer_type]].hessians(args.damping)
            module = C.block_module(model, index, layer_type)
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            head = {
                "block": index,
                "type": layer_type,
                "layer": C.layer_name(index, layer_type),
            }
            rows += measure_layer(args, weight, hessian, system, head)
            del weight, hessian, system
            torch.cuda.empty_cache()
            C.save_result(path, STEM, vars(args), rows=rows)
            C.log(f"  {head['layer']} done in {time.time() - start:.0f}s")
        del grams
    C.log(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
