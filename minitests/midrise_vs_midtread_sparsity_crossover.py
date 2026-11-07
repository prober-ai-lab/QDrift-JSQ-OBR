# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: at which sparsity does the mid-rise grid overtake mid-tread?

Mid-rise has no zero level, so every retained small weight is rounded to at
least +-s/2; that is wasteful for a dense layer but harmless once pruning has
removed the small weights.  For each layer and a WANDA mask (independent of
the grid) we sweep the sparsity and measure, for both grids,

``quant_err``  activation-weighted error of the retained weights only,
               tr(E H E^T) / tr(W H W^T) with E = (Q(W) - W) * M;
``joint_err``  the full joint layer-output error (pruned + quantized),
               without compensation, and with OBR on ``--obr-rows`` rows of
               the layers in ``--obr-blocks``.

The selected clip ratio rho* of every quantization group is histogrammed as
well, since range clipping is what lets 2^b mid-rise levels track the body of
the distribution.

    python minitests/midrise_vs_midtread_sparsity_crossover.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _common as C
import torch

from qdrift.compress.recipe import DEFAULT_CLIP_RHOS
from qdrift.prune.mask import mask_from_score

STEM = "midrise_vs_midtread_sparsity_crossover"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument(
        "--blocks", type=C.ints, default=C.ints("0,4,8,12,16,20,24,28,31")
    )
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument(
        "--sparsities",
        type=C.floats,
        default=C.floats("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument("--bits", type=C.ints, default=C.ints("2,3,4"))
    parser.add_argument(
        "--obr-blocks",
        type=C.ints,
        default=C.ints("1,15,30"),
        help="blocks that also get the OBR joint error ('' to skip)",
    )
    parser.add_argument("--obr-rows", type=int, default=256)
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the sweep."""
    setup = f"rot-{args.rotation}_{C.gran_tag(args.group_size)}_{args.range}_wanda"
    sweep = "_".join(
        [
            C.span(args.blocks, "blk"),
            C.span(args.sparsities, "s"),
            C.span(args.bits, "w"),
        ]
    )
    variant = "midrise+midtread_none"
    if args.obr_blocks:
        variant += f"_obr-{C.span(args.obr_blocks, 'blk')}-rows{args.obr_rows}"
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


@torch.no_grad()
def measure_layer(
    args: argparse.Namespace,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    system: torch.Tensor,
    with_obr: bool,
    head: dict,
) -> tuple[list[dict], list[dict]]:
    """Error rows and clip-ratio counts of one layer."""
    errors, rhos = [], []
    diag = torch.diagonal(hessian).clamp_min(0)
    reference = C.output_error(weight, hessian)
    crit = C.score("wanda", weight, diag)
    masks = {s: mask_from_score(crit, s, "unstructured") for s in args.sparsities}
    subset = C.row_subset(weight.shape[0], args.obr_rows, args.seed, weight.device)
    sub_weight = weight if subset is None else weight[subset]
    sub_reference = C.output_error(sub_weight, hessian)
    clip = args.range == "clip"
    for grid in ("midrise", "midtread"):
        for bits in args.bits:
            if grid == "midtread" and bits < 2:
                continue
            result = C.weight_scale(weight, bits, grid, args.group_size, clip)
            values, counts = torch.unique(
                result.rho_compact.flatten(), return_counts=True
            )
            rhos.append(
                {
                    **head,
                    "grid": grid,
                    "bits": bits,
                    "rho": {f"{float(v):.3g}": int(c) for v, c in zip(values, counts)},
                }
            )
            scale = result.scale
            del result
            eps = C.quant_error(weight, scale, bits, grid)
            for sparsity, mask in masks.items():
                row = {
                    **head,
                    "grid": grid,
                    "bits": bits,
                    "sparsity": sparsity,
                    "quant_err": C.output_error(eps * mask, hessian) / reference,
                    "joint_err": C.output_error(
                        C.plain(weight, mask, scale, bits, grid) - weight, hessian
                    )
                    / reference,
                }
                if with_obr:
                    sub_mask = mask if subset is None else mask[subset]
                    sub_scale = scale if subset is None else scale[subset]
                    state = C.pruning_state(sub_weight, sub_mask, system, args.alpha)
                    new = C.from_state(
                        sub_weight, state, sub_scale, bits, grid, args.alpha
                    )
                    row["joint_err_obr"] = (
                        C.output_error(new - sub_weight, hessian) / sub_reference
                    )
                    del state, new
                errors.append(row)
            del scale, eps
            torch.cuda.empty_cache()
    return errors, rhos


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    path = output_path(args)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(streamer.depth))
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    hidden, kwargs = streamer.start(calib)
    config = {**vars(args), "clip_rhos": list(DEFAULT_CLIP_RHOS)}
    tables: dict[str, list[dict]] = {"errors": [], "rho": []}
    blocks = sorted(set(args.blocks) | set(args.obr_blocks))
    sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
    for index in range(max(blocks) + 1):
        if index not in blocks:
            hidden = streamer.run(index, hidden, kwargs)
            continue
        start = time.time()
        grams, hidden = C.block_grams(streamer, index, hidden, kwargs, sources)
        for layer_type in args.types:
            hessian, system = grams[C.INPUT_SOURCE[layer_type]].hessians(args.damping)
            module = C.block_module(model, index, layer_type)
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            head = {
                "block": index,
                "type": layer_type,
                "layer": C.layer_name(index, layer_type),
            }
            errors, rhos = measure_layer(
                args, weight, hessian, system, index in args.obr_blocks, head
            )
            tables["errors"] += errors
            tables["rho"] += rhos
            del weight, hessian, system
            torch.cuda.empty_cache()
        del grams
        C.save_result(path, STEM, config, **tables)
        C.log(f"block {index} done in {time.time() - start:.0f}s")
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
