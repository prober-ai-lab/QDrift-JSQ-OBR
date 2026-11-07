# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: where does the Q-WANDA trade-off lambda pay off?

For selected layers of the (rotated) model, sweep the Q-WANDA lambda (lambda
= 1 is WANDA) at several bit widths and sparsities and measure the joint
layer-output error

    rel_err = tr(dW H dW^T) / tr(W H W^T),     dW = W_hat - W,

without compensation and with OBR compensation.  The diagonal proxy that
Q-WANDA optimizes, ``sum_P w^2 x^2 + sum_R eps^2 x^2``, is recorded next to it
so the figure can show how well the proxy tracks the true error.

The error decomposes over output rows, so OBR is evaluated on a random subset
of rows (``--rows``) to keep the per-row solves cheap; both compensation modes
use the same rows.

    python minitests/qwanda_lambda_sweep.py --blocks 1,15,30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.prune.mask import mask_from_score

STEM = "qwanda_lambda_sweep"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=C.ints("1,15,30"))
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument(
        "--lambdas",
        type=C.floats,
        default=C.floats("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"),
    )
    parser.add_argument("--bits", type=C.ints, default=C.ints("1,2,3,4"))
    parser.add_argument("--grids", type=C.names, default=["midrise"])
    parser.add_argument("--sparsities", type=C.floats, default=C.floats("0.5,0.7"))
    parser.add_argument("--comps", type=C.names, default=["none", "obr"])
    parser.add_argument(
        "--rows",
        type=int,
        default=256,
        help="output rows evaluated per layer (0 = all)",
    )
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the sweep."""
    setup = f"rot-{args.rotation}_{C.gran_tag(args.group_size)}_{args.range}"
    sweep = "_".join(
        [
            C.span(args.blocks, "blk"),
            C.span(args.lambdas, "lam"),
            C.span(args.sparsities, "s"),
            C.span(args.bits, "w"),
        ]
    )
    variant = "+".join(args.grids) + "_" + "+".join(args.comps) + f"_rows{args.rows}"
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


@torch.no_grad()
def measure_layer(
    args: argparse.Namespace,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    system: torch.Tensor,
    head: dict,
) -> list[dict]:
    """Error rows of one layer for every grid, bit width, s, and lambda."""
    rows = []
    subset = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if subset is not None:
        weight = weight[subset]
    diag = torch.diagonal(hessian).clamp_min(0)
    reference = C.output_error(weight, hessian)
    w2x = weight.pow(2) * diag.reshape(1, -1)
    clip = args.range == "clip"
    for grid in args.grids:
        for bits in args.bits:
            if grid == "midtread" and bits < 2:
                continue
            # Scales are row-local for channel/group granularity, so the row
            # subset sees exactly the scales of the full layer.
            scale = C.weight_scale(weight, bits, grid, args.group_size, clip).scale
            eps = C.quant_error(weight, scale, bits, grid)
            e2x = eps.pow(2) * diag.reshape(1, -1)
            for sparsity in args.sparsities:
                for lam in args.lambdas:
                    crit = C.score("qwanda", weight, diag, eps, lam)
                    mask = mask_from_score(crit, sparsity, "unstructured")
                    proxy = float((w2x * ~mask).sum() + (e2x * mask).sum())
                    for comp in args.comps:
                        if comp == "none":
                            new = C.plain(weight, mask, scale, bits, grid)
                        else:
                            state = C.pruning_state(weight, mask, system, args.alpha)
                            new = C.from_state(
                                weight, state, scale, bits, grid, args.alpha
                            )
                            del state
                        rows.append(
                            {
                                **head,
                                "grid": grid,
                                "bits": bits,
                                "sparsity": sparsity,
                                "lambda": lam,
                                "comp": comp,
                                "rel_err": C.output_error(new - weight, hessian)
                                / reference,
                                "proxy": proxy / float((w2x).sum()),
                            }
                        )
                        del new
            del scale, eps, e2x
            torch.cuda.empty_cache()
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
