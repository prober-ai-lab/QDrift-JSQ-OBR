# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: which quantization-aware pruning score lowers the joint error?

For selected layers of the (rotated) model, build the mask with each pruning
score below, prune + quantize (dense-weight scale), optionally compensate with
OBR, and measure the joint layer-output error

    rel_err = tr(dW H dW^T) / tr(W H W^T),     dW = W_hat - W.

Scores (larger = keep; ``h_j = E[x_j^2]``, ``g_j = 1 / [H^-1]_jj``):

    wanda             w^2 h                      pruning only
    qwanda0.7         max(0, .7 w^2 - .3 eps^2) h
    obs               w^2 g                      pruning only, OBS saliency
    exchange          (w^2 - eps^2) h            realized rounding error
    exchange_exp      (w^2 - D^2/12) h           expected rounding error
    obs_exchange      (w^2 - eps^2) g
    obs_exchange_exp  (w^2 - D^2/12) g

``eps = Q(w) - w`` is the rounding error with the dense-weight scale and ``D``
the quantization step of the weight's group; ``D^2 / 12`` is the expected
squared rounding error, which does not depend on the exact value that OBR
compensation later rounds.  The error is row-decomposable, so it is measured
on ``--rows`` random rows.

    python minitests/criterion_compare.py --blocks 1,15,30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.prune.mask import mask_from_score

STEM = "criterion_compare"
CRITERIA = (
    "wanda",
    "qwanda0.7",
    "obs",
    "exchange",
    "exchange_exp",
    "obs_exchange",
    "obs_exchange_exp",
)
STRUCTURES = {
    "u0.5": ("unstructured", 0.5),
    "u0.7": ("unstructured", 0.7),
    "2to4": ("2:4", 0.5),
}


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=C.ints("1,15,30"))
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument("--criteria", type=C.names, default=list(CRITERIA))
    parser.add_argument("--structures", type=C.names, default=list(STRUCTURES))
    parser.add_argument("--bits", type=C.ints, default=C.ints("1,2,3,4"))
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midrise")
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
        [C.span(args.blocks, "blk"), "+".join(args.structures), C.span(args.bits, "w")]
    )
    variant = f"{args.grid}_" + "+".join(args.comps) + f"_rows{args.rows}"
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


def scores(
    criterion: str,
    weight: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    eps: torch.Tensor,
    step: torch.Tensor,
) -> torch.Tensor:
    """Score of one criterion; ``h`` and ``g`` are per input column."""
    w2 = weight.pow(2)
    if criterion == "wanda":
        return w2 * h
    if criterion == "qwanda0.7":
        return (0.7 * w2 - 0.3 * eps.pow(2)).clamp_min(0) * h
    if criterion == "obs":
        return w2 * g
    if criterion == "exchange":
        return (w2 - eps.pow(2)) * h
    if criterion == "exchange_exp":
        return (w2 - step.pow(2) / 12) * h
    if criterion == "obs_exchange":
        return (w2 - eps.pow(2)) * g
    if criterion == "obs_exchange_exp":
        return (w2 - step.pow(2) / 12) * g
    raise ValueError(f"unknown criterion {criterion}")


@torch.no_grad()
def measure_layer(
    args: argparse.Namespace,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    system: torch.Tensor,
    head: dict,
) -> list[dict]:
    """Error rows of one layer for every criterion, structure, and bit width."""
    rows = []
    h = torch.diagonal(hessian).clamp_min(0).reshape(1, -1)
    g = 1.0 / torch.diagonal(torch.cholesky_inverse(torch.linalg.cholesky(system)))
    g = g.reshape(1, -1)
    subset = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if subset is not None:
        weight = weight[subset]
    reference = C.output_error(weight, hessian)
    clip = args.range == "clip"
    for bits in args.bits:
        if args.grid == "midtread" and bits < 2:
            continue
        # Scales are row-local, so the row subset sees the full layer's scales.
        step = C.weight_scale(weight, bits, args.grid, args.group_size, clip).scale
        eps = C.quant_error(weight, step, bits, args.grid)
        for criterion in args.criteria:
            crit = scores(criterion, weight, h, g, eps, step)
            for tag in args.structures:
                structure, sparsity = STRUCTURES[tag]
                mask = mask_from_score(crit, sparsity, structure)
                for comp in args.comps:
                    if comp == "none":
                        new = C.plain(weight, mask, step, bits, args.grid)
                    else:
                        state = C.pruning_state(weight, mask, system, args.alpha)
                        new = C.from_state(
                            weight, state, step, bits, args.grid, args.alpha
                        )
                        del state
                    rows.append(
                        {
                            **head,
                            "grid": args.grid,
                            "bits": bits,
                            "structure": tag,
                            "criterion": criterion,
                            "comp": comp,
                            "rel_err": C.output_error(new - weight, hessian)
                            / reference,
                        }
                    )
                    del new
        del step, eps
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
