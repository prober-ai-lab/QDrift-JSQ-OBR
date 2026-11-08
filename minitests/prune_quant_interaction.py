# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: does sparsification absorb quantization error?

Only ONE linear layer of the (rotated) model is compressed at a time; every
other layer stays dense.  For that layer we measure the end-to-end NLL change
on WikiText-2 test chunks under

    quant-only   Q(W)              (scale from the dense weight)
    prune-only   W * M             (WANDA mask)
    joint        Q(W) * M

each without compensation and, optionally, with OBR compensation, and report
the interaction

    Delta = dNLL_joint - dNLL_quant - dNLL_prune.

Delta <= 0 means the joint error never exceeds the sum of the two individual
errors: pruning absorbs the quantization error of every weight it removes.
The layer-output error ``tr(dW H dW^T) / tr(W H W^T)`` is recorded as well.

Because a single layer of block ``l`` changes, the dense hidden state at the
input of block ``l`` is cached once and each evaluation only replays blocks
``l..L-1``.

    python minitests/prune_quant_interaction.py \
        --blocks 0,10,20,31 --sparsities 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
        --bits 1,2,3,4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.prune.mask import mask_from_score

STEM = "prune_quant_interaction"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=C.ints("0,10,20,31"))
    parser.add_argument("--types", type=C.layer_types, default=list(C.LAYER_TYPES))
    parser.add_argument(
        "--sparsities",
        type=C.floats,
        default=C.floats("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument("--bits", type=C.ints, default=C.ints("1,2,3,4"))
    parser.add_argument("--grids", type=C.names, default=["midrise"])
    parser.add_argument("--comps", type=C.names, default=["none", "obr"])
    parser.add_argument(
        "--obr-grids",
        type=C.names,
        default=["midrise"],
        help="grids that also get the (expensive) OBR variant",
    )
    parser.add_argument("--criterion", choices=("wanda", "magnitude"), default="wanda")
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the sweep."""
    gran = C.gran_tag(args.group_size)
    setup = f"rot-{args.rotation}_{gran}_{args.range}_{args.criterion}"
    sweep = "_".join(
        [
            C.span(args.blocks, "blk"),
            C.span(args.sparsities, "s"),
            C.span(args.bits, "w"),
        ]
    )
    variant = "+".join(args.grids) + "_" + "+".join(args.comps)
    if "obr" in args.comps:
        variant += "-" + "+".join(args.obr_grids)
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


def bit_grids(args: argparse.Namespace, comp: str) -> list[tuple[str, int]]:
    """(grid, bits) pairs evaluated for one compensation mode."""
    grids = (
        args.grids if comp == "none" else [g for g in args.grids if g in args.obr_grids]
    )
    # A mid-tread grid needs at least 2 bits: its 1-bit grid is just {0}.
    return [
        (grid, bits)
        for grid in grids
        for bits in args.bits
        if not (grid == "midtread" and bits < 2)
    ]


@torch.no_grad()
def measure_layer(
    args: argparse.Namespace,
    streamer: C.BlockStreamer,
    index: int,
    layer_type: str,
    grams: dict[str, C.Gram],
    e_hidden: list[torch.Tensor],
    e_kwargs: list[dict],
    evals: list[torch.Tensor],
    dense_nll: float,
) -> list[dict]:
    """All dNLL rows of one layer; the layer is restored afterwards."""
    module = C.block_module(streamer.model, index, layer_type)
    original = module.weight.data.clone()
    device = streamer.device
    weight = original.to(device=device, dtype=torch.float32)
    hessian, system = grams[C.INPUT_SOURCE[layer_type]].hessians(args.damping)
    diag = torch.diagonal(hessian).clamp_min(0)
    reference = C.output_error(weight, hessian)
    clip = args.range == "clip"
    rows: list[dict] = []

    def evaluate(new_weight: torch.Tensor, **row) -> None:
        """Install ``new_weight``, record its NLL, and restore the layer."""
        module.weight.data.copy_(new_weight.to(module.weight.dtype))
        nll = streamer.nll_from(index, e_hidden, e_kwargs, evals)
        module.weight.data.copy_(original)
        rows.append(
            {
                "block": index,
                "type": layer_type,
                "layer": C.layer_name(index, layer_type),
                **row,
                "nll": nll,
                "dnll": nll - dense_nll,
                "rel_err": C.output_error(new_weight - weight, hessian) / reference,
            }
        )

    scales = {
        (grid, bits): C.weight_scale(weight, bits, grid, args.group_size, clip).scale
        for comp in args.comps
        for grid, bits in bit_grids(args, comp)
    }
    full_mask = torch.ones_like(weight, dtype=torch.bool)

    for comp in args.comps:
        pairs = bit_grids(args, comp)
        # Quant-only: the dense mask is the s = 0 point of the joint sweep.
        state = (
            None
            if comp == "none"
            else C.pruning_state(weight, full_mask, system, args.alpha)
        )
        for grid, bits in pairs:
            scale = scales[(grid, bits)]
            if comp == "none":
                new = C.plain(weight, None, scale, bits, grid)
            else:
                new = C.from_state(weight, state, scale, bits, grid, args.alpha)
            evaluate(
                new, comp=comp, variant="quant", grid=grid, bits=bits, sparsity=0.0
            )

        crit = C.score(args.criterion, weight, diag)
        for sparsity in args.sparsities:
            mask = mask_from_score(crit, sparsity, "unstructured")
            state = (
                None
                if comp == "none"
                else C.pruning_state(weight, mask, system, args.alpha)
            )
            if comp == "none":
                new = C.plain(weight, mask, None)
            else:
                new = C.from_state(weight, state, None)
            evaluate(
                new, comp=comp, variant="prune", grid=None, bits=16, sparsity=sparsity
            )
            for grid, bits in pairs:
                scale = scales[(grid, bits)]
                if comp == "none":
                    new = C.plain(weight, mask, scale, bits, grid)
                else:
                    new = C.from_state(weight, state, scale, bits, grid, args.alpha)
                evaluate(
                    new,
                    comp=comp,
                    variant="joint",
                    grid=grid,
                    bits=bits,
                    sparsity=sparsity,
                )
            del state
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
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    evals = C.eval_batches(tokenizer, args.eval_samples, args.eval_seqlen)
    c_hidden, c_kwargs = streamer.start(calib)
    e_hidden, e_kwargs = streamer.start(evals)
    # Evaluations replay the tail of the model, so keep the tail resident.
    streamer.fit_resident(range(streamer.depth - 1, -1, -1))
    C.log(f"resident blocks: {len(streamer.resident)}/{streamer.depth}")

    config = {**vars(args), "eval_tokens": sum(int(b.numel()) for b in evals)}
    rows: list[dict] = []
    dense: dict[int, float] = {}
    sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
    last = max(args.blocks)
    for index in range(last + 1):
        if index not in args.blocks:
            c_hidden = streamer.run(index, c_hidden, c_kwargs)
            e_hidden = streamer.run(index, e_hidden, e_kwargs)
            continue
        grams, c_next = C.block_grams(streamer, index, c_hidden, c_kwargs, sources)
        dense[index] = streamer.nll_from(index, e_hidden, e_kwargs, evals)
        C.log(f"block {index}: dense NLL {dense[index]:.5f}")
        for layer_type in args.types:
            start = time.time()
            rows += measure_layer(
                args,
                streamer,
                index,
                layer_type,
                grams,
                e_hidden,
                e_kwargs,
                evals,
                dense[index],
            )
            C.save_result(path, STEM, config, dense_nll=dense, rows=rows)
            elapsed = time.time() - start
            C.log(f"  {C.layer_name(index, layer_type)} done in {elapsed:.0f}s")
        del grams
        torch.cuda.empty_cache()
        if index < last:
            c_hidden = c_next
            e_hidden = streamer.run(index, e_hidden, e_kwargs)
    C.log(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
