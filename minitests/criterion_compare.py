# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: which pruning criterion gives the lowest joint error?

For selected layers of the (rotated) model, build the mask with each pruning
criterion, prune + quantize (dense-weight scale), optionally compensate with
OBR, and measure the joint layer-output error

    rel_err = tr(dW H dW^T) / tr(W H W^T),     dW = W_hat - W.

Criteria (larger = keep; ``h_j = E[x_j^2]``, ``g_j = 1 / [H^-1]_jj``):

    magnitude   |w|
    wanda       w^2 h                    pruning loss, no compensation
    sparsegpt   w^2 g                    pruning loss after compensation (OBS)
    qap         (w^2 - eps^2) g          minus the rounding loss

``eps = Q(w) - w`` is the rounding error of the weight under the layer's own
quantizer, the error it incurs if retained.  Candidate redesigns (``D`` the
step, ``VIF_j = H_jj [H^-1]_jj``):

    eqap        (w^2 - D^2/12) g         expected rounding loss
    kqap        (w^2 - k_j D^2/12) g     k_j = 1 on columns whose rounding
                                         error OBR compensates (j < alpha d),
                                         VIF_j on the rest
    wqap        (w^2 - eps^2) h          QAP without compensation

The error is row-decomposable, so it is measured on ``--rows`` random rows,
on the calibration tokens and, as ``rel_err_holdout``, on ``--holdout`` C4
chunks.

    python minitests/criterion_compare.py --blocks 1,15,30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch

from qdrift.data.dataloader import get_loaders
from qdrift.prune.criteria import hessian_inverse_diag
from qdrift.prune.mask import mask_from_score

STEM = "criterion_compare"
CRITERIA = ("magnitude", "wanda", "sparsegpt", "qap")
CANDIDATES = ("eqap", "kqap", "wqap")
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
        "--holdout", type=int, default=0, help="held-out C4 chunks, 0 = none"
    )
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
    if args.criteria != list(CRITERIA):
        variant += "_" + "+".join(args.criteria)
    if args.holdout:
        variant += f"_holdout{args.holdout}"
    return C.result_path(STEM, args.model, setup, sweep, variant, args.suffix)


def candidate_score(
    criterion: str,
    weight: torch.Tensor,
    diag: torch.Tensor,
    hinv: torch.Tensor,
    vif: torch.Tensor,
    step: torch.Tensor,
    error: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Score of one criterion, including the candidate QAP redesigns."""
    if criterion not in CANDIDATES:
        return C.score(criterion, weight, diag, error=error, hinv=hinv)
    if criterion == "wqap":
        return (weight.pow(2) - error.pow(2)) * diag.reshape(1, -1)
    kappa = torch.ones_like(vif)
    if criterion == "kqap":
        late = torch.arange(len(vif), device=vif.device) >= int(len(vif) * alpha)
        kappa = torch.where(late, vif, kappa)
    return (weight.pow(2) - kappa * step.pow(2) / 12.0) / hinv.reshape(1, -1)


@torch.no_grad()
def measure_layer(
    args: argparse.Namespace,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    system: torch.Tensor,
    head: dict,
    holdout: torch.Tensor | None = None,
) -> list[dict]:
    """Error rows of one layer for every criterion, structure, and bit width."""
    rows = []
    diag = torch.diagonal(hessian).clamp_min(0)
    hinv = hessian_inverse_diag(system)
    vif = torch.diagonal(system) * hinv
    subset = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if subset is not None:
        weight = weight[subset]
    reference = C.output_error(weight, hessian)
    held = None if holdout is None else C.output_error(weight, holdout)
    clip = args.range == "clip"
    for bits in args.bits:
        if args.grid == "midtread" and bits < 2:
            continue
        # Scales are row-local, so the row subset sees the full layer's scales.
        step = C.weight_scale(weight, bits, args.grid, args.group_size, clip).scale
        for criterion in args.criteria:
            error = C.quant_error(weight, step, bits, args.grid)
            crit = candidate_score(
                criterion, weight, diag, hinv, vif, step, error, args.alpha
            )
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
                    if held is not None:
                        rows[-1]["rel_err_holdout"] = (
                            C.output_error(new - weight, holdout) / held
                        )
                    del new
        del step
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
    held, held_kwargs = [], []
    if args.holdout:
        chunks = get_loaders(
            "c4", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
        )
        held, held_kwargs = streamer.start(C.batch_chunks(chunks, args.batch_size))
    rows: list[dict] = []
    sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
    for index in range(max(args.blocks) + 1):
        if index not in args.blocks:
            hidden = streamer.run(index, hidden, kwargs)
            if held:
                held = streamer.run(index, held, held_kwargs)
            continue
        grams, hidden = C.block_grams(streamer, index, hidden, kwargs, sources)
        held_grams = {}
        if held:
            held_grams, held = C.block_grams(
                streamer, index, held, held_kwargs, sources
            )
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
            source = C.INPUT_SOURCE[layer_type]
            holdout = (
                held_grams[source].hessians(args.damping)[0] if held_grams else None
            )
            rows += measure_layer(args, weight, hessian, system, head, holdout)
            del weight, hessian, system
            torch.cuda.empty_cache()
            C.save_result(path, STEM, vars(args), rows=rows)
            C.log(f"  {head['layer']} done in {time.time() - start:.0f}s")
        del grams, held_grams
    C.log(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
