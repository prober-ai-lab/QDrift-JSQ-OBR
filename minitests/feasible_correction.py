# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: how much of the drift correction the sparse support can keep.

Measures the two per-layer quantities of the paper's Appendix "Measuring the
Feasible Drift Correction" on OBR's prefix.  For each linear layer, with the
drifted-input system ``H = E[x_hat x_hat^T] + lambda I`` (OBR's damping),
``D = E[dx x_hat^T]``, the full drift correction ``delta_1 = -W D H^-1`` and
the support ``S`` of the layer's OBR mask:

``u``              feasible correction, the change of OBR's first-stage
                   (real-valued) compensation when the target moves from ``W``
                   to ``W + delta_1``; per row ``u_S = -(W D)_S [H]_SS^-1``,
                   which is the H-orthogonal projection of ``delta_1`` onto S
``inherited``      E||W dx||^2
``dense``          ||delta_1||_H^2      (energy a dense update cancels)
``feasible``       ||u||_H^2            (energy an update on S cancels)
``rho``            feasible / dense, the feasible fraction
``alpha_cal``      step along u minimizing the asymmetric error on the
                   calibration tokens with the damped moments; exactly 1
``alpha_held``     the same step on held-out WikiText-2 validation tokens

Every sum runs over the rows of the layer.  The layer is then compressed with
OBR's full routine (``_quantize_layer``), so the next layer sees OBR's prefix.

    python minitests/feasible_correction.py --sparsity 0.7
    python minitests/feasible_correction.py --structure 4:8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch
from drift_accumulation import GROUPS, DualMoments, capture, dense_inputs, trace_form

from qdrift.compress.pipeline import _quantize_layer, _score
from qdrift.compress.recipe import (
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    RotationSpec,
    WeightQuantSpec,
)
from qdrift.obr.obr import compensate_pruning_error
from qdrift.prune.mask import mask_from_score
from qdrift.prune.select import obs_mask
from qdrift.quant.quantize import qrange

STEM = "feasible_correction"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midtread")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--structure", choices=("unstructured", "4:8"), default="unstructured")
    parser.add_argument("--max-blocks", type=int, default=0, help="0 = all blocks")
    parser.add_argument("--holdout", type=int, default=32, help="WikiText-2 validation chunks")
    parser.set_defaults(seqlen=1024, group_size=0, reserve_gb=12.0)
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the recipe."""
    setup = f"rot-{args.rotation}_{args.nsamples}x{args.seqlen}"
    grid = "mr" if args.grid == "midrise" else "mt"
    support = "4to8p-obs" if args.structure == "4:8" else f"u{args.sparsity:g}_sparsegpt"
    config = f"w{args.bits}_{C.gran_tag(args.group_size)}_{grid}_{args.range}_gptq_{support}"
    if args.max_blocks:
        config += f"_first{args.max_blocks}"
    return C.result_path(STEM, args.model, setup, config, "obr-prefix", args.suffix)


def make_recipe(args: argparse.Namespace) -> CompressionRecipe:
    """OBR's recipe for the prefix (mask, quantizer, compensation)."""
    kind = "group" if args.group_size > 0 else "channel"
    pairwise = args.structure == "4:8"
    return CompressionRecipe(
        model=args.model,
        weight=WeightQuantSpec(
            bits=args.bits,
            granularity=GranularitySpec(kind, args.group_size if args.group_size > 0 else None),
            grid=args.grid,
            range=args.range,
            method="gptq",
        ),
        prune=PruneSpec(
            sparsity=0.5 if pairwise else args.sparsity,
            structure=args.structure,
            criterion="sparsegpt",
            select="obs" if pairwise else "score",
        ),
        rotation=RotationSpec(args.rotation, args.seed),
        compensation=CompensationSpec("obr", args.alpha, args.damping),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        pairwise_fraction=1.0 if pairwise else 0.0,
    )


def obr_mask(recipe: CompressionRecipe, weight: torch.Tensor, hessian: torch.Tensor,
             system: torch.Tensor) -> torch.Tensor:
    """The retained mask ``_quantize_layer`` picks for OBR on these inputs."""
    if recipe.prune.structure == "4:8":
        return obs_mask(weight, system, "4:8_pairwise")
    diag = torch.diagonal(hessian).clamp_min(0)
    score = _score(recipe.prune.criterion, weight, diag, system)
    return mask_from_score(score, recipe.prune.sparsity, "unstructured")


def scatter(values: torch.Tensor, keep: torch.Tensor, cols: int) -> torch.Tensor:
    """Pad compact retained values with zeros on the pruned columns."""
    full = torch.zeros(values.shape[0], cols, device=values.device, dtype=values.dtype)
    return full.scatter_(1, keep, values)


@torch.no_grad()
def layer_metrics(weight: torch.Tensor, mask: torch.Tensor, moments: DualMoments,
                  held: DualMoments, system: torch.Tensor) -> dict:
    """Feasible fraction and optimal steps of one layer (all sums over rows)."""
    cols = weight.shape[1]
    drift = weight @ moments.diff  # W D
    factor = torch.linalg.cholesky(system)
    delta = -torch.cholesky_solve(drift.t(), factor).t()  # delta_1 = -W D H^-1
    # u_S = -(W D)_S [H]_SS^-1: OBR's first stage with weight 0 and drift term W D.
    state = compensate_pruning_error(
        torch.zeros_like(weight), system, system, None, mask, drift_weight=drift
    )
    u = scatter(state.delta, state.keep, cols)
    base = scatter(
        compensate_pruning_error(weight, system, system, None, mask).retained, state.keep, cols
    )
    del state
    inherited = trace_form(weight, moments.dd, weight)
    dense = -float((delta * drift).sum())  # delta H delta^T = -delta (W D)^T
    feasible = trace_form(u, system, u)
    # Step along u minimizing the asymmetric error E||(W0 + a u) x_hat - W x||^2:
    # a = tr(U (C W^T - H W0^T)) / tr(U H U^T) with C = E[x_hat x^T], C^T = H_hat - D.
    damp = system - moments.hat  # lambda I
    cal = float((u * (weight @ (moments.hat - moments.diff + damp) - base @ system)).sum())
    held_num = float((u * (weight @ (held.hat - held.diff) - base @ held.hat)).sum())
    held_den = trace_form(u, held.hat, u)
    return {
        "inherited": inherited,
        "dense": dense,
        "feasible": feasible,
        "rho": feasible / dense if dense > 0 else float("nan"),
        "alpha_cal": cal / feasible if feasible > 0 else float("nan"),
        "alpha_held": held_num / held_den if held_den > 0 else float("nan"),
        "held_num": held_num,
        "held_den": held_den,
        "kept": float(mask.float().mean()),
    }


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    path = output_path(args)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(streamer.depth))
    depth = args.max_blocks or streamer.depth
    calib = C.calibration_batches(tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size)
    held = C.get_loaders("wikitext2_val", tokenizer, nsamples=args.holdout,
                         seqlen=args.seqlen, eval_mode=True)
    hidden, kwargs = streamer.start(calib)
    held_hidden, held_kwargs = streamer.start(C.batch_chunks(held, args.batch_size))
    fp_hidden, fp_held = hidden, held_hidden
    recipe = make_recipe(args)
    qparams = qrange(args.bits, args.grid)
    config = {**vars(args), "depth": depth}
    rows: list[dict] = []
    t0 = time.time()
    for index in range(depth):
        fp_inputs, fp_hidden = dense_inputs(streamer, index, fp_hidden, kwargs)
        fp_held_inputs, fp_held = dense_inputs(streamer, index, fp_held, held_kwargs)
        for source, types in GROUPS:
            hat = capture(streamer, index, hidden, kwargs, source)
            moments = DualMoments(fp_inputs.pop(source), hat, streamer.device)
            hat = capture(streamer, index, held_hidden, held_kwargs, source)
            held_moments = DualMoments(fp_held_inputs.pop(source), hat, streamer.device)
            del hat
            system = C.damped(moments.hat, args.damping)
            for layer_type in types:
                module = C.block_module(streamer.model, index, layer_type)
                weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
                mask = obr_mask(recipe, weight, moments.hat, system)
                metrics = layer_metrics(weight, mask, moments, held_moments, system)
                result = _quantize_layer(
                    recipe, qparams, C.layer_name(index, layer_type), weight, weight,
                    moments.hat, system, system, None, None,
                )
                module.weight.data.copy_(result.new_weight.to(module.weight.device))
                rows.append({"block": index, "type": layer_type,
                             "layer": C.layer_name(index, layer_type), **metrics})
                del weight, mask, result
            del moments, held_moments, system
            torch.cuda.empty_cache()
        hidden = streamer.run(index, hidden, kwargs)
        held_hidden = streamer.run(index, held_hidden, held_kwargs)
        last = rows[-1]
        C.log(f"block {index}: down rho {last['rho']:.3f} alpha_cal {last['alpha_cal']:.4f} "
              f"alpha_held {last['alpha_held']:.3f}, elapsed {time.time() - t0:.0f}s")
        C.save_result(path, STEM, config, rows=rows)
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
