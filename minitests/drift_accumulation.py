# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: input drift accumulates with depth; DAC cancels part of it.

The whole model is compressed sequentially in memory with the pipeline's own
per-layer routine (``_quantize_layer``), once per compensation mode
(none / OBR / DAC), with the same mask criterion for all of them.  Before
each linear layer is compressed we have its full-precision input ``X`` (dense
model) and its actual input ``X_hat`` (compressed prefix); after compression
we record, over all calibration tokens,

``drift``      ||dX|| / ||X||                              input drift
``inherited``  E||W dx||^2                                 drift passed on by W
``injected``   E||dW x_hat||^2                             this layer's own error
``total``      E||W_hat x_hat - W x||^2                    asymmetric error
``cancel``     share of E||W dx||^2 that a free update of W could cancel,
               1 - min_D E||W dx + D x_hat||^2 / E||W dx||^2

all normalized by E||W x||^2 where useful.  The same quantities are measured
on held-out WikiText-2 test inputs (``held_*``), with the compressed weight and the update
``D`` fitted on the calibration set, so ``held_cancel`` is the share of the
inherited drift that a calibration-fitted update cancels out of sample.
Every quantity follows from the dual-path moments E[x_hat x_hat^T],
E[dx x_hat^T], E[dx dx^T], so no token matrix product is needed.  The dense
model is streamed block by block next to the compressed one, so only one
block's full-precision inputs are held.  The end-to-end WikiText-2 NLL of each
compressed model is recorded too.

    python minitests/drift_accumulation.py --comps none,obr,drift
    python minitests/drift_accumulation.py --bits 4 --grid midtread --group-size 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import common as C
import torch

from qdrift.compress.pipeline import _quantize_layer
from qdrift.compress.recipe import (
    PRUNING_CRITERIA,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    RotationSpec,
    WeightQuantSpec,
)
from qdrift.obr.drift import drift_weight
from qdrift.quant.quantize import qrange

STEM = "drift_accumulation"
# Capture groups in block order: later groups read the compressed earlier ones.
GROUPS = (
    ("q", ("q", "k", "v")),
    ("o", ("o",)),
    ("gate", ("gate", "up")),
    ("down", ("down",)),
)
_ROWS = 4096


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--comps", type=C.names, default=["none", "obr", "drift"])
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--grid", choices=("midrise", "midtread"), default="midrise")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--criterion", choices=PRUNING_CRITERIA, default="sparsegpt")
    parser.add_argument("--quant", choices=("rtn", "gptq"), default="gptq")
    parser.add_argument("--max-blocks", type=int, default=0, help="0 = all blocks")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--holdout", type=int, default=32, help="WikiText-2 test chunks")
    # Four hidden-state streams and two sets of moments share the card.
    parser.set_defaults(reserve_gb=12.0)
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    """Result file named after the recipe and the compensation modes."""
    setup = f"rot-{args.rotation}_{args.nsamples}x{args.seqlen}"
    gran = C.gran_tag(args.group_size)
    crit = args.criterion
    grid = "mr" if args.grid == "midrise" else "mt"
    config = (
        f"w{args.bits}_{gran}_{grid}_{args.range}_{args.quant}"
        f"_u{args.sparsity:g}_{crit}"
    )
    if args.max_blocks:
        config += f"_first{args.max_blocks}"
    return C.result_path(
        STEM, args.model, setup, config, "+".join(args.comps), args.suffix
    )


def make_recipe(args: argparse.Namespace, comp: str) -> CompressionRecipe:
    """Pipeline recipe for one compensation mode."""
    kind = "group" if args.group_size > 0 else "channel"
    return CompressionRecipe(
        model=args.model,
        weight=WeightQuantSpec(
            bits=args.bits,
            granularity=GranularitySpec(
                kind, args.group_size if args.group_size > 0 else None
            ),
            grid=args.grid,
            range=args.range,
            method=args.quant,
        ),
        prune=PruneSpec(
            sparsity=args.sparsity,
            structure="unstructured",
            criterion=args.criterion,
        ),
        rotation=RotationSpec(args.rotation, args.seed),
        compensation=CompensationSpec(comp, args.alpha, args.damping),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        pairwise_fraction=0.0,
    )


class DualMoments:
    """E[x_hat x_hat^T], E[dx x_hat^T], E[dx dx^T] and E[x x^T] on the GPU."""

    def __init__(
        self, fp: torch.Tensor, hat: torch.Tensor, device: torch.device
    ) -> None:
        """Accumulate the moments over all calibration tokens."""
        dim = fp.shape[1]
        zeros = lambda: torch.zeros(dim, dim, device=device, dtype=torch.float32)  # noqa: E731
        self.hat, self.diff, self.dd = zeros(), zeros(), zeros()
        for start in range(0, fp.shape[0], _ROWS):
            zhat = hat[start : start + _ROWS].to(device=device, dtype=torch.float32)
            dx = zhat - fp[start : start + _ROWS].to(device=device, dtype=torch.float32)
            self.hat.addmm_(zhat.t(), zhat)
            self.diff.addmm_(dx.t(), zhat)
            self.dd.addmm_(dx.t(), dx)
            del zhat, dx
        count = max(fp.shape[0], 1)
        for matrix in (self.hat, self.diff, self.dd):
            matrix.div_(count)
        # x = x_hat - dx  =>  E[x x^T] = H_hat - H_diff - H_diff^T + E[dx dx^T]
        self.fp = self.hat - self.diff - self.diff.t() + self.dd


def trace_form(a: torch.Tensor, matrix: torch.Tensor, b: torch.Tensor) -> float:
    """tr(a M b^T)."""
    return float((a @ matrix).mul_(b).sum())


def decomposition(
    weight: torch.Tensor, delta: torch.Tensor, update: torch.Tensor, moments: DualMoments
) -> dict:
    """Error terms of ``W + delta`` and the residual drift under ``update``."""
    reference = trace_form(weight, moments.fp, weight)
    inherited = trace_form(weight, moments.dd, weight)
    injected = trace_form(delta, moments.hat, delta)
    # E[(dW x_hat) . (W dx)] = tr(dW E[x_hat dx^T] W^T) = tr(dW H_diff^T W^T)
    cross = trace_form(delta, moments.diff.t(), weight)
    residual = (
        inherited
        + 2.0 * float((update * (weight @ moments.diff)).sum())
        + trace_form(update, moments.hat, update)
    )
    return {
        "reference": reference,
        "inherited": inherited / reference,
        "injected": injected / reference,
        "cross": 2.0 * cross / reference,
        "total": (injected + inherited + 2.0 * cross) / reference,
        "cancel": 1.0 - residual / inherited if inherited > 0 else math.nan,
    }


@torch.no_grad()
def layer_metrics(
    weight: torch.Tensor,
    new: torch.Tensor,
    moments: DualMoments,
    held: DualMoments,
    system: torch.Tensor,
) -> dict:
    """Error decomposition of one compressed layer, in and out of sample."""
    delta = new - weight
    # Best dense update against the inherited drift: D = -(W H_diff) H_sys^-1,
    # fitted on the calibration moments.
    drift = weight @ moments.diff
    update = -torch.cholesky_solve(drift.t(), torch.linalg.cholesky(system)).t()
    out = decomposition(weight, delta, update, moments)
    for key, value in decomposition(weight, delta, update, held).items():
        out[f"held_{key}"] = value
    return out


def capture(
    streamer: C.BlockStreamer,
    index: int,
    hidden: list[torch.Tensor],
    kwargs: list[dict],
    source: str,
) -> torch.Tensor:
    """Capture the current input of ``source`` in block ``index``."""
    module = C.block_module(streamer.model, index, source)
    values: list[torch.Tensor] = []

    def tap(value: torch.Tensor) -> None:
        # The compressed path is captured in fp32, as SequentialCalibration does.
        """Store one batch of the compressed-path input."""
        values.append(value.reshape(-1, value.shape[-1]).to("cpu", torch.float32))

    streamer.run(index, hidden, kwargs, {module: tap})
    return torch.cat(values)


def dense_inputs(
    streamer: C.BlockStreamer, index: int, hidden: list[torch.Tensor], kwargs: list[dict]
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Inputs of every capture point of dense block ``index``, and its output."""
    values: dict[str, list[torch.Tensor]] = {source: [] for source, _ in GROUPS}
    taps = {
        C.block_module(streamer.model, index, source): (
            lambda value, source=source: values[source].append(
                value.reshape(-1, value.shape[-1]).to("cpu", torch.float16)
            )
        )
        for source, _ in GROUPS
    }
    hidden = streamer.run(index, hidden, kwargs, taps)
    return {source: torch.cat(chunks) for source, chunks in values.items()}, hidden


@torch.no_grad()
def compress_and_measure(
    args: argparse.Namespace,
    comp: str,
    streamer: C.BlockStreamer,
    start: tuple[list[torch.Tensor], list[dict]],
    held_start: tuple[list[torch.Tensor], list[dict]],
    depth: int,
    evals: list[torch.Tensor],
) -> dict:
    """Compress the model sequentially and measure every layer.

    The dense streams pass block ``index`` before it is compressed, the
    compressed streams after.
    """
    recipe = make_recipe(args, comp)
    qparams = qrange(args.bits, args.grid)
    hidden, kwargs = start
    held, held_kwargs = held_start
    fp_hidden, fp_held = hidden, held
    rows = []
    t0 = time.time()
    for index in range(depth):
        fp_inputs, fp_hidden = dense_inputs(streamer, index, fp_hidden, kwargs)
        fp_held_inputs, fp_held = dense_inputs(streamer, index, fp_held, held_kwargs)
        for source, types in GROUPS:
            hat = capture(streamer, index, hidden, kwargs, source)
            moments = DualMoments(fp_inputs.pop(source), hat, streamer.device)
            del hat
            hat = capture(streamer, index, held, held_kwargs, source)
            held_moments = DualMoments(fp_held_inputs.pop(source), hat, streamer.device)
            del hat
            system = C.damped(moments.hat, args.damping)
            drift_rel = math.sqrt(
                float(torch.trace(moments.dd))
                / max(float(torch.trace(moments.fp)), 1e-30)
            )
            for layer_type in types:
                module = C.block_module(streamer.model, index, layer_type)
                weight = module.weight.data.to(
                    device=streamer.device, dtype=torch.float32
                )
                use_drift = comp == "drift"
                result = _quantize_layer(
                    recipe,
                    qparams,
                    C.layer_name(index, layer_type),
                    weight,
                    weight,
                    moments.hat,
                    system,
                    system,
                    moments.diff if use_drift else None,
                    drift_weight(weight, moments.diff) if use_drift else None,
                )
                new = result.new_weight.to(device=streamer.device, dtype=torch.float32)
                module.weight.data.copy_(result.new_weight.to(module.weight.device))
                rows.append(
                    {
                        "comp": comp,
                        "block": index,
                        "type": layer_type,
                        "layer": C.layer_name(index, layer_type),
                        "drift": drift_rel,
                        **layer_metrics(weight, new, moments, held_moments, system),
                    }
                )
                del weight, new, result
            del moments, held_moments, system
            torch.cuda.empty_cache()
        hidden = streamer.run(index, hidden, kwargs)
        held = streamer.run(index, held, held_kwargs)
        last = rows[-1]
        C.log(
            f"[{comp}] block {index} done, down drift {last['drift']:.3f} "
            f"cancel {last['cancel']:.3f} held {last['held_cancel']:.3f}, "
            f"elapsed {time.time() - t0:.0f}s"
        )
    e_hidden, e_kwargs = streamer.start(evals)
    nll = (
        streamer.nll_from(0, e_hidden, e_kwargs, evals)
        if depth == streamer.depth
        else None
    )
    return {"rows": rows, "nll": nll}


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    path = output_path(args)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(streamer.depth))
    C.log(f"resident blocks: {len(streamer.resident)}/{streamer.depth}")
    depth = args.max_blocks or streamer.depth
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    evals = C.eval_batches(tokenizer, args.eval_samples, args.eval_seqlen)
    held = C.get_loaders(
        "wikitext2", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
    )
    start = streamer.start(calib)
    held_start = streamer.start(C.batch_chunks(held, args.batch_size))
    e_hidden, e_kwargs = streamer.start(evals)
    dense_nll = streamer.nll_from(0, e_hidden, e_kwargs, evals)
    del e_hidden, e_kwargs
    C.log(f"dense NLL {dense_nll:.5f}")

    linears = (
        {
            # Kept on the CPU: resident blocks would put the copy on the GPU.
            name: module.weight.data.to("cpu", copy=True)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.")
        }
        if len(args.comps) > 1
        else {}
    )
    config = {
        **vars(args),
        "depth": depth,
        "eval_tokens": sum(int(b.numel()) for b in evals),
    }
    rows: list[dict] = []
    nll: dict[str, float | None] = {"dense": dense_nll}
    for comp in args.comps:
        for name, weight in linears.items():
            model.get_submodule(name).weight.data.copy_(weight)
        result = compress_and_measure(
            args, comp, streamer, start, held_start, depth, evals
        )
        rows += result["rows"]
        nll[comp] = result["nll"]
        C.save_result(path, STEM, config, nll=nll, rows=rows)
        C.log(f"[{comp}] NLL {result['nll']}")
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
