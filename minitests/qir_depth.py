# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: where does QIR change the deployed error, block by block?

Only one linear layer is compressed at a time, on its full-precision input
``x`` (dense prefix).  At deployment it sees ``q = Q_a(x) = x + n`` with the
runtime's per-token activation quantizer.  Both mechanisms run OBR's solver
on a system ``H`` and a cross matrix ``B`` (row target ``b = B w``, i.e.
``D = H - B^T``), fitted on calibration inputs:

``obr``  H = E[x x^T],          B = E[x x^T]
``qir``  H = E[x_hat x_hat^T],  B = E[x_hat x^T],  x_hat = Q_a(x) on the
         outlier tokens (``max|x_t|`` above 8x the median), x elsewhere
``all``  H = E[q q^T],          B = E[q x^T],      every token quantized

On held-out WikiText-2 test inputs the deployment error is ``err = E||V q - W x||^2``
over ``E||W x||^2``; ``err_cal`` is the same on the calibration inputs.  Per
layer, ``noise_over_signal = E||W n||^2 / E||W x||^2`` and ``outlier_frac`` /
``outlier_noise`` give the share of outlier tokens and their share of
``E||W n||^2``.

    python minitests/qir_depth.py --blocks 0,1,2,3,7,15,23,30,31
"""

from __future__ import annotations

import argparse
import time

import common as C
import torch

from qdrift.compress.pipeline import _quantize_layer
from qdrift.compress.recipe import (
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    RotationSpec,
    WeightQuantSpec,
)
from qdrift.mechanisms.qir import OUTLIER_FACTOR
from qdrift.quant.activation import fake_quantize_activations, outlier_tokens
from qdrift.quant.quantize import qrange

STEM = "qir_depth"
MECHANISMS = ("obr", "qir", "all")
# Tokens moved to the GPU at once while the moments are accumulated.
CHUNK = 8192


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=[0, 1, 2, 3, 7, 15, 23, 30, 31])
    parser.add_argument("--types", type=C.layer_types, default=["q", "o", "up", "down"])
    parser.add_argument("--sparsities", type=C.floats, default=[0.0, 0.5])
    parser.add_argument("--criterion", default="sparsegpt")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--holdout", type=int, default=32, help="WikiText-2 test chunks")
    parser.add_argument("--rows", type=int, default=256, help="0 = all rows")
    # Leaves room for the float64 moments of a down_proj next to the blocks.
    parser.set_defaults(group_size=0, reserve_gb=12.0)
    return parser.parse_args()


def recipe_for(args: argparse.Namespace, mechanism: str, sparsity: float):
    """W4A4 per-channel mid-tread, unstructured, GPTQ late half."""
    group = args.group_size if args.group_size > 0 else None
    return CompressionRecipe(
        model=args.model,
        weight=WeightQuantSpec(
            bits=args.w_bits,
            granularity=GranularitySpec("group" if group else "channel", group),
            grid="midtread",
            range=args.range,
            method="gptq",
        ),
        activation=ActivationQuantSpec(bits=args.a_bits, grid="midtread"),
        prune=PruneSpec(
            sparsity=sparsity, structure="unstructured", criterion=args.criterion
        ),
        rotation=RotationSpec(mode=args.rotation, seed=args.seed),
        compensation=CompensationSpec(
            method="obr",
            alpha=args.alpha,
            damping=args.damping,
            mechanism="obr" if mechanism == "obr" else "qir",
        ),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        pairwise_fraction=0.0,
    )


class Inputs:
    """One tapped input source, in float64 moments of ``x``, ``q`` and ``x_hat``.

    Tokens are kept on the host in float16 and streamed to ``device`` in
    chunks, so 128x2048 calibration tokens of a ``down_proj`` fit a 24 GB card.
    """

    def __init__(self, bits: int, device: torch.device):
        self.bits = bits
        self.device = device
        self.parts: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> None:
        self.parts.append(value.reshape(-1, value.shape[-1]).to("cpu", torch.float16))

    def finish(self) -> None:
        x_all = torch.cat(self.parts)
        self.parts.clear()
        hot_all = outlier_tokens(x_all, OUTLIER_FACTOR, self.device).to(self.device)
        dim, count = x_all.shape[1], x_all.shape[0]
        names = ("xx", "qq", "qx", "hh", "hx", "nn_o")
        moments = {
            name: torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
            for name in names
        }
        for start in range(0, count, CHUNK):
            x = x_all[start : start + CHUNK].to(self.device, torch.float32)
            hot = hot_all[start : start + CHUNK]
            q = fake_quantize_activations(x, self.bits, "midtread").double()
            x = x.double()
            x_hat = torch.where(hot[:, None], q, x)
            n = (q - x)[hot]
            moments["xx"] += x.t() @ x
            moments["qq"] += q.t() @ q
            moments["qx"] += q.t() @ x
            moments["hh"] += x_hat.t() @ x_hat
            moments["hx"] += x_hat.t() @ x
            moments["nn_o"] += n.t() @ n
            del x, q, x_hat, n
        # The moments wait on the host: the other input set needs the card.
        for name in names:
            setattr(self, name, moments.pop(name).div_(count).cpu())
        self.nn = self.qq - self.qx - self.qx.t() + self.xx
        self.outlier_frac = float(hot_all.float().mean())

    def system(self, mechanism: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(H, B)`` of one mechanism."""
        if mechanism == "obr":
            return self.xx, self.xx
        if mechanism == "qir":
            return self.hh, self.hx
        return self.qq, self.qx


def form(matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
    """``sum_rows left M right^T``."""
    return float(((left @ matrix) * right).sum())


def deploy_error(m: Inputs, v: torch.Tensor, w: torch.Tensor) -> float:
    """``sum_rows E(v^T q - w^T x)^2 / E(w^T x)^2`` on the inputs ``m``."""
    error = form(m.qq, v, v) - 2 * form(m.qx, v, w) + form(m.xx, w, w)
    return error / form(m.xx, w, w)


@torch.no_grad()
def measure(args, name: str, layer_type: str, weight, cal: Inputs, held: Inputs):
    """Every (sparsity, mechanism) row of one layer."""
    rows = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if rows is not None:
        weight = weight[rows]
    w = weight.double().cpu()
    noise = form(held.nn, w, w)
    layer = {
        "layer": name,
        "block": int(name.split(".")[2]),
        "type": layer_type,
        "rows": int(weight.shape[0]),
        "noise_over_signal": noise / form(held.xx, w, w),
        "outlier_frac": held.outlier_frac,
        "outlier_noise": form(held.nn_o, w, w) / max(noise, 1e-30),
    }
    out = []
    for sparsity in args.sparsities:
        for mechanism in MECHANISMS:
            hessian, cross = cal.system(mechanism)
            diff = None if mechanism == "obr" else (hessian - cross.t()).float()
            diff = None if diff is None else diff.to(weight.device)
            hessian = hessian.float().to(weight.device)
            system = C.damped(hessian, args.damping)
            result = _quantize_layer(
                recipe_for(args, mechanism, sparsity),
                qrange(args.w_bits, "midtread"),
                name,
                weight,
                weight,
                hessian,
                system,
                system,
                diff,
            )
            v = result.new_weight.to(device="cpu", dtype=torch.float64)
            out.append(
                {
                    **layer,
                    "sparsity": sparsity,
                    "mechanism": mechanism,
                    "err": deploy_error(held, v, w),
                    "err_cal": deploy_error(cal, v, w),
                }
            )
    return out


@torch.no_grad()
def main() -> None:
    """Run the minitest."""
    args = parse_args()
    path = C.result_path(
        STEM,
        args.model,
        f"rot-{args.rotation}_{C.gran_tag(args.group_size)}_{args.range}"
        f"_c{args.nsamples}x{args.seqlen}",
        f"blk{'+'.join(map(str, args.blocks))}_w{args.w_bits}a{args.a_bits}",
        C.span(args.sparsities, "s") + f"_{args.criterion}_rows{args.rows}",
        args.suffix,
    )
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    streamer.fit_resident(range(streamer.depth))
    calib = C.calibration_batches(
        tokenizer, args.nsamples, args.seqlen, args.seed, args.batch_size
    )
    hidden, kwargs = streamer.start(calib)
    chunks = C.get_loaders(
        "wikitext2", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
    )
    held, held_kwargs = streamer.start(C.batch_chunks(chunks, args.batch_size))
    rows: list[dict] = []
    for index in range(max(args.blocks) + 1):
        if index not in args.blocks:
            hidden = streamer.run(index, hidden, kwargs)
            held = streamer.run(index, held, held_kwargs)
            continue
        for layer_type in args.types:
            # One input source at a time keeps the float64 moments small.
            start = time.time()
            module = C.block_module(model, index, layer_type)
            source = C.block_module(model, index, C.INPUT_SOURCE[layer_type])
            cal = Inputs(args.a_bits, streamer.device)
            ref = Inputs(args.a_bits, streamer.device)
            streamer.run(index, hidden, kwargs, {source: cal})
            streamer.run(index, held, held_kwargs, {source: ref})
            cal.finish()
            ref.finish()
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            name = C.layer_name(index, layer_type)
            layer_rows = measure(args, name, layer_type, weight, cal, ref)
            rows.extend(layer_rows)
            by = {(r["sparsity"], r["mechanism"]): r["err"] for r in layer_rows}
            first = layer_rows[0]
            summary = " | ".join(
                f"s{s:g} "
                + " ".join(f"{mech} {by[(s, mech)]:.4f}" for mech in MECHANISMS)
                for s in args.sparsities
            )
            C.log(
                f"  {name}: noise {first['noise_over_signal']:.4f} "
                f"outliers {first['outlier_frac']:.4f} "
                f"({first['outlier_noise']:.2f} of noise) | {summary} "
                f"({time.time() - start:.0f}s)"
            )
            del weight, cal, ref
            torch.cuda.empty_cache()
        hidden = streamer.run(index, hidden, kwargs)
        held = streamer.run(index, held, held_kwargs)
    C.save_result(path, STEM, vars(args), errors=rows)
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
