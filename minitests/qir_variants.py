# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: which part of the quantized-input statistics generalizes?

Only one linear layer is compressed at a time, on its full-precision input
``x`` (dense prefix).  At deployment it sees ``q = Q_a(x) = x + n`` with the
runtime's per-token activation quantizer.  Every variant runs OBR's solver
on a system ``H`` and a cross matrix ``B`` (row target ``b = B w``, i.e.
``D = H - B^T``), fitted on calibration inputs:

``obr``          H = E[x x^T],  B = E[x x^T]            (noise invisible)
``qir``          H = E[q q^T],  B = E[q x^T]            (exact, all tokens)
``qir_half``     the average of the two systems above   (shrinkage)
``qir_diag``     H = E[x x^T] + diag E[n n^T],  B = E[x x^T] + diag E[n x^T]
``qir_outlier``  the exact terms of the outlier tokens only
``qir_both``     outlier tokens exact, the other tokens' noise diagonal

Every ``H`` is a sum of PSD terms.

Outlier tokens are those whose ``max|x_t|`` exceeds ``--outlier-factor``
times the median over their batch: the massive-activation tokens whose large
quantization step makes their noise systematic.  On held-out C4 inputs the
deployment error splits as

``err``     E||V q - W x||^2        ``err_w``  E||(V - W) x||^2
``err_a``   E||V n||^2              all over E||W x||^2,

``err_cal`` is ``err`` on the calibration moments, and per layer
``outlier_frac`` / ``outlier_noise`` give the share of outlier tokens and
their share of ``E||W n||^2``.

    python minitests/qir_variants.py --blocks 0,1,2,3,7,15,23,30,31
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
from qdrift.quant.activation import fake_quantize_activations
from qdrift.quant.quantize import qrange

STEM = "qir_variants"
VARIANTS = ("obr", "qir", "qir_half", "qir_diag", "qir_outlier", "qir_both")


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
    parser.add_argument("--outlier-factor", type=float, default=8.0)
    parser.add_argument("--holdout", type=int, default=32, help="C4 chunks")
    parser.add_argument("--rows", type=int, default=256, help="0 = all rows")
    parser.set_defaults(group_size=0)
    return parser.parse_args()


def recipe_for(args: argparse.Namespace, variant: str, sparsity: float):
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
            mechanism="obr" if variant == "obr" else "qir",
        ),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        pairwise_fraction=0.0,
    )


class Moments:
    """``E[x x^T]``, ``E[q q^T]``, ``E[q x^T]`` over all and outlier tokens."""

    NAMES = ("xx", "qq", "qx", "xx_o", "qq_o", "qx_o")

    def __init__(self, dim: int, device: torch.device, bits: int, factor: float):
        for name in self.NAMES:
            setattr(self, name, torch.zeros(dim, dim, device=device, dtype=torch.float64))
        self.bits = bits
        self.factor = factor
        self.count = 0
        self.outliers = 0

    def __call__(self, value: torch.Tensor) -> None:
        x = value.reshape(-1, value.shape[-1]).float()
        q = fake_quantize_activations(x, self.bits, "midtread").double()
        amax = x.abs().amax(dim=1)
        hot = amax > self.factor * amax.median()
        x = x.double()
        self.xx.addmm_(x.t(), x)
        self.qq.addmm_(q.t(), q)
        self.qx.addmm_(q.t(), x)
        xo, qo = x[hot], q[hot]
        self.xx_o.addmm_(xo.t(), xo)
        self.qq_o.addmm_(qo.t(), qo)
        self.qx_o.addmm_(qo.t(), xo)
        self.count += x.shape[0]
        self.outliers += int(hot.sum())

    def finish(self) -> None:
        for name in self.NAMES:
            setattr(self, name, getattr(self, name) / max(self.count, 1))
        self.nn = self.qq - self.qx - self.qx.t() + self.xx
        self.nn_o = self.qq_o - self.qx_o - self.qx_o.t() + self.xx_o
        self.nx = self.qx - self.xx
        self.nx_o = self.qx_o - self.xx_o

    def system(self, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(H, B)`` of one variant."""
        if variant == "obr":
            return self.xx, self.xx
        if variant == "qir":
            return self.qq, self.qx
        if variant == "qir_half":
            return 0.5 * (self.xx + self.qq), 0.5 * (self.xx + self.qx)
        if variant == "qir_diag":
            return self.xx + diag(self.nn), self.xx + diag(self.nx)
        out_h, out_b = self.qq_o - self.xx_o, self.qx_o - self.xx_o
        if variant == "qir_outlier":
            return self.xx + out_h, self.xx + out_b
        if variant == "qir_both":
            return (
                self.xx + out_h + diag(self.nn - self.nn_o),
                self.xx + out_b + diag(self.nx - self.nx_o),
            )
        raise ValueError(variant)


def diag(matrix: torch.Tensor) -> torch.Tensor:
    """The diagonal part of ``matrix``."""
    return torch.diag(torch.diagonal(matrix))


def form(matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
    """``sum_rows left M right^T``."""
    return float(((left @ matrix) * right).sum())


def deploy_error(m: Moments, v: torch.Tensor, w: torch.Tensor) -> float:
    """``sum_rows E(v^T q - w^T x)^2`` under the moments ``m``."""
    return form(m.qq, v, v) - 2 * form(m.qx, v, w) + form(m.xx, w, w)


@torch.no_grad()
def measure(args, name: str, layer_type: str, weight, cal: Moments, held: Moments):
    """Every (sparsity, variant) row of one layer."""
    rows = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if rows is not None:
        weight = weight[rows]
    w = weight.double()
    ref = form(held.xx, w, w)
    layer = {
        "layer": name,
        "block": int(name.split(".")[2]),
        "type": layer_type,
        "rows": int(weight.shape[0]),
        "noise_over_signal": form(held.nn, w, w) / ref,
        "outlier_frac": held.outliers / max(held.count, 1),
        "outlier_noise": form(held.nn_o, w, w) / max(form(held.nn, w, w), 1e-30),
    }
    out = []
    for sparsity in args.sparsities:
        for variant in VARIANTS:
            hessian, cross = cal.system(variant)
            hessian = 0.5 * (hessian + hessian.t())
            diff = None if variant == "obr" else (hessian - cross.t()).float()
            hessian = hessian.float()
            system = C.damped(hessian, args.damping)
            result = _quantize_layer(
                recipe_for(args, variant, sparsity),
                qrange(args.w_bits, "midtread"),
                name,
                weight,
                weight,
                hessian,
                system,
                system,
                diff,
            )
            v = result.new_weight.to(device=weight.device, dtype=torch.float64)
            delta = v - w
            out.append(
                {
                    **layer,
                    "sparsity": sparsity,
                    "mechanism": variant,
                    "err": deploy_error(held, v, w) / ref,
                    "err_w": form(held.xx, delta, delta) / ref,
                    "err_a": form(held.nn, v, v) / ref,
                    "err_cal": deploy_error(cal, v, w) / form(cal.xx, w, w),
                    "energy": float(v.pow(2).sum() / w.pow(2).sum()),
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
        f"rot-{args.rotation}_{C.gran_tag(args.group_size)}_{args.range}",
        f"blk{'+'.join(map(str, args.blocks))}_w{args.w_bits}a{args.a_bits}",
        C.span(args.sparsities, "s")
        + f"_{args.criterion}_of{args.outlier_factor:g}_rows{args.rows}",
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
        "c4", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
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
            dim = source.in_features
            cal = Moments(dim, streamer.device, args.a_bits, args.outlier_factor)
            ref = Moments(dim, streamer.device, args.a_bits, args.outlier_factor)
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
                f"s{s:g} " + " ".join(f"{v} {by[(s, v)]:.4f}" for v in VARIANTS)
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
