# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: pruning compensation amplifies activation quantization noise.

Only one linear layer is compressed at a time, on its full-precision input
``x`` (dense prefix), so ``x`` is also the input the target output is
computed on.  At deployment the layer sees ``q = Q_a(x) = x + n``, with the
runtime's per-token activation quantizer.  For each sparsity ``s`` the layer
is compressed by

``obr``  OBR on ``H = E[x x^T]`` (the noise is invisible to the solve);
``qir``  OBR's solver on ``E[q q^T]`` and ``D = E[(q - x) q^T]``, i.e. the
         fit ``min_v E(v^T q - w^T x)^2`` of the qir mechanism,

and on held-out C4 inputs the deployment error splits as

``err``     E||V q - W x||^2                       total
``err_w``   E||(V - W) x||^2                       weight error
``err_a``   E||V n||^2                             activation noise passed by V
``cross``   err - err_w - err_a

all over ``E||W x||^2``.  ``gain_a = E||V n||^2 / E||W n||^2`` is how much
the compressed weights amplify the activation noise relative to the dense
ones, and ``energy = |V|^2 / |W|^2``.  Every quantity follows from the
held-out moments ``E[x x^T]``, ``E[q q^T]``, ``E[q x^T]``.  ``err_cal`` is
``err`` on the calibration moments.  Per layer, ``ls_held`` / ``ls_cal`` give
the error of the unconstrained fit ``argmin_v E(v^T q - w^T x)^2`` (no
pruning, no weight quantization, fitted on calibration) over that of ``W``.

    python minitests/noise_amplification.py --sparsities 0,0.3,0.5,0.7
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

STEM = "noise_amplification"
MECHANISMS = ("obr", "qir")


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--blocks", type=C.ints, default=[1, 15, 31])
    parser.add_argument("--types", type=C.layer_types, default=["q", "o", "up", "down"])
    parser.add_argument("--sparsities", type=C.floats, default=[0.0, 0.3, 0.5, 0.7])
    parser.add_argument("--criterion", default="sparsegpt")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--holdout", type=int, default=32, help="C4 chunks")
    parser.add_argument("--rows", type=int, default=256, help="0 = all rows")
    parser.set_defaults(group_size=0)
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
            method="obr", alpha=args.alpha, damping=args.damping, mechanism=mechanism
        ),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        pairwise_fraction=0.0,
    )


class Moments:
    """``E[x x^T]``, ``E[q q^T]``, ``E[q x^T]`` with ``q = Q_a(x)``, float64."""

    def __init__(self, dim: int, device: torch.device, bits: int) -> None:
        self.xx = torch.zeros(dim, dim, device=device, dtype=torch.float64)
        self.qq = torch.zeros_like(self.xx)
        self.qx = torch.zeros_like(self.xx)
        self.bits = bits
        self.count = 0

    def __call__(self, value: torch.Tensor) -> None:
        x = value.reshape(-1, value.shape[-1]).float()
        q = fake_quantize_activations(x, self.bits, "midtread").double()
        x = x.double()
        self.xx.addmm_(x.t(), x)
        self.qq.addmm_(q.t(), q)
        self.qx.addmm_(q.t(), x)
        self.count += x.shape[0]

    def finish(self) -> None:
        for name in ("xx", "qq", "qx"):
            setattr(self, name, getattr(self, name) / max(self.count, 1))
        self.nn = self.qq - self.qx - self.qx.t() + self.xx


def form(matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
    """``sum_rows left M right^T``."""
    return float(((left @ matrix) * right).sum())


def deploy_error(m: Moments, v: torch.Tensor, w: torch.Tensor) -> float:
    """``sum_rows E(v^T q - w^T x)^2`` under the moments ``m``."""
    return form(m.qq, v, v) - 2 * form(m.qx, v, w) + form(m.xx, w, w)


@torch.no_grad()
def measure(args, name: str, weight: torch.Tensor, cal: Moments, held: Moments):
    """Every (sparsity, mechanism) row of one layer."""
    rows = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if rows is not None:
        weight = weight[rows]
    systems = {
        "obr": (cal.xx.float(), None),
        "qir": (cal.qq.float(), (cal.qq - cal.qx.t()).float()),
    }
    w = weight.double()
    ref = form(held.xx, w, w)
    noise_dense = form(held.nn, w, w)
    eye = torch.eye(cal.qq.shape[0], device=cal.qq.device, dtype=cal.qq.dtype)
    ridge = 1e-6 * torch.diagonal(cal.qq).mean()
    v_ls = torch.linalg.solve(cal.qq + ridge * eye, cal.qx @ w.t()).t()
    ls = {
        "ls_held": deploy_error(held, v_ls, w) / deploy_error(held, w, w),
        "ls_cal": deploy_error(cal, v_ls, w) / deploy_error(cal, w, w),
    }
    out = []
    for sparsity in args.sparsities:
        for mechanism in MECHANISMS:
            hessian, diff = systems[mechanism]
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
            v = result.new_weight.to(device=weight.device, dtype=torch.float64)
            delta = v - w
            err = form(held.qq, v, v) - 2 * form(held.qx, v, w) + ref
            err_w = form(held.xx, delta, delta)
            err_a = form(held.nn, v, v)
            out.append(
                {
                    "layer": name,
                    "type": name.split(".")[-1].split("_")[0],
                    "sparsity": sparsity,
                    "mechanism": mechanism,
                    "rows": int(weight.shape[0]),
                    "err": err / ref,
                    "err_w": err_w / ref,
                    "err_a": err_a / ref,
                    "cross": (err - err_w - err_a) / ref,
                    "gain_a": err_a / noise_dense,
                    "energy": float(v.pow(2).sum() / w.pow(2).sum()),
                    "noise_over_signal": noise_dense / ref,
                    "err_cal": deploy_error(cal, v, w) / form(cal.xx, w, w),
                    **ls,
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
        "c4", tokenizer, nsamples=args.holdout, seqlen=args.seqlen, eval_mode=True
    )
    held, held_kwargs = streamer.start(C.batch_chunks(chunks, args.batch_size))
    rows: list[dict] = []
    for index in range(max(args.blocks) + 1):
        if index not in args.blocks:
            hidden = streamer.run(index, hidden, kwargs)
            held = streamer.run(index, held, held_kwargs)
            continue
        sources = sorted({C.INPUT_SOURCE[t] for t in args.types})
        cal_taps, held_taps, cal_hooks, held_hooks = {}, {}, {}, {}
        for source in sources:
            module = C.block_module(model, index, source)
            dim = module.in_features
            cal_taps[source] = Moments(dim, streamer.device, args.a_bits)
            held_taps[source] = Moments(dim, streamer.device, args.a_bits)
            cal_hooks[module] = cal_taps[source]
            held_hooks[module] = held_taps[source]
        hidden = streamer.run(index, hidden, kwargs, cal_hooks)
        held = streamer.run(index, held, held_kwargs, held_hooks)
        for tap in (*cal_taps.values(), *held_taps.values()):
            tap.finish()
        for layer_type in args.types:
            start = time.time()
            source = C.INPUT_SOURCE[layer_type]
            module = C.block_module(model, index, layer_type)
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            name = C.layer_name(index, layer_type)
            layer_rows = measure(args, name, weight, cal_taps[source], held_taps[source])
            rows.extend(layer_rows)
            by = {(r["sparsity"], r["mechanism"]): r for r in layer_rows}
            summary = " ".join(
                f"s{s:g}: obr {by[(s, 'obr')]['err']:.4f}"
                f"/{by[(s, 'obr')]['err_cal']:.4f}"
                f" qir {by[(s, 'qir')]['err']:.4f}"
                f"/{by[(s, 'qir')]['err_cal']:.4f}"
                for s in args.sparsities
            )
            first = layer_rows[0]
            summary += f" ls held {first['ls_held']:.3f} cal {first['ls_cal']:.3f}"
            C.log(f"  {name}: {summary} ({time.time() - start:.0f}s)")
            del weight
            torch.cuda.empty_cache()
        del cal_taps, held_taps, cal_hooks, held_hooks
        torch.cuda.empty_cache()
    C.save_result(path, STEM, vars(args), errors=rows)
    C.log(f"wrote {path}")


if __name__ == "__main__":
    main()
