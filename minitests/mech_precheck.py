# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minitest: can activation-noise damping matter at all?  (wiener pre-check)

On the six diagnostic layers (``q/o/up/down`` of the middle block, ``down``
of block 1 and of the last block), with the calibration Hessian ``H`` and
``sigma_a^2`` (the evaluation-time A4 noise on the same inputs):

    f_tail = sum_rows sum_{i: h_i < sigma_a^2} (u_i^T w)^2 / sum_rows |w|^2
    bound  = f_tail * sigma_a^2 * |W|^2 / E_obr

``u_i, h_i`` the eigenpairs of ``H`` and ``E_obr`` the measured OBR layer
error with A4-quantized held-out inputs, ``sum_rows E||w_hat^T Q_a(x) -
w^T x||^2``.  ``bound`` caps the share of that error the Wiener shrinkage
can remove.  The wiener error on the same rows is reported for reference.

Weights are W4 per-channel mid-tread (clip search), 50% unstructured
magnitude mask, GPTQ late half; activations A4 per token, mid-tread.  The
calibration is the pipeline's (WikiText-2, ``--nsamples x --seqlen``,
``--seed`` for the rotation and the draw) on full-precision layer inputs;
held-out inputs are ``--holdout`` C4 validation chunks.

    python minitests/mech_precheck.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import common as C
import torch

from qdrift.compress import pipeline
from qdrift.compress.recipe import (
    ActivationQuantSpec,
    CompensationSpec,
    CompressionRecipe,
    GranularitySpec,
    PruneSpec,
    WeightQuantSpec,
)
from qdrift.compress.runs import RESULTS_V2, claim, git_rev, run_name
from qdrift.quant.activation import activation_step, fake_quantize_activations
from qdrift.quant.quantize import qrange

STEM = "precheck"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.common_arguments(parser)
    parser.add_argument("--middle", type=int, default=15)
    parser.add_argument("--holdout", type=int, default=64)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--rows", type=int, default=256, help="0 = all rows")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(nsamples=128, seqlen=2048, seed=0)
    return parser.parse_args()


def recipe_for(args: argparse.Namespace, mechanism: str) -> CompressionRecipe:
    """The W4A4 50% unstructured recipe of one mechanism."""
    return CompressionRecipe(
        weight=WeightQuantSpec(
            bits=args.w_bits,
            granularity=GranularitySpec("channel", None),
            grid="midtread",
            range="clip",
            method="gptq",
        ),
        activation=ActivationQuantSpec(bits=args.a_bits, grid="midtread"),
        prune=PruneSpec(sparsity=0.5, structure="unstructured", criterion="magnitude"),
        compensation=CompensationSpec(method="obr", mechanism=mechanism),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        pairwise_fraction=0.0,
    )


class NoiseTap:
    """Accumulate ``H = E[z z^T]`` and ``E[step^2]`` of the A4 quantizer."""

    def __init__(self, dim: int, device: torch.device, bits: int) -> None:
        self.gram = C.Gram(dim, device)
        self.bits = bits
        self.step2 = 0.0
        self.count = 0

    def __call__(self, value: torch.Tensor) -> None:
        flat = value.reshape(-1, value.shape[-1]).float()
        self.gram(flat)
        step = activation_step(flat, self.bits, "midtread")
        self.step2 += float(step.double().pow(2).sum())
        self.count += step.numel()

    @property
    def act_noise(self) -> float:
        return self.step2 / max(self.count, 1) / 12.0


class HeldTap:
    """Held-out ``E[q q^T]``, ``E[q x^T]``, ``E[x x^T]`` with ``q = Q_a(x)``."""

    def __init__(self, dim: int, device: torch.device, bits: int) -> None:
        shape = (dim, dim)
        self.qq = torch.zeros(shape, device=device, dtype=torch.float64)
        self.qx = torch.zeros_like(self.qq)
        self.xx = torch.zeros_like(self.qq)
        self.bits = bits
        self.count = 0

    def __call__(self, value: torch.Tensor) -> None:
        x = value.reshape(-1, value.shape[-1]).float()
        q = fake_quantize_activations(x, self.bits, "midtread")
        x, q = x.double(), q.double()
        self.qq.addmm_(q.t(), q)
        self.qx.addmm_(q.t(), x)
        self.xx.addmm_(x.t(), x)
        self.count += x.shape[0]

    def error(self, new: torch.Tensor, weight: torch.Tensor) -> tuple[float, float]:
        """``sum_rows E||new q - weight x||^2`` and the weight-only error."""
        new, weight = new.double(), weight.double()
        n = max(self.count, 1)
        joint = (
            ((new @ self.qq) * new).sum()
            - 2 * ((new @ self.qx) * weight).sum()
            + ((weight @ self.xx) * weight).sum()
        ) / n
        delta = new - weight
        return float(joint), float(((delta @ self.xx) * delta).sum() / n)


@torch.no_grad()
def measure(args, name, weight, tap: NoiseTap, held: HeldTap) -> dict:
    """f_tail, bound, and the OBR / wiener errors of one layer."""
    rows = C.row_subset(weight.shape[0], args.rows, args.seed, weight.device)
    if rows is not None:
        weight = weight[rows]
    hessian, system = tap.gram.hessians(args.damping)
    sigma = tap.act_noise
    eigval, eigvec = torch.linalg.eigh(hessian.double())
    tail = eigval < sigma
    proj = weight.double() @ eigvec
    norm2 = float(weight.double().pow(2).sum())
    f_tail = float(proj[:, tail].pow(2).sum()) / norm2
    mean_diag = float(torch.diagonal(hessian).mean())
    out = {
        "layer": name,
        "rows": int(weight.shape[0]),
        "act_noise": sigma,
        "act_noise_over_mean_diag": sigma / mean_diag,
        "obr_lambda_over_mean_diag": args.damping,
        "tail_eigs": int(tail.sum()),
        "f_tail": f_tail,
    }
    for mechanism in ("obr", "wiener"):
        recipe = recipe_for(args, mechanism)
        result = pipeline._quantize_layer(
            recipe,
            qrange(args.w_bits, "midtread"),
            name,
            weight,
            weight,
            hessian,
            system,
            system,
            None,
            act_noise=sigma,
        )
        joint, weight_only = held.error(
            result.new_weight.to(weight.device).float(), weight
        )
        out[f"err_{mechanism}"] = joint
        out[f"err_weight_{mechanism}"] = weight_only
        out[f"meta_{mechanism}"] = {
            k: v
            for k, v in result.meta.items()
            if k in ("lambda", "floor", "mechanism")
        }
    out["bound"] = f_tail * sigma * norm2 / out["err_obr"]
    out["wiener_gain"] = 1.0 - out["err_wiener"] / out["err_obr"]
    return out


@torch.no_grad()
def main() -> None:
    """Run the pre-check."""
    args = parse_args()
    path = RESULTS_V2 / STEM / f"{run_name(recipe_for(args, 'obr'))}.json"
    claim([path], args.overwrite)
    C.log(f"output {path}")
    model, tokenizer = C.load_model(args.model, args.rotation, args.seed, args.device)
    streamer = C.BlockStreamer(model, args.device, args.reserve_gb)
    layers = [
        (args.middle, "q"),
        (args.middle, "o"),
        (args.middle, "up"),
        (args.middle, "down"),
        (1, "down"),
        (streamer.depth - 1, "down"),
    ]
    blocks = sorted({index for index, _ in layers})
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
    for index in range(max(blocks) + 1):
        if index not in blocks:
            hidden = streamer.run(index, hidden, kwargs)
            held = streamer.run(index, held, held_kwargs)
            continue
        types = [t for i, t in layers if i == index]
        sources = sorted({C.INPUT_SOURCE[t] for t in types})
        taps, held_taps = {}, {}
        calib_hooks, held_hooks = {}, {}
        for source in sources:
            module = C.block_module(model, index, source)
            taps[source] = NoiseTap(module.in_features, streamer.device, args.a_bits)
            held_taps[source] = HeldTap(
                module.in_features, streamer.device, args.a_bits
            )
            calib_hooks[module] = taps[source]
            held_hooks[module] = held_taps[source]
        hidden = streamer.run(index, hidden, kwargs, calib_hooks)
        held = streamer.run(index, held, held_kwargs, held_hooks)
        for layer_type in types:
            start = time.time()
            source = C.INPUT_SOURCE[layer_type]
            module = C.block_module(model, index, layer_type)
            weight = module.weight.data.to(device=streamer.device, dtype=torch.float32)
            name = C.layer_name(index, layer_type)
            rows.append(measure(args, name, weight, taps[source], held_taps[source]))
            r = rows[-1]
            C.log(
                f"  {name}: sigma^2/mean diag {r['act_noise_over_mean_diag']:.4f} "
                f"f_tail {r['f_tail']:.4f} bound {r['bound']:.4f} "
                f"wiener gain {r['wiener_gain']:+.4f} ({time.time() - start:.0f}s)"
            )
            del weight
            torch.cuda.empty_cache()
        del taps, held_taps, calib_hooks, held_hooks
        torch.cuda.empty_cache()
    passed = sum(r["bound"] >= 0.02 for r in rows)
    payload = {
        "test": STEM,
        "git": git_rev(),
        "config": {k: v for k, v in vars(args).items()},
        "rows": rows,
        "layers_bound_ge_2pct": passed,
        "stop": passed < 3,
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    C.log(f"bound >= 2% on {passed}/6 layers; wrote {path}")


if __name__ == "__main__":
    main()
