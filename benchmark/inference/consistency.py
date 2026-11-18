# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a real W4A4 checkpoint against a dense fake-quant reference.

The reference uses the same checkpoint codes, scales, and sparse mask, the same
CUDA activation quantizer, and an exact INT8 integer matmul of the INT4 codes.
It therefore checks the real fused W4A4 kernel and KV cache without relying on
a second quantized GEMM implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.eval.runtime import _unpack_mask, make_layer_case
from benchmark.inference.checkpoint import build_checkpoint_model
from benchmark.inference.config import ModelSpec
from benchmark.inference.quant import unpack_i4
from benchmark.kernels.lib.int8_linear import build_int8_weight
from benchmark.kernels.lib.w4a4 import quarot_module
from qdrift.compress.recipe import CompressionRecipe
from qdrift.models.backbone import resolve_model_source

__all__ = ["main"]


class _FakeW4A4Linear(nn.Module):
    """Dense INT4 fake-quant reference for one checkpoint layer."""

    def __init__(
        self, config, tensors, name, regime, recipe, device, allow_sparse=True
    ):
        super().__init__()
        case = make_layer_case(config, tensors, name)
        if recipe.activation.bits != 4 or recipe.weight.bits != 4:
            raise ValueError("consistency check expects W4A4")
        kind = recipe.weight.granularity.kind
        if kind not in ("tensor", "channel"):
            raise ValueError("consistency check expects channel/tensor scales")
        self.in_features = int(case.in_features)
        self.out_features = int(case.out_features)
        layer_meta = config.get("layers", {}).get(name, {})
        layer_structure = layer_meta.get(
            "structure",
            recipe.prune.structure,
        )
        self.sparse = (
            regime == "joint"
            and str(layer_structure) in ("2:4_pairwise", "4:8_pairwise")
        )
        self.clip_ratio = float(recipe.activation.clip_ratio)
        mask = (
            _unpack_mask(case.mask, case.in_features)
            if case.mask is not None
            else None
        )
        group_size = (
            int(recipe.weight.granularity.group_size or case.in_features)
            if kind == "group"
            else case.in_features
        )
        int8 = build_int8_weight(
            recipe.weight.bits,
            recipe.weight.grid,
            case.packed,
            case.scales,
            (case.out_features, case.in_features),
            group_size,
            kind,
            mask,
            sparse=self.sparse,
        )
        codes = int8.codes
        self.padded_in_features = (
            ((self.in_features + 255) // 256) * 256
            if self.sparse
            else self.in_features
        )
        if self.padded_in_features != self.in_features:
            codes = torch.nn.functional.pad(
                codes, (0, self.padded_in_features - self.in_features)
            )
        self.codes_t = codes.t().contiguous().to(device)
        self.weight_scale = int8.scales[:, 0].float().half().to(device)
        del codes, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1])
        if self.padded_in_features != x2d.shape[-1]:
            x2d = F.pad(x2d, (0, self.padded_in_features - x2d.shape[-1]))
        packed, a_scale = quarot_module().sym_quant_dynamic(
            x2d.half().contiguous(), self.clip_ratio
        )
        x_codes = unpack_i4(packed)[:, : self.padded_in_features]
        rows = x_codes.shape[0]
        target = max(17, ((rows + 15) // 16) * 16)
        if target != rows:
            x_codes = F.pad(x_codes, (0, 0, 0, target - rows))
        acc = torch._int_mm(x_codes.contiguous(), self.codes_t)[:rows]
        if self.sparse:
            out = (
                acc.float()
                * a_scale[:rows].float()[:, None]
                * self.weight_scale[None, :].float()
            ).half()
        else:
            out = acc.half() * (
                a_scale[:rows, None].half() * self.weight_scale[None, :].half()
            )
        out = out[..., : self.out_features]
        return out.reshape(*shape, -1)

    def storage_bytes(self) -> int:
        return (
            self.codes_t.numel() * self.codes_t.element_size()
            + self.weight_scale.numel() * self.weight_scale.element_size()
        )


class _ReferenceKVCache:
    """Float32 dequant + softmax reference for quantized decode attention."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decode_attention(self, layer, query, length):
        keys = self.inner.k[layer, :, :, :length]
        values = self.inner.v[layer, :, :, :length]
        if self.inner.bits == 4:
            keys = unpack_i4(keys)
            values = unpack_i4(values)
        keys = keys.float() * self.inner.ks[layer, :, :, :length, None].float()
        values = values.float() * self.inner.vs[layer, :, :, :length, None].float()
        group = query.shape[1] // self.inner.n_kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = (query.float() @ keys.transpose(-1, -2)) / (
            query.shape[-1] ** 0.5
        )
        probs = torch.softmax(scores, dim=-1)
        return (probs @ values).half()


def _metrics(name, reference, actual):
    ref = reference.float()
    out = actual.float()
    diff = (out - ref).abs()
    cosine = F.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
    top1 = (ref.argmax(-1) == out.argmax(-1)).float().mean().item()
    ref5 = ref.topk(5, dim=-1).indices.reshape(-1, 5).tolist()
    out5 = out.topk(5, dim=-1).indices.reshape(-1, 5).tolist()
    overlap = sum(len(set(a) & set(b)) for a, b in zip(ref5, out5))
    return {
        "name": name,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cosine": cosine,
        "top1_agreement": top1,
        "top5_overlap": overlap / (len(ref5) * 5.0),
    }


@torch.inference_mode()
def _run(
    recipe,
    checkpoint,
    regime,
    manifest,
    device,
    batch,
    prefill_len,
    decode_steps,
    kv_bits,
    fused_kv,
    input_ids,
    next_ids,
    reference,
):
    factory = None
    if reference:
        factory = _FakeW4A4Linear
    model, _ = build_checkpoint_model(
        recipe,
        checkpoint,
        regime,
        manifest,
        device,
        prefill_len,
        decode_steps,
        linear_factory=factory,
    )
    max_len = prefill_len + decode_steps + 1
    cache = model.new_cache(
        batch, max_len, bits=kv_bits, fused=fused_kv if not reference else False
    )
    if reference and fused_kv:
        cache = _ReferenceKVCache(cache)
    prefill = model.prefill(input_ids, cache).float().cpu()
    decode = []
    for _ in range(decode_steps):
        decode.append(model.decode(next_ids, cache).float().cpu())
    del model, cache
    torch.cuda.empty_cache()
    return prefill, decode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regime", choices=("quant", "joint"), default="joint")
    parser.add_argument("--manifest", default="modelzoo/manifest.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, default=4, choices=(16, 8, 4))
    parser.add_argument("--fused-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    recipe = CompressionRecipe.from_config(config)
    if recipe.activation.bits != 4 or recipe.weight.bits != 4:
        raise SystemExit("consistency check expects a W4A4 checkpoint")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = resolve_model_source(recipe.model, args.manifest)
    spec = ModelSpec.from_dir(source)
    input_ids = torch.randint(
        0, spec.vocab_size, (args.batch, args.prefill_len), device=args.device
    )
    next_ids = torch.full(
        (args.batch, 1),
        min(100, spec.vocab_size - 1),
        dtype=torch.long,
        device=args.device,
    )

    started = time.time()
    fake_prefill, fake_decode = _run(
        recipe, checkpoint, args.regime, args.manifest, args.device,
        args.batch, args.prefill_len, args.decode_steps, args.kv_bits,
        args.fused_kv, input_ids, next_ids, reference=True,
    )
    real_prefill, real_decode = _run(
        recipe, checkpoint, args.regime, args.manifest, args.device,
        args.batch, args.prefill_len, args.decode_steps, args.kv_bits,
        args.fused_kv, input_ids, next_ids, reference=False,
    )

    rows = [_metrics("prefill", fake_prefill, real_prefill)]
    rows += [
        _metrics(f"decode[{i}]", ref, out)
        for i, (ref, out) in enumerate(zip(fake_decode, real_decode))
    ]
    summary = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("max_abs", "mean_abs", "cosine", "top1_agreement", "top5_overlap")
    }
    payload = {
        "format": "qdrift-inference-consistency",
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.short_name,
        "config": {
            "checkpoint": str(checkpoint),
            "regime": args.regime,
            "kv_bits": args.kv_bits,
            "batch": args.batch,
            "prefill_len": args.prefill_len,
            "decode_steps": args.decode_steps,
            "fused_kv": args.fused_kv,
            "seed": args.seed,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "rows": rows,
        "summary": summary,
        "elapsed_s": time.time() - started,
    }
    output = Path(args.output) if args.output else (
        Path("results") / "inference"
        / f"consistency_{spec.short_name}_{args.regime}_kv{args.kv_bits}_{'fused' if args.fused_kv else 'nofused'}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'step':>12} {'max_abs':>10} {'mean_abs':>10} {'cosine':>9} {'top1':>7} {'top5':>7}")
    for row in rows:
        print(
            f"{row['name']:>12} {row['max_abs']:>10.5f} {row['mean_abs']:>10.5f} "
            f"{row['cosine']:>9.6f} {row['top1_agreement']:>7.4f} "
            f"{row['top5_overlap']:>7.4f}"
        )
    print(
        f"{'mean':>12} {summary['max_abs']:>10.5f} {summary['mean_abs']:>10.5f} "
        f"{summary['cosine']:>9.6f} {summary['top1_agreement']:>7.4f} "
        f"{summary['top5_overlap']:>7.4f}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
