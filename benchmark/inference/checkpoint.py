# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed fast inference benchmark.

This backend reuses the checkpoint cases from ``benchmark.eval.runtime`` with
the small graph-friendly runtime in this package.  It is intended for
checkpoints produced with ``--rotation none``; the HF-backed engine remains the
default for rotated checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file

from benchmark.eval.runtime import (
    CompressedLinear,
    checkpoint_stats,
    make_regime_case,
)
from benchmark.inference.config import BenchConfig, ModelSpec
from benchmark.inference.linear import DenseLinear
from benchmark.inference.measure import measure_e2e
from benchmark.inference.model import (
    Attention,
    Block,
    CausalLM,
    MLP,
    RMSNorm,
    RotaryEmbedding,
)
from benchmark.inference.weights import SafetensorStore
from qdrift.compress.recipe import CompressionRecipe
from qdrift.models.backbone import resolve_model_source

__all__ = ["build_checkpoint_model", "evaluate_inference"]

_REGIME_LABELS = {
    "dense": "dense",
    "sparse": "sparse_fp16",
    "quant": "int4_marlin",
    "joint": "joint_sparse_int4",
}


def _linear(
    config: dict,
    tensors: dict[str, torch.Tensor],
    name: str,
    regime: str,
    recipe: CompressionRecipe,
    device: str,
    allow_sparse: bool = True,
) -> CompressedLinear:
    case = make_regime_case(config, tensors, name, regime, device, allow_sparse)
    original = SimpleNamespace(
        in_features=case.in_features,
        out_features=case.out_features,
        bias=None,
    )
    return CompressedLinear(original, case, regime, recipe)


@torch.no_grad()
def build_checkpoint_model(
    recipe: CompressionRecipe,
    checkpoint: str | Path,
    regime: str,
    manifest_path: str,
    device: str,
    prefill_len: int,
    decode_steps: int,
    linear_factory=None,
    allow_sparse: bool = True,
) -> tuple[CausalLM, dict]:
    if recipe.rotation.mode != "none":
        raise ValueError(
            "the inference engine requires --rotation none; use --engine hf "
            "for rotated checkpoints"
        )
    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    source = resolve_model_source(recipe.model, manifest_path)
    spec = ModelSpec.from_dir(source)
    store = SafetensorStore(source)
    tensors = load_file(str(checkpoint))
    if any(key.endswith(".weight_smooth_scale") for key in tensors):
        raise ValueError(
            "checkpoint uses smoothing, which is no longer supported; "
            "recompress it with the current pipeline"
        )
    make_linear_layer = linear_factory or _linear

    embed_weight = store.get("model.embed_tokens.weight", device).half()
    embed_tokens = nn.Embedding(
        spec.vocab_size, spec.hidden_size, device=device, dtype=torch.float16
    )
    embed_tokens.weight = nn.Parameter(embed_weight, requires_grad=False)

    max_len = prefill_len + decode_steps + 1
    rotary = RotaryEmbedding(spec.head_dim, spec.rope_theta, max_len, device)

    layers: list[Block] = []
    for index in range(spec.num_hidden_layers):
        prefix = f"model.layers.{index}"
        attn_prefix = f"{prefix}.self_attn"
        q_proj = make_linear_layer(config, tensors, f"{attn_prefix}.q_proj", regime, recipe, device, allow_sparse)
        k_proj = make_linear_layer(config, tensors, f"{attn_prefix}.k_proj", regime, recipe, device, allow_sparse)
        v_proj = make_linear_layer(config, tensors, f"{attn_prefix}.v_proj", regime, recipe, device, allow_sparse)
        o_proj = make_linear_layer(config, tensors, f"{attn_prefix}.o_proj", regime, recipe, device, allow_sparse)

        mlp_prefix = f"{prefix}.mlp"
        gate_proj = make_linear_layer(config, tensors, f"{mlp_prefix}.gate_proj", regime, recipe, device, allow_sparse)
        up_proj = make_linear_layer(config, tensors, f"{mlp_prefix}.up_proj", regime, recipe, device, allow_sparse)
        down_proj = make_linear_layer(config, tensors, f"{mlp_prefix}.down_proj", regime, recipe, device, allow_sparse)

        input_norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps, device)
        input_norm.weight = nn.Parameter(
            store.get(f"{prefix}.input_layernorm.weight", device).half(),
            requires_grad=False,
        )
        post_norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps, device)
        post_norm.weight = nn.Parameter(
            store.get(f"{prefix}.post_attention_layernorm.weight", device).half(),
            requires_grad=False,
        )

        attn = Attention(spec, index, q_proj, k_proj, v_proj, o_proj, rotary)
        mlp = MLP(gate_proj, up_proj, down_proj)
        layers.append(Block(input_norm, attn, post_norm, mlp))
        del q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

    norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps, device)
    norm.weight = nn.Parameter(
        store.get("model.norm.weight", device).half(), requires_grad=False
    )

    tied = spec.tie_word_embeddings or ("lm_head.weight" not in store)
    lm_head = None if tied else DenseLinear(store.get("lm_head.weight", device))
    store.close()

    model = CausalLM(
        spec=spec,
        embed_tokens=embed_tokens,
        layers=nn.ModuleList(layers),
        norm=norm,
        lm_head=lm_head,
        tied=tied,
        max_len=max_len,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, {"model": spec.short_name, "load_s": 0.0}


@torch.inference_mode()
def evaluate_inference(
    recipe: CompressionRecipe,
    paths: dict[str, Path],
    *,
    manifest_path: str,
    device: str,
    batch: int,
    prefill_len: int,
    decode_steps: int,
    warmup: int,
    trials: int,
    cuda_graph: bool,
    kv_bits: int | None = None,
    regimes: tuple[str, ...] | None = None,
    allow_sparse: bool = True,
) -> dict[str, object]:
    if kv_bits is None:
        kv_bits = int(recipe.kv_cache.k_bits)
        if kv_bits != int(recipe.kv_cache.v_bits):
            print("warning: inference engine uses KV16 when K/V bits differ")
            kv_bits = 16
    if kv_bits not in (16, 8, 4):
        kv_bits = 16

    source = resolve_model_source(recipe.model, manifest_path)
    group_size = recipe.weight.granularity.group_size or 128
    selected = tuple(regimes) if regimes else tuple(paths)
    efficiency: dict[str, object] = {}
    for checkpoint_regime in selected:
        path = paths[checkpoint_regime]
        torch.cuda.empty_cache()
        start = time.perf_counter()
        model, _ = build_checkpoint_model(
            recipe,
            path,
            checkpoint_regime,
            manifest_path,
            device,
            prefill_len,
            decode_steps,
            allow_sparse=allow_sparse,
        )
        load_s = time.perf_counter() - start
        cfg = BenchConfig(
            model_dir=source,
            regime=_REGIME_LABELS[checkpoint_regime],
            device=device,
            batch=batch,
            prefill_len=prefill_len,
            decode_steps=decode_steps,
            kv_bits=kv_bits,
            warmup=warmup,
            trials=trials,
            group_size=group_size,
            cuda_graph=cuda_graph,
        )
        measured = measure_e2e(model, cfg)
        config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        efficiency[checkpoint_regime] = {
            "prefill": measured["prefill"],
            "decode": measured["decode"],
            "e2e": measured["e2e"],
            "memory": measured["memory"],
            "timing_mode": measured["timing_mode"],
            "load_s": load_s,
            "peak_memory_mb": measured["memory"]["peak_allocated_bytes"] / 2**20,
            **checkpoint_stats(path, config),
        }
        del model
        torch.cuda.empty_cache()
    return {"efficiency": efficiency}
