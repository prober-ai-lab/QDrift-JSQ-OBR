# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed fast inference benchmark.

This backend reuses the checkpoint cases from ``benchmark.eval.runtime`` with
the small graph-friendly runtime in this package.  It supports both ``--rotation
none`` and ``--rotation hadamard`` checkpoints: for the rotated case it replays
the same QuaRot rotation on the embeddings and head, uses weightless RMSNorm
(the scales are already fused into the compressed weights), and installs the
online Hadamard transforms before ``o_proj`` and ``down_proj``, so the timing
path matches the rotated model that the perplexity tables report.
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
from qdrift.quarot.hadamard import matmul_hadU, random_hadamard_matrix

__all__ = ["build_checkpoint_model", "evaluate_inference"]

_REGIME_LABELS = {
    "dense": "dense",
    "sparse": "sparse_fp16",
    "quant": "int4_marlin",
    "joint": "joint_sparse_int4",
}

# Matches ``qdrift.models.rotation._CHUNK_ROWS`` so the fp64 folds stay
# bit-identical to the compress/eval path.
_ROTATE_CHUNK = 8192


def _rotate_rows(weight: torch.Tensor, q: torch.Tensor) -> None:
    """Rotate rows in place (``weight @ q``) in fp64, matching rotation.py."""
    dtype = weight.dtype
    rows = weight.shape[0]
    chunk = min(rows, _ROTATE_CHUNK)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        part = weight[start:end].to(device=q.device, dtype=torch.float64)
        weight[start:end] = torch.matmul(part, q).to(dtype=dtype)


def _fuse_norm_scale(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Multiply each row by the fp64 norm scale, matching ``fuse_ln_linear``."""
    dtype = weight.dtype
    scale = scale.double()
    rows = weight.shape[0]
    chunk = min(rows, _ROTATE_CHUNK)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        weight[start:end] = (weight[start:end].double() * scale).to(dtype)


def _install_online_hadamards(
    attn: Attention,
    mlp: MLP,
    num_heads: int,
    head_dim: int,
) -> None:
    """Install the runtime Hadamards that cannot be fused into the weights.

    Mirrors ``qdrift.models.rotation.install_online_hadamards`` for the minimal
    graph runtime: the O projection gets the per-head then full Hadamard, the
    MLP down projection (which follows SiLU) gets the full Hadamard.
    """

    def o_hook(module, inputs):
        if not inputs:
            return None
        value = inputs[0]
        shape = value.shape
        work = value.reshape(*shape[:-1], num_heads, head_dim)
        work = matmul_hadU(work)  # H_head per head
        work = matmul_hadU(work.reshape(shape))  # H_hidden
        return (work,) + inputs[1:]

    def down_hook(module, inputs):
        if not inputs:
            return None
        work = matmul_hadU(inputs[0])
        return (work,) + inputs[1:]

    attn.o_proj.register_forward_pre_hook(o_hook)
    mlp.down_proj.register_forward_pre_hook(down_hook)


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
    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    source = resolve_model_source(recipe.model, manifest_path)
    spec = ModelSpec.from_dir(source)
    rotated = recipe.rotation.mode != "none"
    if rotated and spec.tie_word_embeddings:
        raise NotImplementedError(
            "rotated inference with tied embeddings is not supported by the "
            "minimal runtime; use the default HF engine instead"
        )
    q: torch.Tensor | None = None
    if rotated:
        # ``random_hadamard_matrix`` draws from the CPU global RNG, exactly as
        # the compress/eval path does after its ``torch.manual_seed``.
        torch.manual_seed(recipe.rotation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe.rotation.seed)
        q = random_hadamard_matrix(spec.hidden_size, device)
    store = SafetensorStore(source)
    tensors = load_file(str(checkpoint))
    if any(key.endswith(".weight_smooth_scale") for key in tensors):
        raise ValueError(
            "checkpoint uses smoothing, which is no longer supported; "
            "recompress it with the current pipeline"
        )
    make_linear_layer = linear_factory or _linear

    embed_weight = store.get("model.embed_tokens.weight", device).half()
    if q is not None:
        _rotate_rows(embed_weight, q)
    # from_pretrained skips random init, which would otherwise draw from the
    # CUDA RNG (and fail if an earlier graph capture left it mid-capture).
    embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=True)

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
        post_norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps, device)
        if q is None:
            # Un-rotated checkpoints keep the original norm scales here; the
            # rotated ones have them fused into the compressed weights, so the
            # norms stay weightless (all-ones).
            input_norm.weight = nn.Parameter(
                store.get(f"{prefix}.input_layernorm.weight", device).half(),
                requires_grad=False,
            )
            post_norm.weight = nn.Parameter(
                store.get(f"{prefix}.post_attention_layernorm.weight", device).half(),
                requires_grad=False,
            )

        attn = Attention(spec, index, q_proj, k_proj, v_proj, o_proj, rotary)
        mlp = MLP(gate_proj, up_proj, down_proj)
        if q is not None:
            _install_online_hadamards(
                attn, mlp, spec.num_attention_heads, spec.head_dim
            )
        layers.append(Block(input_norm, attn, post_norm, mlp))
        del q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

    norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps, device)
    final_norm_scale: torch.Tensor | None = None
    if q is None:
        norm.weight = nn.Parameter(
            store.get("model.norm.weight", device).half(), requires_grad=False
        )
    else:
        final_norm_scale = store.get("model.norm.weight", device).half()

    tied = spec.tie_word_embeddings or ("lm_head.weight" not in store)
    if tied:
        lm_head = None
    else:
        head_weight = store.get("lm_head.weight", device).half()
        if q is not None:
            assert final_norm_scale is not None
            # Compress/eval fuses the final norm into the head, then rotates it.
            _fuse_norm_scale(head_weight, final_norm_scale)
            _rotate_rows(head_weight, q)
        lm_head = DenseLinear(head_weight)
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
