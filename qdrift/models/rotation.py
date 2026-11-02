# Adapted from: https://github.com/spcl/QuaRot (commit 5008669)
# License: Apache-2.0
"""Rotation folding for Llama-style decoder models."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from qdrift.quarot.hadamard import (
    apply_exact_had_to_linear,
    matmul_hadU,
    random_hadamard_matrix,
)
from qdrift.quarot.normalization import RMSNorm


def fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: Iterable[torch.nn.Linear],
) -> None:
    """Fuse a normalization scale into adjacent linear layers."""
    for linear in linear_layers:
        dtype = linear.weight.dtype
        weight = linear.weight.data.double()
        fused = weight * layernorm.weight.double()
        linear.weight.data = fused.to(dtype)

        if getattr(layernorm, "bias", None) is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            bias = linear.bias.data.double()
            bias = bias + torch.matmul(weight, layernorm.bias.double())
            linear.bias.data = bias.to(dtype)


def fuse_layer_norms(model: torch.nn.Module) -> None:
    """Fuse RMSNorm scales and replace norms with weightless RMSNorm."""
    hidden_size = model.config.hidden_size
    # Qwen2.5-0.5B ties the embedding and lm_head weights.  Rotation/fusion
    # treats them in different bases, so break the tie first.
    if getattr(model.config, "tie_word_embeddings", False):
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False
    embeddings = model.model.embed_tokens
    # Llama/Qwen use RMSNorm, for which rotation invariance already holds; the
    # SliceGPT-style embedding mean subtraction is only needed for LayerNorm.
    eps = float(getattr(model.config, "rms_norm_eps", 1e-5))

    norm_kwargs: dict[str, torch.device | torch.dtype] = {
        "device": embeddings.weight.device,
        "dtype": embeddings.weight.dtype,
    }

    for layer in model.model.layers:
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )
        layer.input_layernorm = RMSNorm(hidden_size, eps=eps).to(**norm_kwargs)
        layer.post_attention_layernorm = RMSNorm(hidden_size, eps=eps).to(**norm_kwargs)

    fuse_ln_linear(model.model.norm, [model.lm_head])
    model.model.norm = RMSNorm(hidden_size, eps=eps).to(**norm_kwargs)


def get_orthogonal_matrix(
    size: int,
    mode: str,
    device: torch.device | str,
    seed: int = 0,
) -> torch.Tensor:
    """Build a fast Hadamard or identity matrix."""
    if mode == "hadamard":
        return random_hadamard_matrix(size, device)
    if mode == "none":
        return torch.eye(size, device=device, dtype=torch.float32)
    raise ValueError(f"unknown rotation mode {mode}")


def _rotate_weight_input(
    linear: torch.nn.Linear,
    q: torch.Tensor,
) -> None:
    dtype = linear.weight.dtype
    weight = linear.weight.data.to(device=q.device, dtype=torch.float64)
    linear.weight.data = torch.matmul(weight, q).to(dtype=dtype)


def _rotate_weight_output(
    linear: torch.nn.Linear,
    q: torch.Tensor,
) -> None:
    dtype = linear.weight.dtype
    weight = linear.weight.data.to(device=q.device, dtype=torch.float64)
    linear.weight.data = torch.matmul(q.T, weight).to(dtype=dtype)
    if linear.bias is not None:
        bias = linear.bias.data.to(device=q.device, dtype=torch.float64)
        linear.bias.data = torch.matmul(q.T, bias).to(dtype=dtype)


def rotate_embeddings(model: torch.nn.Module, q: torch.Tensor) -> None:
    """Rotate the input embedding matrix."""
    embedding = model.model.embed_tokens
    dtype = embedding.weight.dtype
    weight = embedding.weight.data.to(device=q.device, dtype=torch.float64)
    embedding.weight.data = torch.matmul(weight, q).to(dtype=dtype)


def rotate_lm_head(model: torch.nn.Module, q: torch.Tensor) -> None:
    """Rotate the language-model head."""
    head = model.lm_head
    dtype = head.weight.dtype
    weight = head.weight.data.to(device=q.device, dtype=torch.float64)
    head.weight.data = torch.matmul(weight, q).to(dtype=dtype)


def rotate_attention(
    layer: torch.nn.Module,
    q: torch.Tensor,
    head_dim: int,
) -> None:
    """Rotate attention weights and fuse the head-wise V/O Hadamards."""
    for linear in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ):
        _rotate_weight_input(linear, q)

    _rotate_weight_output(layer.self_attn.o_proj, q)
    # V output: H_head per KV head.  O input: the same full H used online.
    apply_exact_had_to_linear(
        layer.self_attn.v_proj,
        had_dim=head_dim,
        output=True,
    )
    apply_exact_had_to_linear(
        layer.self_attn.o_proj,
        had_dim=-1,
        output=False,
    )


def rotate_mlp(layer: torch.nn.Module, q: torch.Tensor) -> None:
    """Rotate the MLP input and output projections."""
    _rotate_weight_input(layer.mlp.gate_proj, q)
    _rotate_weight_input(layer.mlp.up_proj, q)
    _rotate_weight_output(layer.mlp.down_proj, q)
    apply_exact_had_to_linear(
        layer.mlp.down_proj,
        had_dim=-1,
        output=False,
    )


@torch.inference_mode()
def rotate_model(
    model: torch.nn.Module,
    mode: str = "hadamard",
    device: torch.device | str | None = None,
    seed: int = 0,
) -> None:
    """Rotate embeddings, head, attention, and MLP weights in place."""
    if device is None:
        device = next(model.parameters()).device

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    q = get_orthogonal_matrix(hidden_size, mode, device, seed)

    rotate_embeddings(model, q)
    rotate_lm_head(model, q)
    for layer in model.model.layers:
        rotate_attention(layer, q, head_dim)
        rotate_mlp(layer, q)


def install_online_hadamards(model: torch.nn.Module) -> None:
    """Install the runtime Hadamards that cannot be fused into weights.

    Attention V/O: V gets H_head per head, O gets full H_hidden.  The online
    O transform applies H_head per head and then H_hidden, so the two full
    Hadamards cancel.  MLP down: the activation follows SiLU, so apply H_full
    online.  Every transform here uses the fast Hadamard.
    """
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    def down_hook(module, inputs):
        if not inputs:
            return None
        return (matmul_hadU(inputs[0]),) + inputs[1:]

    def o_hook(module, inputs):
        if not inputs:
            return None
        value = inputs[0]
        shape = value.shape
        work = value.reshape(*shape[:-1], num_heads, head_dim)
        work = matmul_hadU(work)  # H_head per head
        work = matmul_hadU(work.reshape(shape))  # H_hidden
        return (work,) + inputs[1:]

    for layer in model.model.layers:
        layer.self_attn.o_proj.register_forward_pre_hook(o_hook)
        layer.mlp.down_proj.register_forward_pre_hook(down_hook)
