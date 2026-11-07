# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal Llama-style runtime with swappable sparse/quantized linears.

The runtime is intentionally small: it exposes exactly the pieces needed to
separate prefill and decode and to replace every projection with a benchmark
kernel.  It does not depend on transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.inference.config import ModelSpec
from benchmark.inference.kv_cache import KVCache

__all__ = ["Attention", "Block", "CausalLM", "MLP", "RMSNorm", "RotaryEmbedding"]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, device: str | torch.device):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float16, device=device),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float,
        max_len: int,
        device: str | torch.device,
    ):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (
            float(theta) ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        positions = torch.arange(max_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos().half(), persistent=False)
        self.register_buffer("sin", emb.sin().half(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        return query, key


class Attention(nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        layer_idx: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        rotary: RotaryEmbedding,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.group = spec.group_size
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.rotary = rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: KVCache,
        positions: torch.Tensor,
        pos_int: int | None,
        prefill: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, seq_len, self.n_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        query, key = self.rotary(query, key, positions)

        if prefill:
            cache.write_prefill(self.layer_idx, key, value)
            key = key.transpose(1, 2).repeat_interleave(self.group, dim=1)
            value = value.transpose(1, 2).repeat_interleave(self.group, dim=1)
            attn = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key,
                value,
                is_causal=True,
            )
        else:
            assert pos_int is not None
            cache.append(self.layer_idx, key, value, pos_int)
            attn = cache.decode_attention(
                self.layer_idx,
                query.transpose(1, 2),
                pos_int + 1,
            )

        attn = attn.transpose(1, 2).reshape(
            batch, seq_len, self.n_heads * self.head_dim
        )
        return self.o_proj(attn)


class MLP(nn.Module):
    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(
        self,
        input_layernorm: RMSNorm,
        self_attn: Attention,
        post_attention_layernorm: RMSNorm,
        mlp: MLP,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: KVCache,
        positions: torch.Tensor,
        pos_int: int | None,
        prefill: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, cache, positions, pos_int, prefill
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CausalLM(nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        embed_tokens: nn.Embedding,
        layers: nn.ModuleList,
        norm: RMSNorm,
        lm_head: nn.Module | None,
        tied: bool,
        max_len: int,
        device: str | torch.device,
    ):
        super().__init__()
        self.spec = spec
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.tied = tied
        self.device = torch.device(device)
        # Static position buffer: slicing it is CUDA-graph safe, unlike
        # torch.tensor([pos], device=...) which performs a H2D copy.
        self.register_buffer(
            "position_ids",
            torch.arange(max_len, dtype=torch.long, device=device),
            persistent=False,
        )

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return F.linear(hidden_states, self.embed_tokens.weight)
        assert self.lm_head is not None
        return self.lm_head(hidden_states)

    def prefill(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        positions = self.position_ids[: input_ids.shape[1]]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cache, positions, None, True
            )
        cache.length = int(input_ids.shape[1])
        hidden_states = self.norm(hidden_states[:, -1:])
        return self._logits(hidden_states)

    def decode(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        pos = int(cache.length)
        hidden_states = self.embed_tokens(input_ids)
        positions = self.position_ids[pos : pos + 1]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cache, positions, pos, False
            )
        cache.length = pos + 1
        hidden_states = self.norm(hidden_states)
        return self._logits(hidden_states)

    def new_cache(
        self, batch: int, max_len: int, bits: int = 16, fused: bool = True
    ) -> KVCache:
        return KVCache(
            n_layers=self.spec.num_hidden_layers,
            batch=batch,
            n_kv_heads=self.spec.num_key_value_heads,
            head_dim=self.spec.head_dim,
            max_len=max_len,
            bits=bits,
            fused=fused,
            device=self.device,
        )

    def weight_bytes(self) -> int:
        total = self.embed_tokens.weight.numel() * self.embed_tokens.weight.element_size()
        total += self.norm.weight.numel() * self.norm.weight.element_size()
        for layer in self.layers:
            total += layer.input_layernorm.weight.numel() * 2
            total += layer.post_attention_layernorm.weight.numel() * 2
            for module in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                total += module.storage_bytes()
        if not self.tied and self.lm_head is not None:
            total += self.lm_head.storage_bytes()
        return total
