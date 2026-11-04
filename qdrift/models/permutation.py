# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight channel permutations around QuaRot's fast Hadamard."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelPermutations:
    """CPU index vectors for hidden, o_proj input, and down_proj input."""

    hidden: torch.Tensor
    o: torch.Tensor
    down: torch.Tensor

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "hidden": self.hidden.tolist(),
            "o": self.o.tolist(),
            "down": self.down.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ChannelPermutations":
        return cls(
            hidden=torch.tensor(data["hidden"], dtype=torch.long),
            o=torch.tensor(data["o"], dtype=torch.long),
            down=torch.tensor(data["down"], dtype=torch.long),
        )


def _pair_order(importance: torch.Tensor) -> torch.Tensor:
    """Sort channels so the strongest pair is first in every four-column group."""
    order = torch.argsort(
        importance.detach().float().cpu(),
        descending=True,
        stable=True,
    )
    return order.to(torch.long)


def _identity(width: int) -> torch.Tensor:
    return torch.arange(width, dtype=torch.long)


@torch.no_grad()
def build_channel_permutations(model: torch.nn.Module) -> ChannelPermutations:
    """Build deterministic permutations from rotated weight importance."""
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)
    hidden = torch.zeros(hidden_size, dtype=torch.float32)
    o_input = torch.zeros(hidden_size, dtype=torch.float32)
    down_input = torch.zeros(intermediate_size, dtype=torch.float32)

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj):
            hidden += linear.weight.detach().abs().sum(dim=0).float().cpu()
        o_input += attn.o_proj.weight.detach().abs().sum(dim=0).float().cpu()
        down_input += mlp.down_proj.weight.detach().abs().sum(dim=0).float().cpu()

    return ChannelPermutations(
        hidden=_pair_order(hidden),
        o=_pair_order(o_input),
        down=_pair_order(down_input),
    )


def _select_rows(weight: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return weight.index_select(0, index.to(weight.device))


def _select_cols(weight: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return weight.index_select(1, index.to(weight.device))


@torch.no_grad()
def apply_channel_permutations(
    model: torch.nn.Module,
    permutations: ChannelPermutations,
) -> None:
    """Apply the hidden basis P and the o/down input gathers to model weights.

    The hidden permutation is absorbed into embeddings, norms, q/k/v/gate/up
    input columns, and o/down output rows.  o_proj and down_proj additionally
    receive their own input-column permutations; their runtime hooks gather the
    matching activations after the online Hadamard.
    """
    hidden = permutations.hidden
    o_perm = permutations.o
    down_perm = permutations.down

    embedding = model.model.embed_tokens
    embedding.weight.data = _select_cols(embedding.weight.data, hidden)
    model.lm_head.weight.data = _select_cols(model.lm_head.weight.data, hidden)

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.data = _select_rows(module.weight.data, hidden)
        elif module.__class__.__name__ == "RMSNorm" and weight is not None:
            module.weight.data = _select_rows(module.weight.data, hidden)

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj):
            linear.weight.data = _select_cols(linear.weight.data, hidden)

        attn.o_proj.weight.data = _select_cols(
            _select_rows(attn.o_proj.weight.data, hidden),
            o_perm,
        )
        if attn.o_proj.bias is not None:
            attn.o_proj.bias.data = _select_rows(attn.o_proj.bias.data, hidden)

        mlp.down_proj.weight.data = _select_cols(
            _select_rows(mlp.down_proj.weight.data, hidden),
            down_perm,
        )
        if mlp.down_proj.bias is not None:
            mlp.down_proj.bias.data = _select_rows(mlp.down_proj.bias.data, hidden)
