# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark configuration shared by the sparse/quantized runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

REGIMES = ("dense", "sparse_fp16", "int4_marlin", "joint_sparse_int4")


@dataclass(frozen=True)
class ModelSpec:
    model_dir: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelSpec":
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", heads))
        head_dim = int(config.get("head_dim", hidden // heads))
        return cls(
            model_dir=str(model_dir),
            model_type=str(config.get("model_type", "llama")),
            hidden_size=hidden,
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=int(config["intermediate_size"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            max_position_embeddings=int(
                config.get("max_position_embeddings", 32768)
            ),
        )

    @property
    def short_name(self) -> str:
        name = Path(self.model_dir).name.lower()
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in name)
        return safe or self.model_type


@dataclass
class BenchConfig:
    model_dir: str
    regime: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    batch: int = 1
    prefill_len: int = 512
    decode_steps: int = 32
    kv_bits: int = 16
    warmup: int = 2
    trials: int = 5
    group_size: int = 128
    cuda_graph: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}; choose from {REGIMES}")
        if self.kv_bits not in (16, 8, 4):
            raise ValueError(f"kv_bits must be 16, 8 or 4, got {self.kv_bits}")
        if self.batch < 1 or self.prefill_len < 1 or self.decode_steps < 1:
            raise ValueError("batch, prefill_len and decode_steps must be positive")
        if self.group_size not in (-1, 128):
            raise ValueError("Marlin only supports group_size -1 or 128")
        if self.model_dir is None:
            raise ValueError("model_dir is required")
