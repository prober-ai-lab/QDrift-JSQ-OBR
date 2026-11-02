# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Hugging Face tokenizers from the model manifest."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from qdrift.models.backbone import resolve_model_source

__all__ = ["load_tokenizer"]


def load_tokenizer(
    model_name: str,
    manifest_path: str | Path = "modelzoo/manifest.yaml",
) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching a manifest identifier or direct path."""
    source = resolve_model_source(model_name, manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
