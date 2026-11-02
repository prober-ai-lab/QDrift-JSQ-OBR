# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Hugging Face backbones from the model manifest."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

__all__ = [
    "load_backbone",
    "load_manifest",
    "quantizable_linear_layers",
    "resolve_model_source",
]


def load_manifest(path: str | Path) -> dict[str, dict[str, object]]:
    """Load the model manifest."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    models = data.get("models")
    if not isinstance(models, dict):
        raise TypeError(f"manifest has no models mapping: {path}")
    return models


def resolve_model_source(
    model_name: str,
    manifest_path: str | Path = "modelzoo/manifest.yaml",
) -> str:
    """Resolve a manifest identifier or direct path to a local/HF source."""
    source = model_name
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return source
    models = load_manifest(manifest_file)
    if model_name not in models:
        return source
    spec = models[model_name]
    local_path = spec.get("local_path")
    hf_id = spec.get("hf_id")
    if local_path and Path(str(local_path)).exists():
        return str(local_path)
    if hf_id:
        return str(hf_id)
    return source


def load_backbone(
    model_name: str,
    manifest_path: str | Path = "modelzoo/manifest.yaml",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Load a model backbone by manifest identifier or by a direct path."""
    source = resolve_model_source(model_name, manifest_path)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def quantizable_linear_layers(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """Return all linear layers except the language-model head."""
    result: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            result[name] = module
    return result
