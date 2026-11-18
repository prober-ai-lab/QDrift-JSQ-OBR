# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Hugging Face backbones from the model manifest."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

__all__ = [
    "load_backbone",
    "load_manifest",
    "offload_backbone",
    "offload_device",
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
    offload: bool = False,
    device_map: str | dict | None = None,
) -> torch.nn.Module:
    """Load a model backbone by manifest identifier or by a direct path.

    With ``offload=True`` the decoder blocks stay on the host and only the
    embedding, rotary, final norm and lm-head move to ``device``; the
    calibration helpers shuttle one block to ``device`` at a time.  This lets a
    model whose weights exceed one GPU (for example LLaMA-2-13B on a 24 GB card)
    be compressed on a single device.  Every matmul still runs on ``device`` in
    the same order and dtype, so the result is bit-identical to ``offload=False``.

    ``device_map`` (e.g. ``"auto"``) is passed straight to
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` and lets
    accelerate shard the fp16 backbone across every visible GPU.  When it is
    set, ``offload`` and ``device`` are ignored because accelerate owns the
    placement.
    """
    source = resolve_model_source(model_name, manifest_path)
    kwargs: dict[str, object] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
        # Force GPU-only placement: accelerate's ``device_map="auto"`` otherwise
        # offloads some layers to CPU/meta, and the compressed-eval Triton kernels
        # require CUDA tensors (they fail with "cannot be accessed from Triton (cpu
        # tensor?)").  Give every visible GPU a memory budget and omit "cpu" so the
        # whole model is packed onto the GPUs (7B/8B/13B all fit across the cards).
        # ``QDRIFT_SHARD_MAX_MEM_GIB`` forces a smaller per-GPU budget (used to make
        # a model that would fit on one card span several, e.g. for shard tests).
        if isinstance(device_map, str) and torch.cuda.is_available():
            override = os.environ.get("QDRIFT_SHARD_MAX_MEM_GIB")
            max_memory: dict[int, str] = {}
            for i in range(torch.cuda.device_count()):
                if override:
                    gib = float(override)
                else:
                    free, _total = torch.cuda.mem_get_info(i)
                    gib = free / (1024**3) * 0.85
                max_memory[i] = f"{gib:.1f}GiB"
            kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if device_map is None:
        if offload:
            offload_backbone(model, device)
        else:
            model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def offload_backbone(model: torch.nn.Module, device: str) -> None:
    """Park the decoder blocks on the host, everything else on ``device``.

    Records ``device`` on the model so the calibration helpers know to shuttle
    blocks on demand (:func:`offload_device`).
    """
    inner = model.model
    inner.layers.to("cpu")
    inner.embed_tokens.to(device)
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is not None:
        rotary.to(device)
    inner.norm.to(device)
    head = getattr(model, "lm_head", None)
    if head is not None:
        head.to(device)
    model._qdrift_offload_device = device


def offload_device(model: torch.nn.Module) -> str | None:
    """The compute device if ``model`` is offloaded, else ``None``."""
    return getattr(model, "_qdrift_offload_device", None)


def quantizable_linear_layers(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """Return all linear layers except the language-model head."""
    result: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            result[name] = module
    return result
