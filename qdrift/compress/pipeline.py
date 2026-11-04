# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compression pipeline with optional single-node multi-GPU execution.

Order per layer:

    mask -> pruning compensation -> scale -> quantization compensation
    -> scale -> quantize -> save

The model/calibration live on the main GPU.  Workers only receive the tensors
needed for their layer or row slice.  ``devices`` of length one is the original
sequential path.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

import torch
from safetensors.torch import load_file, save_file

from qdrift.compress.recipe import CompressionRecipe, variant_recipes
from qdrift.data.dataloader import make_calibration_chunks
from qdrift.data.tokenizer import load_tokenizer
from qdrift.models.backbone import load_backbone, quantizable_linear_layers
from qdrift.models.permutation import (
    apply_channel_permutations,
    build_channel_permutations,
)
from qdrift.models.rotation import (
    fuse_layer_norms,
    install_online_hadamards,
    rotate_model,
)
from qdrift.obr.calibrate import (
    SequentialCalibration,
    capture_calibration_start,
    capture_inputs,
    layer_groups,
)
from qdrift.obr.drift import drift_weight
from qdrift.obr.obr import (
    compensate_pruning_error,
    compensate_quantization_error,
    split_mask,
)
from qdrift.prune.criteria import (
    magnitude_score,
    qwanda_score,
    random_score,
    wanda_score,
)
from qdrift.prune.mask import mask_from_score
from qdrift.quant.gptq import gptq_quantize, rtn_quantize
from qdrift.quant.pack import pack_codes
from qdrift.quant.quantize import dequantize, fake_quantize, qrange
from qdrift.quant.scale import compute_scale
from qdrift.quant.smooth import channel_max, smooth_scale, smooth_weight

__all__ = [
    "DevicePool",
    "compress",
    "compress_regimes",
    "normalize_device",
    "parse_devices",
    "write_fp16_baseline",
]


def _hessian_from_inputs(
    inputs: torch.Tensor,
    inverse_smooth: torch.Tensor | None,
) -> torch.Tensor:
    if inverse_smooth is not None:
        inputs = inputs * inverse_smooth.reshape(1, -1)
    hessian = inputs.t() @ inputs / max(inputs.shape[0], 1)
    return 0.5 * (hessian + hessian.t())


_STREAM_HESSIAN_ELEMENTS = 1 << 29
_HESSIAN_STREAM_ROWS = 8192


@torch.no_grad()
def _hessians_streaming(
    fp_input: torch.Tensor,
    compressed_input: torch.Tensor | None,
    inverse_smooth: torch.Tensor | None,
    damping: float,
    shift_scale: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Accumulate calibration Hessians without moving the full input to GPU."""
    total = int(fp_input.shape[0])
    width = int(fp_input.shape[-1])
    hessian_fp = torch.zeros(width, width, device=device, dtype=torch.float32)
    hessian_hat = (
        torch.zeros_like(hessian_fp) if compressed_input is not None else None
    )
    cross = torch.zeros_like(hessian_fp) if compressed_input is not None else None
    for start in range(0, total, _HESSIAN_STREAM_ROWS):
        end = min(start + _HESSIAN_STREAM_ROWS, total)
        zfp = fp_input[start:end].to(device=device, dtype=torch.float32)
        if inverse_smooth is not None:
            zfp = zfp * inverse_smooth.reshape(1, -1)
        hessian_fp.add_(zfp.t() @ zfp)
        if compressed_input is not None and hessian_hat is not None and cross is not None:
            zhat = compressed_input[start:end].to(device=device, dtype=torch.float32)
            if inverse_smooth is not None:
                zhat = zhat * inverse_smooth.reshape(1, -1)
            hessian_hat.add_(zhat.t() @ zhat)
            cross.add_(zfp.t() @ zhat)
            del zhat
        del zfp
    scale = max(total, 1)
    hessian_fp.div_(scale)
    if compressed_input is not None and hessian_hat is not None and cross is not None:
        hessian_hat.div_(scale)
        cross.div_(scale)
        hessian_diff = shift_scale * (hessian_hat - cross)
        hessian_system = (1.0 - shift_scale) * hessian_fp + shift_scale * hessian_hat
    else:
        hessian_diff = None
        hessian_system = hessian_fp.clone()
    damp = damping * torch.diagonal(hessian_system).clamp_min(0).mean()
    hessian_system = hessian_system + damp * torch.eye(
        hessian_system.shape[0],
        device=hessian_system.device,
        dtype=hessian_system.dtype,
    )
    hessian_system = 0.5 * (hessian_system + hessian_system.t())
    return hessian_fp, hessian_system, hessian_diff


def _score(
    criterion: str,
    weight: torch.Tensor,
    diag: torch.Tensor,
    eps: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    if criterion == "wanda":
        return wanda_score(weight, diag)
    if criterion == "qwanda":
        return qwanda_score(weight, diag, eps, lam)
    if criterion == "magnitude":
        return magnitude_score(weight)
    if criterion == "random":
        return random_score(weight)
    raise ValueError(f"unknown criterion {criterion}")


def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
    rows, cols = mask.shape
    mask_bits = mask.to(torch.uint8)
    padding = (-cols) % 8
    if padding:
        mask_bits = torch.nn.functional.pad(mask_bits, (0, padding))
    weights = (1 << torch.arange(8, device=mask.device)).to(torch.int16)
    return (
        (mask_bits.reshape(rows, -1, 8).to(torch.int16) * weights)
        .sum(dim=2)
        .to(torch.uint8)
        .cpu()
        .contiguous()
    )


def _smooth_for_layer(
    weight: torch.Tensor,
    fp_input: torch.Tensor | None,
    alpha: float | None,
) -> torch.Tensor | None:
    if alpha is None or fp_input is None:
        return None
    return smooth_scale(
        fp_input.abs().amax(dim=0),
        channel_max(weight),
        alpha,
    )


def _hessians(
    fp_input: torch.Tensor | None,
    compressed_input: torch.Tensor | None,
    inverse_smooth: torch.Tensor | None,
    damping: float,
    shift_scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if fp_input is None:
        return None, None, None
    hessian_fp = _hessian_from_inputs(fp_input, inverse_smooth)
    if compressed_input is not None:
        hessian_hat = _hessian_from_inputs(compressed_input, inverse_smooth)
        zfp = fp_input if inverse_smooth is None else fp_input * inverse_smooth
        zhat = (
            compressed_input
            if inverse_smooth is None
            else compressed_input * inverse_smooth
        )
        cross = zfp.t() @ zhat / max(zfp.shape[0], 1)
        hessian_diff = shift_scale * (hessian_hat - cross)
        hessian_system = (1.0 - shift_scale) * hessian_fp + shift_scale * hessian_hat
    else:
        hessian_diff = None
        hessian_system = hessian_fp.clone()
    damp = damping * torch.diagonal(hessian_system).clamp_min(0).mean()
    hessian_system = hessian_system + damp * torch.eye(
        hessian_system.shape[0],
        device=hessian_system.device,
        dtype=hessian_system.dtype,
    )
    hessian_system = 0.5 * (hessian_system + hessian_system.t())
    return hessian_fp, hessian_system, hessian_diff


@dataclass
class LayerResult:
    tensors: dict[str, torch.Tensor]
    meta: dict[str, object]
    new_weight: torch.Tensor
    value_layout: str
    scale_layout: str
    code_encoding: str


def _write_layer_mask(sink: Path, name: str, packed: torch.Tensor) -> None:
    """Atomically publish one packed mask for the sparse-baseline worker."""
    sink.mkdir(parents=True, exist_ok=True)
    path = sink / f"{name}.mask"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(packed.cpu().contiguous(), temporary)
    os.replace(temporary, path)


def _quantize_layer(
    recipe: CompressionRecipe,
    qparams: tuple[int, int, float],
    name: str | None,
    weight: torch.Tensor,
    weight_rot: torch.Tensor,
    smooth: torch.Tensor | None,
    hessian_fp: torch.Tensor | None,
    hessian_system: torch.Tensor | None,
    hessian_raw: torch.Tensor | None,
    hessian_diff: torch.Tensor | None,
    drift_weight: torch.Tensor | None = None,
) -> LayerResult:
    rows, cols = weight.shape
    # Scale is computed independently for this layer from the pre-mask rotated
    # weight.  Do not reuse a global scale across layers, and do not rebuild it
    # from the compensated retained values.
    full_scale = compute_scale(
        weight_rot,
        recipe.weight.bits,
        recipe.weight.grid,
        recipe.weight.granularity.kind,
        recipe.weight.granularity.group_size,
        recipe.weight.effective_rhos(),
        mse_clip=(
            recipe.weight.range == "clip"
            and recipe.weight.granularity.kind in ("channel", "tensor")
        ),
    )
    quant_error = (
        fake_quantize(
            weight_rot,
            full_scale.scale,
            qparams,
            recipe.weight.grid,
        )
        - weight_rot
    )
    if hessian_fp is not None:
        diag = torch.diagonal(hessian_fp).clamp_min(0)
    else:
        diag = torch.ones(cols, device=weight.device)
    score = _score(
        recipe.prune.criterion,
        weight_rot,
        diag,
        quant_error,
        recipe.prune.lam,
    )
    base_structure = recipe.prune.structure
    structure = base_structure
    pairwise_ratio = None
    if base_structure in ("2:4", "4:8"):
        pair_structure = f"{base_structure}_pairwise"
        if name is not None and recipe.pairwise_layers:
            structure = pair_structure if name in recipe.pairwise_layers else base_structure
        elif recipe.pairwise_fraction >= 1.0:
            structure = pair_structure
        elif recipe.pairwise_fraction <= 0.0:
            structure = base_structure
        else:
            dense_mask = mask_from_score(score, recipe.prune.sparsity, base_structure)
            pair_mask = mask_from_score(score, recipe.prune.sparsity, pair_structure)
            dense_loss = score.masked_fill(dense_mask, 0).sum()
            pair_loss = score.masked_fill(pair_mask, 0).sum()
            pairwise_ratio = float(pair_loss / dense_loss.clamp_min(1e-12))
            structure = (
                pair_structure
                if pairwise_ratio <= recipe.pairwise_threshold
                else base_structure
            )
        mask = mask_from_score(score, recipe.prune.sparsity, structure)
    else:
        mask = mask_from_score(
            score,
            recipe.prune.sparsity,
            base_structure,
        )
    keep, _drop = split_mask(mask)
    full_layout = (
        recipe.weight.method == "gptq"
        or recipe.weight.granularity.kind == "group"
    )
    if recipe.weight.method == "rtn":
        # OBR computes the quantizer on the rotated, pre-mask W.  Keep that
        # scale for both retained-column compensation and final quantization.
        retained_scale = full_scale
        quant_scale = full_scale.scale.gather(1, keep)
    else:
        retained_scale = None
        quant_scale = full_scale.scale.gather(1, keep)
    hessian_fp = None
    quant_error = None
    score = None
    torch.cuda.empty_cache()

    pruning_state = None
    retained = None
    retained2 = None
    codes_compact = None
    codes_full = None
    dequant_compact = None
    dequant_full = None
    compensated_full = None
    rotated = None

    if recipe.compensation.method != "none":
        if hessian_system is None or hessian_raw is None:
            raise ValueError("compensation requires calibration inputs")
        if recipe.compensation.method == "drift" and hessian_diff is None:
            raise ValueError("DriftOBR requires dual-path calibration inputs")
        pruning_state = compensate_pruning_error(
            weight_rot,
            hessian_system,
            hessian_raw,
            hessian_diff if recipe.compensation.method == "drift" else None,
            mask,
            alpha=recipe.compensation.alpha,
            drift_weight=drift_weight,
        )
        retained = pruning_state.retained
        retained2 = compensate_quantization_error(
            weight_rot,
            pruning_state,
            quant_scale,
            qparams,
            recipe.weight.grid,
            alpha=recipe.compensation.alpha,
        )
    else:
        retained = weight_rot.gather(1, keep)
        retained2 = retained

    tensors: dict[str, torch.Tensor] = {}
    if recipe.weight.method == "rtn":
        if full_layout:
            retained_full = torch.zeros(
                rows,
                cols,
                device=weight.device,
                dtype=weight.dtype,
            )
            retained_full.scatter_(1, keep, retained2)
            codes_full, dequant_full = rtn_quantize(
                retained_full,
                retained_scale.scale,
                qparams,
                recipe.weight.grid,
            )
            rotated = dequant_full * mask
            value_layout = "full"
            scale_layout = "original"
            packed, code_encoding = pack_codes(
                codes_full.cpu(),
                recipe.weight.bits,
            )
            tensors["weight_packed"] = packed.contiguous()
            tensors["weight_scale"] = (
                retained_scale.scale_compact.to(torch.float16).cpu().contiguous()
            )
            tensors["weight_clip_rho"] = (
                retained_scale.rho_compact.to(torch.float16).cpu().contiguous()
            )
            stored_scale_shape = list(retained_scale.scale_compact.shape)
        else:
            codes_compact, dequant_compact = rtn_quantize(
                retained2,
                quant_scale,
                qparams,
                recipe.weight.grid,
            )
            rotated = torch.zeros(rows, cols, device=weight.device, dtype=weight.dtype)
            rotated.scatter_(1, keep, dequant_compact)
            value_layout = "compacted"
            scale_layout = "retained"
            packed, code_encoding = pack_codes(
                codes_compact.cpu(),
                recipe.weight.bits,
            )
            tensors["weight_kept_packed"] = packed.contiguous()
            tensors["weight_kept_shape"] = torch.tensor(
                [rows, keep.shape[1]],
                dtype=torch.int32,
            )
            tensors["weight_kept_scale"] = (
                retained_scale.scale_compact.to(torch.float16).cpu().contiguous()
            )
            tensors["weight_kept_rho"] = (
                retained_scale.rho_compact.to(torch.float16).cpu().contiguous()
            )
            stored_scale_shape = list(retained_scale.scale_compact.shape)
    elif recipe.weight.method == "gptq":
        if hessian_system is None:
            raise ValueError("GPTQ requires calibration inputs")
        compensated_full = torch.zeros(
            rows,
            cols,
            device=weight.device,
            dtype=weight.dtype,
        )
        compensated_full.scatter_(1, keep, retained2)
        gptq_alpha = (
            recipe.compensation.alpha if recipe.compensation.method != "none" else 0.0
        )
        codes_full = gptq_quantize(
            compensated_full,
            hessian_system,
            mask,
            full_scale.scale,
            qparams,
            recipe.weight.grid,
            alpha=gptq_alpha,
            damp=0.0,
        )
        dequant_full = dequantize(codes_full, full_scale.scale, qparams) * mask
        rotated = dequant_full
        value_layout = "full"
        scale_layout = "original"
        packed, code_encoding = pack_codes(
            codes_full.cpu(),
            recipe.weight.bits,
        )
        tensors["weight_packed"] = packed.contiguous()
        tensors["weight_scale"] = (
            full_scale.scale_compact.to(torch.float16).cpu().contiguous()
        )
        tensors["weight_clip_rho"] = (
            full_scale.rho_compact.to(torch.float16).cpu().contiguous()
        )
        stored_scale_shape = list(full_scale.scale_compact.shape)
    else:
        raise ValueError(f"unknown weight method {recipe.weight.method}")

    new_weight = rotated
    if smooth is not None:
        new_weight = new_weight / smooth.reshape(1, -1)

    tensors["weight_mask"] = _pack_mask(mask)
    tensors["weight_shape"] = torch.tensor([rows, cols], dtype=torch.int32)
    if smooth is not None:
        tensors["weight_smooth_scale"] = smooth.to(torch.float16).cpu().contiguous()
    meta = {
        "shape": [rows, cols],
        "value_layout": value_layout,
        "scale_layout": scale_layout,
        "kept_shape": [rows, int(keep.shape[1])],
        "bits": recipe.weight.bits,
        "grid": recipe.weight.grid,
        "range": recipe.weight.range,
        "granularity": recipe.weight.granularity.to_dict(),
        "scale_shape": stored_scale_shape,
        "code_encoding": code_encoding,
        "nnz": int(mask.sum().item()),
        "smooth_alpha": recipe.smoothing.alpha,
        "structure": structure,
        "pairwise_ratio": pairwise_ratio,
    }
    return LayerResult(
        tensors=tensors,
        meta=meta,
        new_weight=new_weight.to("cpu", dtype=torch.float16),
        value_layout=value_layout,
        scale_layout=scale_layout,
        code_encoding=code_encoding,
    )


def _process_full_layer(
    recipe: CompressionRecipe,
    qparams: tuple[int, int, float],
    name: str | None,
    weight: torch.Tensor,
    fp_input: torch.Tensor | None,
    compressed_input: torch.Tensor | None,
    device: str,
    need_hessian: bool,
) -> LayerResult:
    torch.cuda.set_device(device)
    local_weight = weight.detach().float().to(device)
    local_fp = None
    local_hat = None
    if recipe.smoothing.alpha is not None and fp_input is not None:
        activation_max = fp_input.abs().amax(dim=0).to(device)
        smooth = smooth_scale(
            activation_max,
            channel_max(local_weight),
            recipe.smoothing.alpha,
        )
    else:
        smooth = None
    inverse_smooth = smooth.reciprocal() if smooth is not None else None
    weight_smoothed = (
        smooth_weight(local_weight, smooth) if smooth is not None else local_weight
    )
    if need_hessian:
        if fp_input is not None and fp_input.numel() > _STREAM_HESSIAN_ELEMENTS:
            hessian_fp, hessian_system, hessian_diff = _hessians_streaming(
                fp_input,
                compressed_input,
                inverse_smooth,
                recipe.compensation.damping,
                recipe.compensation.shift_scale,
                device,
            )
            hessian_raw = hessian_system
        else:
            local_fp = (
                fp_input.to(device=device, dtype=torch.float32)
                if fp_input is not None
                else None
            )
            local_hat = (
                compressed_input.to(device=device, dtype=torch.float32)
                if compressed_input is not None
                else None
            )
            hessian_fp, hessian_system, hessian_diff = _hessians(
                local_fp,
                local_hat,
                inverse_smooth,
                recipe.compensation.damping,
                recipe.compensation.shift_scale,
            )
            hessian_raw = hessian_system
    else:
        hessian_fp = hessian_system = hessian_raw = hessian_diff = None
    result = _quantize_layer(
        recipe,
        qparams,
        name,
        local_weight,
        weight_smoothed,
        smooth,
        hessian_fp,
        hessian_system,
        hessian_raw,
        hessian_diff,
    )
    del local_weight, local_fp, local_hat, weight_smoothed
    torch.cuda.empty_cache()
    return result


def _process_row_slice(
    recipe: CompressionRecipe,
    qparams: tuple[int, int, float],
    name: str | None,
    weight_rows: torch.Tensor,
    smooth: torch.Tensor | None,
    hessian_fp: torch.Tensor | None,
    hessian_system: torch.Tensor | None,
    hessian_raw: torch.Tensor | None,
    hessian_diff: torch.Tensor | None,
    drift_weight: torch.Tensor | None,
    device: str,
) -> LayerResult:
    torch.cuda.set_device(device)
    local_weight = weight_rows.detach().float().to(device)
    local_smooth = smooth.to(device) if smooth is not None else None
    local_fp = hessian_fp.to(device) if hessian_fp is not None else None
    local_system = hessian_system.to(device) if hessian_system is not None else None
    local_raw = hessian_raw.to(device) if hessian_raw is not None else None
    local_diff = hessian_diff.to(device) if hessian_diff is not None else None
    local_drift = drift_weight.to(device) if drift_weight is not None else None
    weight_smoothed = (
        smooth_weight(local_weight, local_smooth)
        if local_smooth is not None
        else local_weight
    )
    result = _quantize_layer(
        recipe,
        qparams,
        name,
        local_weight,
        weight_smoothed,
        local_smooth,
        local_fp,
        local_system,
        local_raw,
        local_diff,
        local_drift,
    )
    del (
        local_weight,
        local_smooth,
        local_fp,
        local_system,
        local_raw,
        local_diff,
        local_drift,
        weight_smoothed,
    )
    torch.cuda.empty_cache()
    return result


def _merge_slices(results: list[LayerResult]) -> LayerResult:
    first = results[0]
    row_keys = {
        "weight_kept_packed",
        "weight_kept_scale",
        "weight_kept_rho",
        "weight_mask",
        "weight_packed",
        "weight_scale",
        "weight_clip_rho",
    }
    tensors: dict[str, torch.Tensor] = {}
    for key, value in first.tensors.items():
        if key in row_keys:
            tensors[key] = torch.cat(
                [result.tensors[key] for result in results],
                dim=0,
            )
        else:
            tensors[key] = value
    rows = sum(int(result.meta["shape"][0]) for result in results)
    cols = int(first.meta["shape"][1])
    kept = int(first.meta["kept_shape"][1])
    tensors["weight_shape"] = torch.tensor([rows, cols], dtype=torch.int32)
    if "weight_kept_shape" in tensors:
        tensors["weight_kept_shape"] = torch.tensor(
            [rows, kept],
            dtype=torch.int32,
        )
    meta = dict(first.meta)
    meta["shape"] = [rows, cols]
    meta["kept_shape"] = [rows, kept]
    scale_shape = list(first.meta["scale_shape"])
    granularity = first.meta.get("granularity", {})
    if isinstance(granularity, dict) and granularity.get("kind") == "tensor":
        meta["scale_shape"] = [1]
    else:
        scale_shape[0] = rows
        meta["scale_shape"] = scale_shape
    meta["nnz"] = sum(int(result.meta["nnz"]) for result in results)
    new_weight = torch.cat(
        [result.new_weight for result in results],
        dim=0,
    )
    return LayerResult(
        tensors=tensors,
        meta=meta,
        new_weight=new_weight,
        value_layout=first.value_layout,
        scale_layout=first.scale_layout,
        code_encoding=first.code_encoding,
    )


@torch.no_grad()
def _pairwise_sensitivity(
    model: torch.nn.Module,
    groups,
    layers: dict[str, torch.nn.Linear],
    sequential: SequentialCalibration,
    device: str,
    base_structure: str,
) -> dict[str, float]:
    """Cheap pair-vs-dense sensitivity ranking from WANDA score loss."""
    ratios: dict[str, float] = {}
    for group in groups:
        inputs = sequential.capture(group.capture)
        if inputs.numel() == 0:
            continue
        values = inputs.to(device=device, dtype=torch.float32)
        diag = values.pow(2).mean(dim=0).clamp_min(0).sqrt().reshape(1, -1)
        for name in group.names:
            weight = layers[name].weight.data.float().to(device)
            score = weight.abs() * diag
            dense_mask = mask_from_score(score, 0.5, base_structure)
            pair_mask = mask_from_score(score, 0.5, f"{base_structure}_pairwise")
            dense_loss = score.masked_fill(dense_mask, 0).sum()
            pair_loss = score.masked_fill(pair_mask, 0).sum()
            ratios[name] = float(pair_loss / dense_loss.clamp_min(1e-12))
    return ratios


def _batch_chunks(chunks: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    if batch_size <= 1 or not chunks:
        return chunks
    return [
        torch.stack(chunks[start : start + batch_size])
        for start in range(0, len(chunks), batch_size)
    ]


def compress(
    recipe: CompressionRecipe,
    manifest_path: str = "modelzoo/manifest.yaml",
    device: str = "cuda",
    output_dir: str = "checkpoints",
    devices: list[str] | None = None,
    mask_sink: str | Path | None = None,
) -> Path:
    """Compress a model and write a checkpoint plus sidecar metadata.

    When ``mask_sink`` is set, each layer mask is published as soon as that
    layer is compressed.  A concurrent sparse-baseline worker can then consume
    masks while the joint run is still in progress.
    """
    recipe.validate()
    pool = DevicePool(devices or [device])
    main_device = pool.main
    mask_dir = Path(mask_sink) if mask_sink is not None else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(recipe.rotation.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.rotation.seed)

    model = load_backbone(
        recipe.model,
        manifest_path,
        device=main_device,
    )
    tokenizer = load_tokenizer(recipe.model, manifest_path)
    permutations = None
    if recipe.rotation.mode != "none":
        fuse_layer_norms(model)
        torch.manual_seed(recipe.rotation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe.rotation.seed)
        rotate_model(
            model,
            recipe.rotation.mode,
            main_device,
            recipe.rotation.seed,
        )
    if recipe.channel_permutation:
        permutations = build_channel_permutations(model)
        apply_channel_permutations(model, permutations)
    if recipe.rotation.mode != "none":
        install_online_hadamards(
            model,
            permutations.o if permutations is not None else None,
            permutations.down if permutations is not None else None,
        )
    layers = quantizable_linear_layers(model)
    if recipe.max_layers > 0:
        selected = list(layers)[: recipe.max_layers]
        layers = {name: layers[name] for name in selected}
    chunks = make_calibration_chunks(
        tokenizer,
        nsamples=recipe.nsamples,
        seqlen=recipe.seqlen,
        seed=recipe.rotation.seed,
    )
    chunks = _batch_chunks(chunks, recipe.batch_size)
    groups = layer_groups(model, layers)

    need_hessian = (
        recipe.prune.criterion in ("wanda", "qwanda")
        or recipe.compensation.method != "none"
        or recipe.weight.method == "gptq"
        or recipe.smoothing.alpha is not None
    )
    fp_inputs: dict[str, torch.Tensor] = {}
    sequential: SequentialCalibration | None = None
    if need_hessian:
        capture_names = (
            sorted({group.capture for group in groups})
            if recipe.compensation.method == "drift"
            else []
        )
        calibration = capture_calibration_start(
            model,
            capture_names,
            chunks,
            main_device,
        )
        fp_inputs = calibration.inputs
        sequential = SequentialCalibration(model, calibration)
        if (
            recipe.prune.structure in ("2:4", "4:8")
            and 0.0 < recipe.pairwise_fraction < 1.0
        ):
            ratios = _pairwise_sensitivity(
                model,
                groups,
                layers,
                sequential,
                main_device,
                recipe.prune.structure,
            )
            count = len(ratios)
            pair_count = min(
                max(int(round(recipe.pairwise_fraction * count)), 0),
                count,
            )
            if pair_count <= 0:
                threshold = float("-inf")
            elif pair_count >= count:
                threshold = float("inf")
            else:
                ordered = sorted(ratios.values())
                threshold = ordered[pair_count - 1] + 1e-9
            ordered_names = sorted(ratios, key=ratios.get)
            recipe = replace(
                recipe,
                pairwise_threshold=threshold,
                pairwise_layers=tuple(ordered_names[:pair_count]),
            )
            print(
                f"pairwise fraction: {pair_count}/{count} "
                f"threshold={threshold:.6g}",
                flush=True,
            )
            # Re-run calibration from the start; the sensitivity pass only read inputs.
            sequential = SequentialCalibration(model, calibration)

    qparams = qrange(recipe.weight.bits, recipe.weight.grid)
    tensors: dict[str, torch.Tensor] = {}
    layer_meta: dict[str, dict[str, object]] = {}
    start = time.time()
    index = 0
    total = len(layers)
    value_layout = "compacted"
    scale_layout = "retained"
    code_encoding = (
        "signed_int4_packed"
        if recipe.weight.bits == 4
        else f"signed_int{recipe.weight.bits}"
    )

    for group in groups:
        fp_input = None
        compressed_input = None
        if need_hessian:
            calibrated = sequential.capture(group.capture)
            if calibrated.numel() == 0:
                calibrated = capture_inputs(
                    model,
                    [group.capture],
                    chunks,
                    main_device,
                )[group.capture]
            if recipe.compensation.method == "drift":
                fp_input = fp_inputs[group.capture]
                compressed_input = calibrated
            else:
                fp_input = calibrated

        results: list[LayerResult] = []
        names = list(group.names)
        if pool.size > 1 and len(names) > 1:
            tasks = []
            for name in names:
                layer_weight = layers[name].weight.data
                tasks.append(
                    lambda dev, name=name, layer_weight=layer_weight, fp_input=fp_input, compressed_input=compressed_input: (
                        _process_full_layer(
                            recipe,
                            qparams,
                            name,
                            layer_weight,
                            fp_input,
                            compressed_input,
                            dev,
                            need_hessian,
                        )
                    )
                )
            results = pool.run(tasks)
            for name, result in zip(names, results):
                layers[name].weight.data = result.new_weight.to(
                    device=main_device,
                    dtype=layers[name].weight.dtype,
                )
                tensors.update(
                    {f"{name}.{key}": value for key, value in result.tensors.items()}
                )
                layer_meta[name] = result.meta
                if mask_dir is not None:
                    _write_layer_mask(mask_dir, name, result.tensors["weight_mask"])
                value_layout = result.value_layout
                scale_layout = result.scale_layout
                code_encoding = result.code_encoding
                index += 1
                print(
                    f"[{index}/{total}] {name} "
                    f"nnz={result.meta['nnz']} "
                    f"structure={result.meta.get('structure', recipe.prune.structure)} "
                    f"pairwise_ratio={result.meta.get('pairwise_ratio')} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
        elif (
            pool.size > 1
            and len(names) == 1
            and recipe.weight.granularity.kind != "tensor"
            and layers[names[0]].in_features <= 4096
            and not (
                recipe.prune.structure in ("2:4", "4:8")
                and 0.0 < recipe.pairwise_fraction < 1.0
            )
        ):
            name = names[0]
            weight = layers[name].weight.data.detach().float().to(main_device)
            smooth = _smooth_for_layer(
                weight,
                fp_input,
                recipe.smoothing.alpha,
            )
            inverse_smooth = smooth.reciprocal() if smooth is not None else None
            if need_hessian:
                hessian_fp, hessian_system, hessian_diff = _hessians(
                    fp_input,
                    compressed_input,
                    inverse_smooth,
                    recipe.compensation.damping,
                    recipe.compensation.shift_scale,
                )
                hessian_raw = hessian_system
            else:
                hessian_fp = hessian_system = hessian_raw = hessian_diff = None
            smoothed = smooth_weight(weight, smooth) if smooth is not None else weight
            local_drift_weight = (
                drift_weight(smoothed, hessian_diff)
                if hessian_diff is not None
                else None
            )
            del smoothed
            rows = weight.shape[0]
            splits = [
                (rows * index // pool.size, rows * (index + 1) // pool.size)
                for index in range(pool.size)
            ]
            tasks = [
                lambda dev, start_row=start, end_row=end, weight=weight, smooth=smooth, hessian_fp=hessian_fp, hessian_system=hessian_system, hessian_raw=hessian_raw, hessian_diff=hessian_diff, drift_weight=local_drift_weight: (
                    _process_row_slice(
                        recipe,
                        qparams,
                        name,
                        weight[start_row:end_row],
                        smooth,
                        hessian_fp,
                        hessian_system,
                        hessian_raw,
                        hessian_diff,
                        drift_weight[start_row:end_row]
                        if drift_weight is not None
                        else None,
                        dev,
                    )
                )
                for start, end in splits
            ]
            slice_results = pool.run(tasks)
            result = _merge_slices(slice_results)
            layers[name].weight.data = result.new_weight.to(
                device=main_device,
                dtype=layers[name].weight.dtype,
            )
            tensors.update(
                {f"{name}.{key}": value for key, value in result.tensors.items()}
            )
            layer_meta[name] = result.meta
            if mask_dir is not None:
                _write_layer_mask(mask_dir, name, result.tensors["weight_mask"])
            value_layout = result.value_layout
            scale_layout = result.scale_layout
            code_encoding = result.code_encoding
            index += 1
            print(
                f"[{index}/{total}] {name} "
                f"nnz={result.meta['nnz']} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            del (
                weight,
                smooth,
                hessian_fp,
                hessian_system,
                hessian_raw,
                hessian_diff,
                local_drift_weight,
            )
        else:
            for name in names:
                result = _process_full_layer(
                    recipe,
                    qparams,
                    name,
                    layers[name].weight.data,
                    fp_input,
                    compressed_input,
                    main_device,
                    need_hessian,
                )
                layers[name].weight.data = result.new_weight.to(
                    device=main_device,
                    dtype=layers[name].weight.dtype,
                )
                tensors.update(
                    {f"{name}.{key}": value for key, value in result.tensors.items()}
                )
                layer_meta[name] = result.meta
                if mask_dir is not None:
                    _write_layer_mask(mask_dir, name, result.tensors["weight_mask"])
                value_layout = result.value_layout
                scale_layout = result.scale_layout
                code_encoding = result.code_encoding
                index += 1
                print(
                    f"[{index}/{total}] {name} "
                    f"nnz={result.meta['nnz']} "
                    f"structure={result.meta.get('structure', recipe.prune.structure)} "
                    f"pairwise_ratio={result.meta.get('pairwise_ratio')} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
        del fp_input, compressed_input
        torch.cuda.empty_cache()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = recipe.checkpoint_id()
    tensor_path = out_dir / f"{checkpoint_id}.safetensors"
    config_path = out_dir / f"{checkpoint_id}.json"
    save_file(tensors, str(tensor_path))

    config = recipe.base_config()
    if permutations is not None:
        config["permutation"] = permutations.to_dict()
    config["layout"] = {
        "pack_axis": "in_features",
        "bit_order": "lsb_first",
        "code_encoding": code_encoding,
        "code_bits": recipe.weight.bits,
        "scale_dtype": "float16",
        "value_layout": value_layout,
        "scale_layout": scale_layout,
    }
    config["layers"] = layer_meta
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if mask_dir is not None:
        (mask_dir / "_done").write_text(f"{len(layer_meta)}\n", encoding="utf-8")
    print(f"saved {tensor_path}", flush=True)
    print(f"saved {config_path}", flush=True)
    return tensor_path


T = TypeVar("T")
_MISSING = object()


def normalize_device(device: str | torch.device) -> str:
    text = str(device).strip()
    if text == "cuda":
        return "cuda:0"
    if text.isdigit():
        return f"cuda:{text}"
    return text


def parse_devices(value: str) -> list[str]:
    """Parse ``0,1,2`` / ``cuda:0,cuda:1`` into normalized devices."""
    return [normalize_device(item) for item in value.split(",") if item.strip()]


def _device_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.device(device)
    return contextlib.nullcontext()


class DevicePool:
    """Run callables on a list of CUDA devices, one task slot per device."""

    def __init__(self, devices: Sequence[str | torch.device]) -> None:
        if not devices:
            raise ValueError("at least one device is required")
        self.devices = [normalize_device(device) for device in devices]
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"duplicate devices: {self.devices}")
        allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "")
        if len(self.devices) > 1 and "expandable_segments:true" in allocator.lower():
            raise RuntimeError(
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not safe "
                "with multi-GPU: concurrent cuSOLVER calls can corrupt the "
                "allocator and raise an illegal memory access. Unset it for "
                "multi-GPU runs (or use one device)."
            )
        self._solvers_warm = False

    @property
    def size(self) -> int:
        return len(self.devices)

    @property
    def main(self) -> str:
        return self.devices[0]

    def _warm_solvers(self) -> None:
        """Initialize cuSOLVER on every device before threads share the process.

        PyTorch's cuSOLVER lazy handle initialization is not thread-safe when
        two devices are first used concurrently, which can surface as an
        illegal-memory-access error inside Cholesky.  A tiny sequential solve
        per device avoids that race.
        """
        if self._solvers_warm or self.size == 1 or not torch.cuda.is_available():
            return
        for device in self.devices:
            if not device.startswith("cuda"):
                continue
            with _device_context(device):
                matrix = torch.eye(8, device=device)
                factor, _ = torch.linalg.cholesky_ex(matrix)
                torch.cholesky_solve(torch.ones(8, 1, device=device), factor)
            torch.cuda.synchronize(device)
        self._solvers_warm = True

    def run(self, tasks: Sequence[Callable[[str], T]]) -> list[T]:
        """Run ``tasks(device)`` and return results in input order."""
        if not tasks:
            return []
        if self.size == 1:
            with _device_context(self.devices[0]):
                return [task(self.devices[0]) for task in tasks]

        self._warm_solvers()
        results: list[T | object] = [_MISSING] * len(tasks)
        queues: list[list[tuple[int, Callable[[str], T]]]] = [[] for _ in self.devices]
        for index, task in enumerate(tasks):
            queues[index % self.size].append((index, task))

        def run_queue(
            device: str,
            queue: list[tuple[int, Callable[[str], T]]],
        ) -> None:
            with _device_context(device):
                for index, task in queue:
                    results[index] = task(device)

        with ThreadPoolExecutor(max_workers=self.size) as pool:
            futures = [
                pool.submit(run_queue, device, queue)
                for device, queue in zip(self.devices, queues)
                if queue
            ]
            for future in futures:
                future.result()
        if any(result is _MISSING for result in results):
            raise RuntimeError("a device task did not produce a result")
        return [result for result in results if result is not _MISSING]  # type: ignore[misc]


_ROOT = Path(__file__).resolve().parents[2]


def _unpack_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=mask.device)
    byte = mask[:, positions // 8]
    return ((byte >> (positions % 8)[None, :]) & 1).bool()


def _layer_names(
    model: torch.nn.Module,
    max_layers: int,
) -> list[str]:
    names = list(quantizable_linear_layers(model))
    if max_layers > 0:
        names = names[:max_layers]
    return names


def _wait_for_mask(
    mask_dir: str | Path,
    name: str,
    timeout: float = 12 * 3600,
) -> torch.Tensor:
    """Wait for a mask published by a concurrent joint compression run."""
    path = Path(mask_dir) / f"{name}.mask"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.5)
    return torch.load(path, map_location="cpu", weights_only=True)


def write_fp16_baseline(
    joint_checkpoint: str | Path | None,
    recipe: CompressionRecipe,
    output_dir: str | Path,
    manifest_path: str = "modelzoo/manifest.yaml",
    device: str = "cuda",
    use_joint_mask: bool = True,
    mask_dir: str | Path | None = None,
) -> Path:
    """Write an fp16 checkpoint using the joint mask (or a dense mask).

    Dense baselines pass ``joint_checkpoint=None`` and do not depend on the
    joint run.  Sparse baselines either read a finished joint checkpoint or
    wait for per-layer masks published by ``compress(..., mask_sink=...)``.
    """
    joint_tensors: dict[str, torch.Tensor] = {}
    if joint_checkpoint is not None:
        joint_tensors = load_file(str(Path(joint_checkpoint)))
    if use_joint_mask and not joint_tensors and mask_dir is None:
        raise ValueError("sparse baseline requires joint_checkpoint or mask_dir")

    model = load_backbone(recipe.model, manifest_path, device=device)
    permutations = None
    if recipe.rotation.mode != "none":
        fuse_layer_norms(model)
        torch.manual_seed(recipe.rotation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe.rotation.seed)
        rotate_model(model, recipe.rotation.mode, device, recipe.rotation.seed)
    if recipe.channel_permutation:
        permutations = build_channel_permutations(model)
        apply_channel_permutations(model, permutations)
    if recipe.rotation.mode != "none":
        install_online_hadamards(
            model,
            permutations.o if permutations is not None else None,
            permutations.down if permutations is not None else None,
        )

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    modules = dict(model.named_modules())
    for name in _layer_names(model, recipe.max_layers):
        layer = modules[name]
        rows = int(layer.weight.shape[0])
        cols = int(layer.weight.shape[1])
        weight = layer.weight.data.detach().float().to(device)
        if use_joint_mask:
            if mask_dir is not None:
                packed = _wait_for_mask(mask_dir, name).to(device=device)
            else:
                packed = joint_tensors[f"{name}.weight_mask"].to(device=device)
            mask = _unpack_mask(packed, cols)
        else:
            mask = torch.ones(rows, cols, device=device, dtype=torch.bool)
        kept = int(mask[0].sum())
        keep = torch.argsort(
            (~mask).to(torch.int8),
            dim=1,
            stable=True,
        )[:, :kept]
        retained = weight.gather(1, keep).to(torch.float16).cpu().contiguous()
        tensors[f"{name}.weight_kept_fp16"] = retained
        tensors[f"{name}.weight_kept_shape"] = torch.tensor(
            [rows, kept],
            dtype=torch.int32,
        )
        tensors[f"{name}.weight_mask"] = _pack_mask(mask)
        tensors[f"{name}.weight_shape"] = torch.tensor(
            [rows, cols],
            dtype=torch.int32,
        )
        layers[name] = {
            "shape": [rows, cols],
            "value_layout": "fp16_compacted",
            "scale_layout": "none",
            "kept_shape": [rows, kept],
            "bits": 16,
            "nnz": int(mask.sum().item()),
            "smooth_alpha": None,
        }

    config = recipe.base_config()
    if permutations is not None:
        config["permutation"] = permutations.to_dict()
    config["layout"] = {
        "value_layout": "fp16_compacted",
        "scale_layout": "none",
        "code_bits": 16,
        "code_encoding": "fp16_compacted",
    }
    config["layers"] = layers
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = recipe.checkpoint_id()
    tensor_path = out_dir / f"{checkpoint_id}.safetensors"
    config_path = out_dir / f"{checkpoint_id}.json"
    save_file(tensors, str(tensor_path))
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"saved {tensor_path}", flush=True)
    print(f"saved {config_path}", flush=True)
    return tensor_path


def _checkpoint_path(output_dir: str | Path, recipe: CompressionRecipe) -> Path:
    return Path(output_dir) / f"{recipe.checkpoint_id()}.safetensors"


def _split_devices(devices: list[str], groups: int) -> list[list[str]]:
    groups = max(1, min(groups, len(devices)))
    base, extra = divmod(len(devices), groups)
    result: list[list[str]] = []
    start = 0
    for index in range(groups):
        size = base + (1 if index < extra else 0)
        result.append(devices[start : start + size])
        start += size
    return result


def _pump_output(stream, prefix: str) -> None:
    for line in stream:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _worker_command(
    recipe_json: Path,
    variant: str,
    devices: list[str],
    output_dir: str | Path,
    manifest_path: str,
    mask_dir: Path | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "qdrift.compress.pipeline",
        str(recipe_json),
        variant,
        ",".join(devices),
        str(output_dir),
        manifest_path,
        str(mask_dir) if mask_dir is not None else "",
    ]


def _start_worker(
    recipe_json: Path,
    variant: str,
    devices: list[str],
    output_dir: str | Path,
    manifest_path: str,
    mask_dir: Path | None = None,
) -> tuple[subprocess.Popen[str], threading.Thread]:
    command = _worker_command(
        recipe_json,
        variant,
        devices,
        output_dir,
        manifest_path,
        mask_dir,
    )
    process = subprocess.Popen(
        command,
        cwd=str(_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    thread = threading.Thread(
        target=_pump_output,
        args=(process.stdout, variant),
        daemon=True,
    )
    thread.start()
    return process, thread


def _stop_workers(
    running: dict[str, tuple[subprocess.Popen[str], threading.Thread]],
) -> None:
    for process, _thread in running.values():
        if process.poll() is None:
            process.terminate()
    for process, thread in running.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        thread.join(timeout=5)


def _compress_regimes_sequential(
    recipe: CompressionRecipe,
    quant_spec: CompressionRecipe,
    dense_spec: CompressionRecipe,
    sparse_spec: CompressionRecipe,
    manifest_path: str,
    output_dir: str,
    devices: list[str],
    single: bool,
) -> dict[str, Path]:
    joint = compress(
        recipe,
        manifest_path=manifest_path,
        device=devices[0],
        output_dir=output_dir,
        devices=devices,
    )
    if single:
        return {"joint": joint}

    if quant_spec.checkpoint_id() == recipe.checkpoint_id():
        quant = joint
    else:
        quant = compress(
            quant_spec,
            manifest_path=manifest_path,
            device=devices[0],
            output_dir=output_dir,
            devices=devices,
        )
    dense = write_fp16_baseline(
        None,
        dense_spec,
        output_dir,
        manifest_path=manifest_path,
        device=devices[0],
        use_joint_mask=False,
    )
    if sparse_spec.checkpoint_id() == dense_spec.checkpoint_id():
        sparse = dense
    else:
        sparse = write_fp16_baseline(
            joint,
            sparse_spec,
            output_dir,
            manifest_path=manifest_path,
            device=devices[0],
            use_joint_mask=True,
        )
    return {"dense": dense, "sparse": sparse, "quant": quant, "joint": joint}


def _compress_regimes_parallel(
    recipe: CompressionRecipe,
    quant_spec: CompressionRecipe,
    dense_spec: CompressionRecipe,
    sparse_spec: CompressionRecipe,
    manifest_path: str,
    output_dir: str,
    devices: list[str],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need_quant = quant_spec.checkpoint_id() != recipe.checkpoint_id()
    need_sparse = sparse_spec.checkpoint_id() != dense_spec.checkpoint_id()
    started: set[str] = set()
    running: dict[str, tuple[subprocess.Popen[str], threading.Thread]] = {}

    with tempfile.TemporaryDirectory(prefix=".qdrift-regimes-", dir=out_dir) as temp:
        temp_path = Path(temp)
        recipe_json = temp_path / "recipe.json"
        recipe_json.write_text(
            json.dumps(recipe.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        mask_dir = temp_path / "masks"
        mask_dir.mkdir()

        if not need_quant:
            # The quantization-only variant is identical to joint, so give the
            # joint run every device except one for the independent dense pass.
            joint_dev = devices[1:] if len(devices) > 1 else devices
            second_dev = devices[:1]
            spare_dev = joint_dev
        elif len(devices) == 2:
            groups = _split_devices(devices, 2)
            joint_dev, second_dev = groups
            spare_dev = joint_dev
        else:
            groups = _split_devices(devices, 3)
            joint_dev, second_dev, spare_dev = groups

        def start(variant: str, group: list[str]) -> None:
            if variant in started:
                return
            started.add(variant)
            running[variant] = _start_worker(
                recipe_json,
                variant,
                group,
                output_dir,
                manifest_path,
                mask_dir if variant in ("joint", "sparse") else None,
            )

        joint_process = _start_worker(
            recipe_json,
            "joint",
            joint_dev,
            output_dir,
            manifest_path,
            mask_dir,
        )
        started.add("joint")
        running["joint"] = joint_process

        if need_quant:
            start("quant", second_dev)
            if need_sparse and len(devices) > 2:
                start("sparse", spare_dev)
        else:
            start("dense", second_dev)
            if need_sparse and len(devices) > 2:
                start("sparse", spare_dev)

        failure: str | None = None
        try:
            while running:
                for variant, (process, thread) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    thread.join(timeout=5)
                    del running[variant]
                    if code != 0:
                        failure = f"{variant} worker exited with status {code}"
                        break

                    if variant == "joint":
                        if need_sparse and "sparse" not in started:
                            start("sparse", spare_dev)
                        if "dense" not in started and not (
                            len(devices) == 2 and need_sparse
                        ):
                            start("dense", joint_dev)
                    elif variant == "quant":
                        if "dense" not in started and len(devices) == 2:
                            start("dense", second_dev)
                if failure is not None:
                    break
                time.sleep(0.2)
        finally:
            _stop_workers(running)

        if failure is not None:
            raise RuntimeError(failure)

    joint = _checkpoint_path(out_dir, recipe)
    quant = joint if not need_quant else _checkpoint_path(out_dir, quant_spec)
    dense = _checkpoint_path(out_dir, dense_spec)
    sparse = dense if not need_sparse else _checkpoint_path(out_dir, sparse_spec)
    for path in (joint, quant, dense, sparse):
        if not path.exists():
            raise RuntimeError(f"regime task did not write {path}")
    return {"dense": dense, "sparse": sparse, "quant": quant, "joint": joint}


def compress_regimes(
    recipe: CompressionRecipe,
    manifest_path: str = "modelzoo/manifest.yaml",
    device: str = "cuda",
    output_dir: str = "checkpoints",
    single: bool = False,
    devices: list[str] | None = None,
    parallel: str = "auto",
) -> dict[str, Path]:
    """Compress the joint recipe and write dense/sparse/quant counterparts.

    ``parallel=auto`` uses independent variant workers when more than one GPU
    is available.  Joint and quantization-only runs start together; the sparse
    baseline consumes per-layer masks as the joint run publishes them.  Each
    worker may itself use several GPUs for row/layer parallelism.
    ``parallel=intra`` keeps one compression process and uses the devices for
    row/layer parallelism; ``parallel=workers`` forces process-level splitting.
    """
    selected = [normalize_device(item) for item in (devices or [device])]
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate devices: {selected}")
    if parallel not in ("auto", "workers", "intra"):
        raise ValueError(f"unknown parallel mode {parallel}")
    output_dir = str(Path(output_dir).resolve())
    manifest = Path(manifest_path)
    if manifest.exists():
        manifest_path = str(manifest.resolve())
    variants = variant_recipes(recipe)
    quant_spec = variants["quant"]
    dense_spec = variants["dense"]
    sparse_spec = variants["sparse"]
    if single or len(selected) == 1 or parallel == "intra":
        return _compress_regimes_sequential(
            recipe,
            quant_spec,
            dense_spec,
            sparse_spec,
            manifest_path,
            output_dir,
            selected,
            single,
        )
    return _compress_regimes_parallel(
        recipe,
        quant_spec,
        dense_spec,
        sparse_spec,
        manifest_path,
        output_dir,
        selected,
    )


def _worker_entry(argv: list[str]) -> int:
    """Run one compression variant in a worker process."""
    if len(argv) != 6:
        raise ValueError(
            "worker expects recipe, variant, devices, output, manifest, mask"
        )
    recipe_json, variant, devices_text, output_dir, manifest, mask_text = argv
    recipe = CompressionRecipe.from_dict(
        json.loads(Path(recipe_json).read_text(encoding="utf-8"))
    )
    devices = parse_devices(devices_text)
    mask_dir = Path(mask_text) if mask_text else None
    variants = variant_recipes(recipe)

    if variant == "joint":
        path = compress(
            recipe,
            manifest_path=manifest,
            device=devices[0],
            output_dir=output_dir,
            devices=devices,
            mask_sink=mask_dir,
        )
    elif variant == "quant":
        path = compress(
            variants["quant"],
            manifest_path=manifest,
            device=devices[0],
            output_dir=output_dir,
            devices=devices,
        )
    elif variant == "dense":
        path = write_fp16_baseline(
            None,
            variants["dense"],
            output_dir,
            manifest_path=manifest,
            device=devices[0],
            use_joint_mask=False,
        )
    elif variant == "sparse":
        if mask_dir is None:
            raise ValueError("sparse variant requires mask_dir")
        path = write_fp16_baseline(
            None,
            variants["sparse"],
            output_dir,
            manifest_path=manifest,
            device=devices[0],
            use_joint_mask=True,
            mask_dir=mask_dir,
        )
    else:
        raise ValueError(f"unknown variant {variant}")
    print(f"{variant}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_entry(sys.argv[1:]))
