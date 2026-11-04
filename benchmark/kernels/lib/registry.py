# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel registry and device-aware selection for compressed benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qdrift.compress.recipe import CompressionRecipe

__all__ = [
    "KernelEntry",
    "KernelKey",
    "available_kernels",
    "device_profile",
    "kernel_plan",
    "select_kernel",
]


@dataclass(frozen=True)
class KernelKey:
    """Everything that can change the selected kernel."""

    regime: str
    pattern: str
    w_bits: int
    a_bits: int
    k_bits: int
    v_bits: int
    arch: str
    weight_granularity: str = "channel"
    group_size: int | None = None


@dataclass(frozen=True)
class KernelEntry:
    """One registered kernel implementation."""

    name: str
    pattern: str
    backend: str
    integer: bool
    sparse: bool
    requires_sparse_hw: bool = False
    requires_int8: bool = False
    requires_int4: bool = False
    requires_uniform_scale: bool = False
    requires_w4: bool = False
    requires_a16: bool = False
    requires_group128: bool = False

    def supports(self, key: KernelKey, profile: dict[str, object]) -> bool:
        if self.pattern != key.pattern:
            return False
        if self.requires_int4 and not (key.w_bits <= 4 and key.a_bits < 16):
            return False
        if self.requires_w4 and key.w_bits != 4:
            return False
        if self.requires_a16 and key.a_bits != 16:
            return False
        if (
            self.requires_group128
            and key.weight_granularity == "group"
            and key.group_size != 128
        ):
            return False
        if self.requires_uniform_scale and key.weight_granularity not in (
            "tensor",
            "channel",
        ):
            return False
        if self.integer and key.w_bits > 8:
            return False
        if not self.integer and key.w_bits != 16:
            return False
        if self.requires_sparse_hw and not profile["sparse_tensor_core"]:
            return False
        return not (self.requires_int8 and not profile["int8_tensor_core"])


_REGISTRY: tuple[KernelEntry, ...] = (
    KernelEntry(
        name="fp16_dense",
        pattern="dense",
        backend="cublas_fp16",
        integer=False,
        sparse=False,
    ),
    KernelEntry(
        name="cutlass_fp16_2to4",
        pattern="2:4",
        backend="cutlass_fp16_sparse",
        integer=False,
        sparse=True,
        requires_sparse_hw=True,
    ),
    KernelEntry(
        name="cutlass_int4_dense",
        pattern="dense",
        backend="cutlass_int4",
        integer=True,
        sparse=False,
        requires_int4=True,
        requires_uniform_scale=True,
    ),
    KernelEntry(
        name="cutlass_int4_4to8_sparse",
        pattern="4:8",
        backend="cutlass_int4_sparse",
        integer=True,
        sparse=True,
        requires_sparse_hw=True,
        requires_int4=True,
        requires_uniform_scale=True,
    ),
    KernelEntry(
        name="marlin_int4_dense",
        pattern="dense",
        backend="marlin_w4a16",
        integer=True,
        sparse=False,
        requires_w4=True,
        requires_a16=True,
        requires_group128=True,
    ),
    KernelEntry(
        name="sparse_marlin_int4_2to4",
        pattern="2:4",
        backend="sparse_marlin_w4a16",
        integer=True,
        sparse=True,
        requires_sparse_hw=True,
        requires_w4=True,
        requires_a16=True,
        requires_group128=True,
    ),
    KernelEntry(
        name="cublas_int8_dense",
        pattern="dense",
        backend="cublas_int8",
        integer=True,
        sparse=False,
        requires_int8=True,
    ),
    KernelEntry(
        name="cutlass_int8_2to4",
        pattern="2:4",
        backend="cutlass_int8_sparse",
        integer=True,
        sparse=True,
        requires_sparse_hw=True,
        requires_int8=True,
    ),
)


def available_kernels() -> tuple[KernelEntry, ...]:
    return _REGISTRY


def device_profile(device: str = "cuda") -> dict[str, object]:
    """Return the capabilities used by kernel selection."""
    info: dict[str, object] = {
        "device": device,
        "arch": "unknown",
        "sparse_tensor_core": False,
        "int8_tensor_core": False,
        "cutlass_sparse": False,
    }
    if not torch.cuda.is_available():
        return info
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return info
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    info["device"] = torch.cuda.get_device_name(index)
    info["arch"] = f"sm{major}{minor}"
    info["sparse_tensor_core"] = major >= 8 and hasattr(
        torch.sparse, "to_sparse_semi_structured"
    )
    info["cutlass_sparse"] = info["sparse_tensor_core"]
    # INT8 Tensor Cores exist from Turing onward; the benchmark path requires
    # Ada/Ampere-style cuBLAS INT8 support.
    info["int8_tensor_core"] = major >= 8
    return info


def _key(
    recipe: CompressionRecipe,
    regime: str,
    arch: str,
    allow_sparse: bool = True,
) -> KernelKey:
    if regime in ("dense", "sparse"):
        # FP16/INT8 sparse MMA is a standard 2:4 kernel.  The hardware
        # 2:4_pairwise mask is a valid subset of 2:4.  OBR-style 4:8 keeps any
        # four of eight and is therefore NOT a 2:4 subset; it uses dense fp16.
        pattern = "2:4" if regime == "sparse" and allow_sparse and (
            recipe.prune.structure in ("2:4", "2:4_pairwise")
        ) else "dense"
        return KernelKey(
            regime=regime,
            pattern=pattern,
            w_bits=16,
            a_bits=16,
            k_bits=16,
            v_bits=16,
            arch=arch,
            weight_granularity="channel",
            group_size=None,
        )
    pattern = "dense"
    if (
        regime == "joint"
        and allow_sparse
        and recipe.prune.structure in ("2:4", "4:8", "2:4_pairwise", "4:8_pairwise")
    ):
        if recipe.activation.bits < 16 and recipe.weight.bits <= 4:
            # INT4 sparse MMA uses pair-based metadata and only supports
            # per-channel/tensor weight scales.  The hardware 2:4_pairwise
            # mask is the only INT4-sparse structure here.  OBR-style 2:4/4:8
            # (arbitrary N:M) and group-scale checkpoints fall back to dense
            # INT4/INT8.
            int4_sparse = (
                recipe.prune.structure in ("2:4_pairwise", "4:8_pairwise")
                and recipe.weight.granularity.kind in ("tensor", "channel")
            )
            pattern = "4:8" if int4_sparse else "dense"
        else:
            # FP16 / INT8 sparse MMA is standard 2:4.  The 2:4_pairwise mask
            # is a valid subset; OBR-style 4:8 is not.
            pattern = (
                "2:4"
                if recipe.prune.structure in ("2:4", "2:4_pairwise")
                else "dense"
            )
    w_bits = recipe.weight.bits
    return KernelKey(
        regime=regime,
        pattern=pattern,
        w_bits=w_bits,
        a_bits=recipe.activation.bits,
        k_bits=recipe.kv_cache.k_bits,
        v_bits=recipe.kv_cache.v_bits,
        arch=arch,
        weight_granularity=recipe.weight.granularity.kind,
        group_size=recipe.weight.granularity.group_size,
    )


def select_kernel(
    recipe: CompressionRecipe,
    regime: str,
    device: str = "cuda",
    allow_sparse: bool = True,
) -> KernelEntry:
    """Select the highest-priority registered kernel for one regime."""
    profile = device_profile(device)
    key = _key(recipe, regime, str(profile["arch"]), allow_sparse)
    candidates = [
        entry
        for entry in _REGISTRY
        if entry.supports(key, profile) and (allow_sparse or not entry.sparse)
    ]
    if not candidates:
        raise RuntimeError(
            f"no registered kernel for regime={regime} pattern={key.pattern} "
            f"arch={key.arch} on {profile['device']}"
        )
    # Prefer an integer kernel for quantized regimes, then a sparse kernel.
    return max(
        candidates,
        key=lambda entry: (
            entry.integer,
            entry.requires_int4 or entry.requires_w4,
            entry.sparse,
        ),
    )


def kernel_plan(
    recipe: CompressionRecipe,
    device: str = "cuda",
    allow_sparse: bool = True,
) -> dict[str, object]:
    """Return the selected kernel for all four benchmark regimes."""
    profile = device_profile(device)
    selected = {
        regime: select_kernel(recipe, regime, device, allow_sparse).name
        for regime in ("dense", "sparse", "quant", "joint")
    }
    return {"device": profile, "selected": selected}
