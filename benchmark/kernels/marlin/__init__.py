# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Marlin INT4 and Sparse-Marlin 2:4 INT4 kernels (W4A16).

The CUDA sources and the metadata conversion are adapted from
IST-DASLab/Sparse-Marlin (Apache-2.0):

    https://github.com/IST-DASLab/Sparse-Marlin

Marlin performs ``x @ W^T`` with FP16 activations and INT4 weights.  The
2:4 variant stores half of the INT4 weights plus 2-bit metadata and uses
sparse Tensor Cores.  Both kernels support per-channel and group-128
symmetric INT4 weights, so they complement the W4A4 CUTLASS path in
``benchmark/kernels/lib/w4a4.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from .packing import _calculate_meta_reordering_scatter_offsets

__all__ = ["MarlinWeight", "SparseMarlinWeight", "marlin_module"]

_MODULE = None


def _prepare_builder() -> None:
    from torch.utils import cpp_extension

    for flag in (
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ):
        try:
            cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass
    if shutil.which("ninja") is None:
        for candidate in ("/data1/xiaoxiao/.local/bin", "/usr/local/bin"):
            if Path(candidate, "ninja").exists():
                os.environ["PATH"] = f"{candidate}:{os.environ['PATH']}"
                break
    if "CUDA_HOME" not in os.environ:
        for candidate in ("/usr/local/cuda-12.1", "/usr/local/cuda"):
            if Path(candidate, "bin", "nvcc").exists():
                os.environ["CUDA_HOME"] = candidate
                break


def marlin_module():
    """Build (once) and return the Marlin/Sparse-Marlin CUDA extension."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _prepare_builder()
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent / "csrc"
    _MODULE = load(
        name="qdrift_sparse_marlin",
        sources=[
            str(root / "bindings.cpp"),
            str(root / "marlin_cuda_kernel_dense.cu"),
            str(root / "marlin_cuda_kernel_nm.cu"),
        ],
        extra_include_paths=[str(root)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )
    return _MODULE


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = torch.tensor(perm, dtype=torch.int64)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)
    perm = perm.reshape(-1, 8)[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, torch.tensor(scale_perm), torch.tensor(scale_perm_single)


def _get_perms_2_4():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) + 4 * block)
        for j in range(4):
            perm.extend([p + j for p in perm1])
    perm = torch.tensor(perm, dtype=torch.int64)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)
    perm = perm.reshape(-1, 8)[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return perm, torch.tensor(scale_perm), torch.tensor(scale_perm_single)


_PERM, _SCALE_PERM, _SCALE_PERM_SINGLE = _get_perms()
_PERM_24, _SCALE_PERM_24, _SCALE_PERM_SINGLE_24 = _get_perms_2_4()
_TILE = 16


def _pad_weight_and_scales(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int | None,
    out_pad: int,
    in_pad: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad weights/scales and return group-major scales in [groups, out_pad]."""
    out_features, in_features = weight.shape
    if scales.shape[1] == in_features:
        raise ValueError(
            "Sparse-Marlin needs compact group scales, not elementwise scales"
        )
    channel = group_size is None or group_size >= in_features
    groups = 1 if channel else (in_features + group_size - 1) // group_size
    if not channel and in_features % group_size:
        raise ValueError("group_size must divide the input width")
    if not channel and in_pad % group_size:
        raise ValueError("padded input width must be divisible by group_size")

    padded_weight = torch.zeros(
        out_pad, in_pad, device=weight.device, dtype=weight.dtype
    )
    padded_weight[:out_features, :in_features] = weight

    padded_groups = 1 if channel else in_pad // group_size
    padded_scales = torch.ones(
        out_pad, padded_groups, device=scales.device, dtype=scales.dtype
    )
    padded_scales[:out_features, :groups] = scales.reshape(out_features, groups)
    # group-major [groups, out_pad] -> flat [1, groups * out_pad]
    flat = padded_scales.t().contiguous().reshape(1, -1)
    return padded_weight, flat, (in_pad if channel else group_size)


def _quantize_marlin(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int | None,
    *,
    sparse: bool,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    out_features, in_features = weight.shape
    group = in_features if group_size is None else group_size
    if group_size is not None and group_size < in_features:
        if in_features % group_size:
            raise ValueError("group_size must divide the input width")
        w = weight.t().contiguous()  # [K, O]
        w = w.reshape(-1, group_size, out_features).permute(1, 0, 2)
        w = w.reshape(group_size, -1)
        s = scales.reshape(1, -1)
        w = torch.round(w / s).int() + 8
        w = w.clamp(0, 15)
        w = w.reshape(group_size, -1, out_features).permute(1, 0, 2)
        w = w.reshape(in_features, out_features).contiguous()
        scale_perm = _SCALE_PERM_24 if sparse else _SCALE_PERM
        s = s.reshape(-1, scale_perm.numel())[:, scale_perm]
    else:
        w = weight.t().contiguous()
        s = scales.reshape(1, -1)
        w = torch.round(w / s).int() + 8
        w = w.clamp(0, 15)
        scale_perm_single = (
            _SCALE_PERM_SINGLE_24 if sparse else _SCALE_PERM_SINGLE
        )
        s = s.reshape(-1, scale_perm_single.numel())[:, scale_perm_single]
    s = s.reshape(-1, out_features).contiguous()
    return w, s, in_features, out_features


def _pack_nibbles(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    in_features, out_features = weight.shape
    w = weight.reshape(
        in_features // _TILE, _TILE, out_features // _TILE, _TILE
    )
    w = w.permute(0, 2, 1, 3).reshape(in_features // _TILE, out_features * _TILE)
    res = w.reshape(-1, perm.numel())[:, perm].reshape(w.shape)
    res = res.cpu().numpy().astype("uint32")
    packed = torch.zeros(
        (res.shape[0], res.shape[1] // 8), dtype=torch.int32, device=weight.device
    )
    packed_np = packed.cpu().numpy()
    for index in range(8):
        packed_np |= res[:, index::8] << (4 * index)
    return torch.from_numpy(packed_np.astype("int32")).to(weight.device)


def _build_dense(weight, scales, group_size, out_pad, in_pad):
    padded_weight, flat_scales, group = _pad_weight_and_scales(
        weight, scales, group_size, out_pad, in_pad
    )
    w, s, k, n = _quantize_marlin(
        padded_weight, flat_scales, group, sparse=False
    )
    packed = _pack_nibbles(w, _PERM)
    return packed, s, k, n


def _build_sparse(weight, scales, group_size, out_pad, in_pad, mask=None):
    padded_weight, flat_scales, group = _pad_weight_and_scales(
        weight, scales, group_size, out_pad, in_pad
    )
    w, s, k, n = _quantize_marlin(
        padded_weight, flat_scales, group, sparse=True
    )
    codes = w.t().contiguous()  # [out_pad, in_pad] unsigned INT4
    if mask is None:
        raise ValueError("Sparse-Marlin needs the checkpoint 2:4 mask")
    mask = mask.bool()
    if mask.shape != codes.shape:
        padded_mask = torch.zeros_like(codes, dtype=torch.bool)
        padded_mask.reshape(out_pad, in_pad // 4, 4)[:, :, :2] = True
        padded_mask[: mask.shape[0], : mask.shape[1]] = mask
        mask = padded_mask
    groups = codes.reshape(out_pad, in_pad // 4, 4)
    selected = mask.reshape(out_pad, in_pad // 4, 4)
    counts = selected.sum(dim=2)
    if not torch.all(counts == 2):
        raise ValueError(
            "Sparse-Marlin requires exactly two kept values per four columns"
        )
    pair_ids = torch.arange(4, device=codes.device).view(1, 1, 4)
    low = torch.where(selected, pair_ids, torch.full_like(pair_ids, 4)).amin(2)
    high = torch.where(selected, pair_ids, torch.full_like(pair_ids, -1)).amax(2)
    vals = torch.stack(
        [
            groups.gather(2, low.unsqueeze(2)).squeeze(2),
            groups.gather(2, high.unsqueeze(2)).squeeze(2),
        ],
        dim=2,
    ).reshape(out_pad, in_pad // 2)
    compressed = vals.t().contiguous()
    packed = _pack_nibbles(compressed, _PERM_24)

    nibble = (low | (high << 2)).to(torch.int64)
    nibble = nibble.reshape(out_pad, in_pad // 16, 4)
    meta = (
        nibble[:, :, 0]
        | (nibble[:, :, 1] << 4)
        | (nibble[:, :, 2] << 8)
        | (nibble[:, :, 3] << 12)
    ).to(torch.int16)
    offsets = _calculate_meta_reordering_scatter_offsets(
        out_pad, in_pad // 16, torch.int16, meta.device
    )
    meta_reordered = meta.reshape(-1).new_empty(out_pad * in_pad // 16)
    meta_reordered.scatter_(0, offsets, meta.reshape(-1))
    meta_reordered = meta_reordered.view(out_pad, in_pad // 16).contiguous()
    return packed, meta_reordered, s, k, n


def _padded_shape(out_features: int, in_features: int) -> tuple[int, int]:
    out_pad = (out_features + 255) // 256 * 256
    in_pad = (in_features + 127) // 128 * 128
    return out_pad, in_pad


class _MarlinBase:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        padded_in: int,
        padded_out: int,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_in_features = int(padded_in)
        self.padded_out_features = int(padded_out)
        self.workspace = torch.zeros(
            self.padded_out_features // 128 * 16,
            device="cuda",
            dtype=torch.int32,
        )

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.half()
        if self.padded_in_features != self.in_features:
            x = torch.nn.functional.pad(
                x, (0, self.padded_in_features - self.in_features)
            )
        return x.contiguous()

    def precompile(self, x: torch.Tensor) -> None:
        self.forward(x)


class MarlinWeight(_MarlinBase):
    """Dense INT4 W4A16 Marlin weight."""

    def __init__(self, packed, scales, in_features, out_features, padded_in, padded_out):
        super().__init__(in_features, out_features, padded_in, padded_out)
        self.packed = packed.contiguous()
        self.scales = scales.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input(x)
        out = torch.empty(
            (x.shape[0], self.padded_out_features),
            device=x.device,
            dtype=x.dtype,
        )
        marlin_module().mul(
            x, self.packed, out, self.scales, self.workspace, -1, -1, -1, 8
        )
        return out[:, : self.out_features]


class SparseMarlinWeight(_MarlinBase):
    """2:4 sparse INT4 W4A16 Sparse-Marlin weight."""

    def __init__(
        self,
        packed,
        meta,
        scales,
        in_features,
        out_features,
        padded_in,
        padded_out,
    ):
        super().__init__(in_features, out_features, padded_in, padded_out)
        self.packed = packed.contiguous()
        self.meta = meta.contiguous()
        self.scales = scales.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input(x)
        out = torch.empty(
            (x.shape[0], self.padded_out_features),
            device=x.device,
            dtype=x.dtype,
        )
        marlin_module().mul_2_4(
            x,
            self.packed,
            self.meta,
            out,
            self.scales,
            self.workspace,
            -1,
            -1,
            -1,
            8,
        )
        return out[:, : self.out_features]


def build_marlin_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int | None,
    *,
    sparse: bool,
    mask: torch.Tensor | None = None,
) -> MarlinWeight | SparseMarlinWeight:
    """Build a dense Marlin or 2:4 Sparse-Marlin weight.

    ``weight`` is the fake-quantized FP16 matrix ``[out_features, in_features]``
    and ``scales`` is ``[out_features, groups]`` (compact group scales).
    """
    out_features, in_features = weight.shape
    out_pad, in_pad = _padded_shape(out_features, in_features)
    if sparse:
        packed, meta, s, k, n = _build_sparse(
            weight, scales, group_size, out_pad, in_pad, mask
        )
        return SparseMarlinWeight(
            packed, meta, s, in_features, out_features, k, n
        )
    packed, s, k, n = _build_dense(weight, scales, group_size, out_pad, in_pad)
    return MarlinWeight(packed, s, in_features, out_features, k, n)
