# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INT4 CUTLASS GEMM and symmetric activation quantization from QuaRot."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from qdrift.quant.pack import pack_i4

__all__ = ["QuarotInt4Weight", "SparseInt4Weight", "compress_4to8_int4", "quarot_module"]

_MODULE = None
_CUTLASS_CANDIDATES = (
    "/data1/xiaoxiao/vllm/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass",
    "/usr/local/cutlass",
)


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


def _cutlass_include() -> str:
    explicit = os.environ.get("CUTLASS_INCLUDE")
    if explicit:
        return explicit
    for candidate in _CUTLASS_CANDIDATES:
        include = Path(candidate, "include")
        if Path(include, "cutlass", "cutlass.h").exists():
            return str(include)
    raise RuntimeError(
        "CUTLASS headers not found; set CUTLASS_INCLUDE to a CUTLASS include directory"
    )


def quarot_module():
    """Build (once) and return the QuaRot INT4 CUDA extension."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _prepare_builder()
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1] / "w4a4"
    cutlass = _cutlass_include()
    cutlass_root = Path(cutlass).parent
    extra_includes = [str(root), cutlass]
    tools = cutlass_root / "tools" / "util" / "include"
    if (tools / "cutlass" / "util" / "device_memory.h").exists():
        extra_includes.append(str(tools))
    _MODULE = load(
        name="qdrift_quarot_int4",
        sources=[
            str(root / "gemm.cu"),
            str(root / "quant.cu"),
            str(root / "dynamic_quant.cu"),
            str(root / "sparse_gemm.cu"),
            str(root / "bindings.cpp"),
        ],
        extra_include_paths=extra_includes,
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )
    return _MODULE


class QuarotInt4Weight:
    """A channelwise INT4 weight executed by the QuaRot CUTLASS kernel."""

    def __init__(
        self,
        codes: torch.Tensor,
        weight_scale: torch.Tensor,
        clip_ratio: float,
    ) -> None:
        if codes.shape[1] % 2:
            codes = torch.nn.functional.pad(codes, (0, 1))
        self.packed = pack_i4(codes.to(torch.int8)).contiguous()
        self.weight_scale = weight_scale.reshape(-1).float().half().contiguous()
        self.clip_ratio = float(clip_ratio)
        self.in_features = int(codes.shape[1])
        self.out_features = int(codes.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quarot_module().quarot_linear(
            x.half().contiguous(),
            self.packed,
            self.weight_scale,
            self.clip_ratio,
        )

    def precompile(self, x: torch.Tensor) -> None:
        self.forward(x)


def compress_4to8_int4(
    codes: torch.Tensor,
    mask: torch.Tensor,
    *,
    reorder: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a 4:8 pairwise INT4 weight into CUTLASS sparse form.

    Returns the packed compressed weight ``[rows, K/4]`` bytes (two INT4
    nibbles per byte) and the *reordered* metadata words ``[rows, K/64]``
    used by the CUTLASS sparse INT4 kernel.
    """
    module = quarot_module()
    codes = codes.to(torch.int8)
    mask = mask.bool()
    rows, cols = codes.shape
    if mask.shape != codes.shape:
        raise ValueError("mask and codes must have the same shape")
    # CUTLASS sparse INT4 requires K to be a multiple of the 256-wide
    # threadblock K tile.  Padded groups carry zero codes but still need a
    # legal pairwise metadata pattern.
    pad = (-cols) % 256
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
        tail = torch.zeros(
            rows, pad // 8, 4, 2, dtype=torch.bool, device=mask.device
        )
        tail[:, :, 0, :] = True
        tail[:, :, 2, :] = True
        mask = torch.cat([mask, tail.reshape(rows, pad)], dim=1)
        cols += pad
    pair_keep = mask.reshape(rows, cols // 8, 4, 2).all(dim=3)
    counts = pair_keep.sum(dim=2)
    if not torch.all(counts == 2):
        raise ValueError(
            "4:8 INT4 sparse weights must keep exactly two adjacent pairs "
            "in every group of eight columns"
        )
    pair_ids = torch.arange(4, device=codes.device).view(1, 1, 4)
    low = torch.where(pair_keep, pair_ids, torch.full_like(pair_ids, 4)).amin(dim=2)
    high = torch.where(pair_keep, pair_ids, torch.full_like(pair_ids, -1)).amax(dim=2)
    nibble = (low | (high << 2)).to(torch.int64)
    groups = cols // 8
    words = nibble.reshape(rows, groups // 8, 8)
    shifts = (torch.arange(8, device=codes.device) * 4).view(1, 1, 8)
    meta = (words << shifts).sum(dim=2).to(torch.int32).cpu().contiguous()
    if reorder:
        meta = module.sparse_reorder_meta(meta).cuda().contiguous()

    base = torch.arange(0, cols, 8, device=codes.device).view(1, groups)
    index = torch.stack(
        [base + 2 * low, base + 2 * low + 1, base + 2 * high, base + 2 * high + 1],
        dim=2,
    )
    compact = codes.gather(1, index.reshape(rows, -1))
    packed = pack_i4(compact.contiguous()).contiguous()
    return packed, meta


class SparseInt4Weight:
    """A 4:8 pairwise INT4 weight executed by CUTLASS sparse Tensor Cores."""

    def __init__(
        self,
        packed: torch.Tensor,
        meta: torch.Tensor,
        weight_scale: torch.Tensor,
        in_features: int,
        out_features: int,
        clip_ratio: float,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # packed stores two INT4 nibbles per byte, i.e. K / 4 bytes.
        self.padded_in_features = int(packed.shape[1] * 4)
        self.padded_out_features = int(packed.shape[0])
        self.packed = packed.contiguous()
        self.meta = meta.contiguous()
        self.weight_scale = weight_scale.reshape(-1).float().half().contiguous()
        self.clip_ratio = float(clip_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.half()
        rows = x.shape[0]
        if self.padded_in_features != self.in_features:
            x = torch.nn.functional.pad(
                x, (0, self.padded_in_features - self.in_features)
            )
        pad_rows = (-rows) % 32
        if pad_rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_rows))
        out = quarot_module().sparse_linear(
            x.contiguous(),
            self.packed,
            self.meta,
            self.weight_scale,
            self.clip_ratio,
        )
        return out[:rows, : self.out_features]

    def precompile(self, x: torch.Tensor) -> None:
        self.forward(x)


def build_sparse_int4_4to8_weight(
    codes: torch.Tensor,
    mask: torch.Tensor,
    weight_scale: torch.Tensor,
    clip_ratio: float,
) -> SparseInt4Weight:
    """Build a CUTLASS sparse INT4 weight from dense codes and a 4:8 mask."""
    rows, cols = codes.shape
    # CUTLASS sparse INT4 requires the sparse operand rows (out_features) to
    # be a multiple of 32.
    pad_rows = (-rows) % 32
    if pad_rows:
        codes = torch.nn.functional.pad(codes, (0, 0, 0, pad_rows))
        tail = torch.zeros(
            pad_rows,
            mask.shape[1] // 8,
            4,
            2,
            dtype=torch.bool,
            device=mask.device,
        )
        tail[:, :, 0, :] = True
        tail[:, :, 2, :] = True
        mask = torch.cat([mask, tail.reshape(pad_rows, mask.shape[1])], dim=0)
        weight_scale = torch.cat(
            [
                weight_scale.reshape(-1),
                torch.ones(pad_rows, device=weight_scale.device, dtype=weight_scale.dtype),
            ]
        )
    packed, meta = compress_4to8_int4(codes, mask)
    return SparseInt4Weight(
        packed,
        meta,
        weight_scale,
        cols,
        rows,
        clip_ratio,
    )
