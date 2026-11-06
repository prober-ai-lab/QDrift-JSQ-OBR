# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INT8 Tensor Core linear layers for WAKV and 2:4 sparse weights.

The weights are stored in the checkpoint at 1-4 bits.  At runtime they are
expanded once to signed INT8 codes plus group scales.  The actual matrix
multiply is then executed by cuBLAS INT8 Tensor Cores (torch._int_mm) or by
the 2:4 CUTLASS sparse path; no Triton dequantization is used on the GEMM
critical path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmark.kernels.cutlass.fp16_2to4_matmul import make_semi_structured
from qdrift.quant.quantize import qrange
from qdrift.quant.scale import expand_scale

__all__ = [
    "Int8Weight",
    "build_int8_weight",
    "dequantize_checkpoint_weight",
    "quantize_activations",
]


def _qmax(qmin: int, qmax: int) -> int:
    return max(abs(qmin), abs(qmax))


def _quantize_codes(
    values: torch.Tensor,
    bits: int,
    grid: str,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Return INT8 codes, per-group scales, group size and grid offset."""
    rows, cols = values.shape
    if bits < 1 or bits > 8:
        raise ValueError(f"unsupported bit width {bits}")
    if group_size <= 0 or group_size > cols:
        group_size = cols
    groups = (cols + group_size - 1) // group_size
    padded_cols = groups * group_size
    if padded_cols != cols:
        padded = torch.zeros(
            rows,
            padded_cols,
            device=values.device,
            dtype=values.dtype,
        )
        padded[:, :cols] = values
        values = padded
    view = values.reshape(rows, groups, group_size)
    amax = view.abs().amax(dim=2).clamp_min(1e-8)
    if bits == 1:
        scales = amax
        codes = torch.sign(view)
        offset = 0.0
    else:
        qmin, qmax, offset = qrange(bits, grid)
        limit = _qmax(qmin, qmax)
        scales = amax / limit
        if grid == "midrise":
            # Mid-rise levels (c + 0.5) * s are stored as the odd integers
            # 2c + 1 with step s / 2, so the INT8 GEMM is exact and an
            # explicit zero (pruned weight) stays representable.
            # Sign-magnitude floor, matching qdrift.quant.quantize.quantize.
            level = torch.floor(view.abs() / scales.unsqueeze(2))
            codes = torch.where(view < 0, -level - 1, level).clamp(qmin, qmax)
            codes = 2 * codes + 1
            scales = scales / 2
            offset = 0.0
        else:
            codes = torch.round(view / scales.unsqueeze(2)).clamp(qmin, qmax)
    codes = codes.reshape(rows, -1).to(torch.int8)[:, :cols]
    return codes, scales, group_size, offset


def quantize_activations(
    x: torch.Tensor,
    bits: int,
    grid: str,
    granularity: str,
    group_size: int | None,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Dynamically quantize activations to INT8 + per-row/group scales."""
    x = x.float()
    if clip_ratio < 1.0:
        x = x.clamp(
            -x.abs().amax(dim=1, keepdim=True) * clip_ratio,
            x.abs().amax(dim=1, keepdim=True) * clip_ratio,
        )
    if granularity == "tensor":
        group = x.shape[1]
    elif granularity == "group":
        group = int(group_size or x.shape[1])
    else:
        group = x.shape[1]
    if group <= 0 or group > x.shape[1]:
        group = x.shape[1]
    codes, scales, actual_group, offset = _quantize_codes(x, bits, grid, group)
    return codes, scales, actual_group, offset


@dataclass
class Int8Weight:
    """A checkpoint weight represented as INT8 codes and group scales."""

    codes: torch.Tensor
    scales: torch.Tensor
    group_size: int
    grid_offset: float
    codes_t: torch.Tensor | None = None
    sparse: bool = False
    sparse_packed: torch.Tensor | None = None
    sparse_meta: torch.Tensor | None = None
    sparse_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    sparse_group_offsets: list[int] | None = None

    @property
    def in_features(self) -> int:
        return int(self.codes.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.codes.shape[0])

    def _dense_mm(
        self,
        x_codes: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        lhs = x_codes[:, start:end].contiguous()
        if self.codes_t is None:
            rhs = self.codes[:, start:end].t().contiguous()
        else:
            rhs = self.codes_t[start:end]
        return torch._int_mm(lhs, rhs)

    def _sparse_group(
        self,
        x_codes: torch.Tensor,
        group_index: int,
        start: int,
        end: int,
    ) -> torch.Tensor:
        if self.sparse_groups is None:
            raise ValueError("sparse group list is missing")
        packed, meta = self.sparse_groups[group_index]
        lhs = x_codes[:, start:end].contiguous()
        rows = lhs.shape[0]
        pad_rows = (-rows) % 16
        if rows < 16:
            pad_rows += 16
        if pad_rows:
            padded = torch.zeros(
                rows + pad_rows,
                lhs.shape[1],
                device=lhs.device,
                dtype=lhs.dtype,
            )
            padded[:rows] = lhs
            lhs = padded
        result = torch._sparse_semi_structured_linear(
            lhs,
            packed,
            meta,
            out_dtype=torch.int32,
        )
        return result[: x_codes.shape[0]]

    def forward(
        self,
        x: torch.Tensor,
        activation_bits: int,
        activation_grid: str,
        activation_granularity: str,
        activation_group_size: int | None,
        activation_clip_ratio: float,
    ) -> torch.Tensor:
        """Run ``x @ weight.T`` with integer Tensor Cores."""
        rows = x.shape[0]
        if rows < 17:
            pad_rows = 17 - rows
            xpad = torch.zeros(
                17,
                x.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
            xpad[:rows] = x
        else:
            pad_rows = 0
            xpad = x
        if xpad.is_cuda and activation_granularity == "token":
            from benchmark.kernels.triton.quantize_activations import (
                quantize_activations_triton,
            )

            x_codes, x_scales, a_group, a_offset = quantize_activations_triton(
                xpad,
                activation_bits,
                activation_grid,
                activation_granularity,
                activation_group_size,
                activation_clip_ratio,
            )
        else:
            x_codes, x_scales, a_group, a_offset = quantize_activations(
                xpad,
                activation_bits,
                activation_grid,
                activation_granularity,
                activation_group_size,
                activation_clip_ratio,
            )
        groups = int(self.scales.shape[1])
        if groups == 1:
            partial = (
                self._sparse_group(x_codes, 0, 0, self.in_features)
                if self.sparse
                else self._dense_mm(x_codes, 0, self.in_features)
            )
            offset = self.grid_offset + a_offset
            if offset:
                w_sum = self.codes.sum(dim=1, dtype=torch.int32)
                a_sum = x_codes.sum(dim=1, dtype=torch.int32)
                partial = partial + (
                    self.grid_offset * a_sum[:, None]
                    + a_offset * w_sum[None, :]
                    + self.grid_offset * a_offset * self.in_features
                )
            output = (
                partial.float()
                * self.scales[:, 0].reshape(1, -1)
                * x_scales[:, 0].reshape(-1, 1)
            )
        else:
            output = torch.zeros(
                xpad.shape[0],
                self.out_features,
                device=x.device,
                dtype=torch.float32,
            )
            for index in range(groups):
                start = index * self.group_size
                end = min(start + self.group_size, self.in_features)
                if self.sparse:
                    partial = self._sparse_group(x_codes, index, start, end)
                else:
                    partial = self._dense_mm(x_codes, start, end)
                length = end - start
                a_index = min(start // a_group, x_scales.shape[1] - 1)
                w_scale = self.scales[:, index].reshape(1, -1)
                a_scale = x_scales[:, a_index].reshape(-1, 1)
                offset = self.grid_offset + a_offset
                if offset:
                    w_sum = self.codes[:, start:end].sum(dim=1, dtype=torch.int32)
                    a_sum = x_codes[:, start:end].sum(dim=1, dtype=torch.int32)
                    correction = (
                        self.grid_offset * a_sum[:, None]
                        + a_offset * w_sum[None, :]
                        + self.grid_offset * a_offset * length
                    )
                    partial = partial + correction
                output += partial.float() * w_scale * a_scale
        if pad_rows:
            output = output[:rows]
        return output.to(x.dtype)


def dequantize_checkpoint_weight(
    code_bits: int,
    code_grid: str,
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, int],
    group_size: int,
    scale_granularity: str,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    from qdrift.quant.pack import unpack_codes
    from qdrift.quant.quantize import dequantize

    rows, cols = shape
    codes = unpack_codes(packed, code_bits, cols).to(torch.int32)
    if scales.shape[1] == cols:
        expanded = scales
    elif scales.shape[1] == 1:
        if scale_granularity == "tensor":
            expanded = scales[0, 0].expand(rows, cols)
        else:
            expanded = scales.expand(rows, cols)
    else:
        expanded = expand_scale(
            scales,
            scale_granularity,
            group_size,
            rows,
            cols,
        )
    qparams = qrange(code_bits, code_grid)
    weight = dequantize(codes, expanded, qparams)
    if mask is not None:
        weight = weight * mask
    return weight


def _sparse_components(
    codes: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[tuple[torch.Tensor, torch.Tensor]],
    list[int],
]:
    cols = codes.shape[1]
    grouped = codes * mask
    if group_size >= cols:
        sparse = make_semi_structured(grouped)
        return sparse.packed, sparse.meta, [(sparse.packed, sparse.meta)], [0]
    components: list[tuple[torch.Tensor, torch.Tensor]] = []
    offsets: list[int] = []
    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)
        if end - start != 128:
            raise ValueError("2:4 INT8 Tensor Core groups must be exactly 128 columns")
        sparse = make_semi_structured(grouped[:, start:end].contiguous())
        components.append((sparse.packed, sparse.meta))
        offsets.append(start)
    return None, None, components, offsets


def _exact_codes(
    weight: torch.Tensor,
    code_grid: str,
    scales: torch.Tensor,
    group_size: int,
    scale_granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return INT8 codes that reproduce a dequantized checkpoint exactly.

    Mid-tread levels c * s keep code c and step s.  Mid-rise levels
    (c + 0.5) * s become the odd code 2c + 1 with step s / 2, so the GEMM
    needs no offset correction and masked weights stay exactly zero.
    """
    rows, cols = weight.shape
    step = scales.float()
    if step.shape[1] == 1:
        if scale_granularity == "tensor":
            step = step[0:1, 0:1].expand(rows, 1)
        group = cols
    else:
        group = int(group_size)
        if step.shape[1] * group < cols:
            raise ValueError("checkpoint scale groups do not cover the weight")
    if code_grid == "midrise":
        step = step / 2
    expanded = step.repeat_interleave(group, dim=1)[:, :cols]
    codes = torch.round(weight / expanded)
    if float((codes * expanded - weight).abs().max()) > 1e-3 * float(
        weight.abs().max().clamp_min(1e-12)
    ):
        raise ValueError("checkpoint weight is not on its quantization grid")
    return codes.to(torch.int8), step.contiguous(), group


def build_int8_weight(
    code_bits: int,
    code_grid: str,
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, int],
    group_size: int,
    scale_granularity: str,
    mask: torch.Tensor | None,
    *,
    sparse: bool,
    compute_grid: str = "midtread",
) -> Int8Weight:
    """Expand checkpoint codes and requantize them to INT8 for Tensor Cores."""
    cols = shape[1]
    weight = dequantize_checkpoint_weight(
        code_bits,
        code_grid,
        packed,
        scales,
        shape,
        group_size,
        scale_granularity,
        mask,
    )
    if 1 < code_bits <= 7 and scales.shape[1] != cols:
        # Reuse the checkpoint step exactly instead of requantizing: a fresh
        # amax-based requantization moves every weight off its trained level
        # (and onto a different grid for mid-rise checkpoints).
        codes, out_scales, actual_group = _exact_codes(
            weight,
            code_grid,
            scales,
            group_size,
            scale_granularity,
        )
        offset = 0.0
    else:
        if code_bits == 1:
            compute_grid = "midrise"
        out_group = min(cols, group_size)
        codes, out_scales, actual_group, offset = _quantize_codes(
            weight,
            code_bits,
            compute_grid,
            out_group,
        )
    if sparse:
        if mask is None:
            raise ValueError("sparse INT8 weight requires a mask")
        # Re-apply the exact 2:4 mask after requantization.
        mask_bool = mask if mask.dtype == torch.bool else mask.bool()
        codes = codes * mask_bool
        packed_sparse, meta_sparse, sparse_groups, offsets = _sparse_components(
            codes,
            mask_bool,
            actual_group,
        )
    else:
        packed_sparse = meta_sparse = None
        sparse_groups = None
        offsets = None
    return Int8Weight(
        codes=codes,
        scales=out_scales,
        group_size=actual_group,
        grid_offset=offset,
        codes_t=None if sparse else codes.t().contiguous(),
        sparse=sparse,
        sparse_packed=packed_sparse,
        sparse_meta=meta_sparse,
        sparse_groups=sparse_groups,
        sparse_group_offsets=offsets,
    )
