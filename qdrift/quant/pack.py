# Adapted from: https://github.com/spcl/QuaRot (commit 5008669)
# License: Apache-2.0
"""Integer packing helpers for quantized tensors."""

from __future__ import annotations

import torch


def two_compl(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Convert signed integers to their unsigned two's-complement form."""
    return torch.where(x < 0, 2**bits + x, x)


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    """Pack two signed 4-bit values into each uint8 element."""
    if not torch.is_signed(q):
        raise TypeError("q must be a signed integer tensor")

    minq = torch.tensor(-(2 ** (4 - 1)))
    maxq = torch.tensor(2 ** (4 - 1) - 1)
    if not torch.all(torch.logical_and(q >= minq, q <= maxq)):
        raise ValueError("q contains values outside the signed 4-bit range")

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    return q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)


def unpack_i4(x: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 values containing two signed 4-bit codes each."""
    if x.dtype != torch.uint8:
        raise TypeError("x must be uint8")

    out_shape = list(x.shape)
    out_shape[-1] *= 2

    x0 = (x & 0x0F).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    x1 = ((x & 0xF0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    out[:, 0::2] = x0
    out[:, 1::2] = x1
    return out.view(out_shape)


def pack_stream(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed codes into a little-endian bitstream (1-4 bits)."""
    if not 1 <= bits <= 4:
        raise ValueError(f"unsupported stream bit width {bits}")
    rows, cols = codes.shape
    packed_bytes = (cols * bits + 7) // 8
    if bits == 3:
        # A 3-bit code may straddle a byte boundary; keep one padding byte.
        packed_bytes += 1
    positions = torch.arange(cols, device=codes.device, dtype=torch.int64) * bits
    unsigned = codes.to(torch.int32) & ((1 << bits) - 1)
    flat = torch.zeros(
        rows,
        packed_bytes * 8,
        device=codes.device,
        dtype=torch.int32,
    )
    for bit_index in range(bits):
        bit = (unsigned >> bit_index) & 1
        flat.scatter_add_(
            1,
            (positions + bit_index)[None, :].expand(rows, -1),
            bit,
        )
    weights = (1 << torch.arange(8, device=codes.device)).to(torch.int32)
    return (
        (flat.reshape(rows, packed_bytes, 8) * weights)
        .sum(dim=2)
        .to(torch.uint8)
        .contiguous()
    )


def unpack_stream(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Unpack a little-endian bitstream into signed int8 codes."""
    if not 1 <= bits <= 4:
        raise ValueError(f"unsupported stream bit width {bits}")
    rows, packed_bytes = packed.shape
    flat = torch.zeros(
        rows,
        packed_bytes * 8,
        device=packed.device,
        dtype=torch.int32,
    )
    for bit_index in range(8):
        flat[:, bit_index::8] = (
            ((packed >> bit_index) & 1).to(torch.int32).reshape(rows, -1)
        )
    positions = torch.arange(cols, device=packed.device, dtype=torch.int64) * bits
    values = torch.zeros(
        rows,
        cols,
        device=packed.device,
        dtype=torch.int32,
    )
    for bit_index in range(bits):
        values |= (
            flat.gather(
                1,
                (positions + bit_index)[None, :].expand(rows, -1),
            )
            << bit_index
        )
    sign = 1 << (bits - 1)
    values = torch.where(values >= sign, values - (1 << bits), values)
    return values.to(torch.int8)


def pack_codes(codes: torch.Tensor, bits: int) -> tuple[torch.Tensor, str]:
    """Pack integer codes for storage and the Triton kernels."""
    if bits == 4:
        rows, cols = codes.shape
        even_cols = cols + (cols % 2)
        if even_cols != cols:
            padded = torch.zeros(
                (rows, even_cols),
                device=codes.device,
                dtype=torch.int8,
            )
            padded[:, :cols] = codes.to(torch.int8)
            codes = padded
        return pack_i4(codes.to(torch.int8)).contiguous(), "signed_int4_packed"
    if 1 <= bits <= 3:
        return (
            pack_stream(codes.to(torch.int8), bits),
            f"signed_int{bits}_stream",
        )
    if bits == 8:
        return codes.to(torch.int8).contiguous(), "signed_int8"
    raise ValueError(f"unsupported code width {bits}")


def unpack_codes(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Unpack stored codes to float32 ``[rows, cols]``."""
    if bits == 4:
        return unpack_i4(packed)[:, :cols].float()
    if 1 <= bits <= 3:
        return unpack_stream(packed, bits, cols).float()
    if bits == 8:
        return packed[:, :cols].float()
    raise ValueError(f"unsupported code width {bits}")
