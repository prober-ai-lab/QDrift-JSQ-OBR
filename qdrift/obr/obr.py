# Adapted from: https://github.com/csguoh/OBR (commit 960913c)
# License: Apache-2.0
"""OBR pruning and quantization compensation.

The core compensation accepts an optional ``drift_weight`` tensor supplied by
:mod:`qdrift.obr.drift`.  Plain OBR leaves it unset; DriftOBR computes it from
the dual-path Hessian difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qdrift.quant.quantize import fake_quantize

__all__ = [
    "PruningCompensationState",
    "compensate_pruning_error",
    "compensate_quantization_error",
    "split_mask",
]


def _auto_chunk(
    size: int,
    chunk: int,
    width: int,
    cap: float = 2.0e8,
) -> int:
    """Cap the ``[batch, size, width]`` gather output to a fixed byte budget."""
    per_row = max(size, 1) * max(width, 1) * 4
    return max(1, min(chunk, int(cap / per_row)))


def _sub(matrix: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Gather ``matrix[rows[i]][:, cols[i]]`` without a full batched copy.

    ``matrix[rows]`` alone materializes ``[batch, rows, D]``.  For the
    11008-wide down projection this is >1.8 GiB before the final gather, which
    is the source of the repeated OOMs.  Keep only one row-slice temporary.
    """
    batch, row_count = rows.shape
    col_count = cols.shape[1]
    output = torch.empty(
        batch,
        row_count,
        col_count,
        device=matrix.device,
        dtype=matrix.dtype,
    )
    for index in range(batch):
        output[index] = matrix[rows[index]].gather(
            1,
            cols[index][None, :].expand(row_count, -1),
        )
    return output


def split_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact retained/dropped column indices (rows sorted)."""
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    kept = int(mask[0].sum())
    return order[:, :kept].contiguous(), order[:, kept:].contiguous()


@dataclass
class PruningCompensationState:
    """State passed from pruning compensation to quantization compensation."""

    keep: torch.Tensor
    drop: torch.Tensor
    delta: torch.Tensor
    retained: torch.Tensor
    boundary: torch.Tensor
    drift_weight: torch.Tensor | None
    hessian_raw: torch.Tensor
    hessian_system: torch.Tensor


@torch.no_grad()
def compensate_pruning_error(
    weight: torch.Tensor,
    hessian_system: torch.Tensor,
    hessian_raw: torch.Tensor,
    hessian_diff: torch.Tensor | None,
    mask: torch.Tensor,
    alpha: float = 0.5,
    chunk: int = 16,
    drift_weight: torch.Tensor | None = None,
) -> PruningCompensationState:
    """Compensate the error introduced by zeroing pruned columns."""
    rows, cols = weight.shape
    keep, drop = split_mask(mask)
    retained = keep.shape[1]
    cut = int(cols * alpha)
    boundary = torch.searchsorted(
        keep,
        torch.full((rows, 1), cut, dtype=keep.dtype, device=keep.device),
    ).squeeze(1)
    if drift_weight is None and hessian_diff is not None:
        drift_weight = weight @ hessian_diff
    delta = torch.zeros(rows, retained, device=weight.device, dtype=weight.dtype)
    if drop.shape[1] == 0:
        # No pruned columns: every row uses the same retained system.  Factor
        # it once instead of solving one 11008x11008 system per output row.
        if drift_weight is not None:
            keep_row = keep[0]
            system = hessian_system[keep_row][:, keep_row]
            rhs = drift_weight.gather(1, keep)
            delta = -solve_psd_many(system, rhs)
    else:
        step = _auto_chunk(
            max(keep.shape[1], drop.shape[1]),
            chunk,
            hessian_raw.shape[0],
        )

        for start in range(0, rows, step):
            row_slice = slice(start, start + step)
            keep_rows = keep[row_slice]
            drop_rows = drop[row_slice]
            rhs = torch.zeros(
                keep_rows.shape[0],
                retained,
                device=weight.device,
                dtype=weight.dtype,
            )
            if drift_weight is not None:
                rhs += drift_weight[row_slice].gather(1, keep_rows)
            if drop_rows.shape[1] > 0:
                rhs -= torch.bmm(
                    weight[row_slice].gather(1, drop_rows).unsqueeze(1),
                    _sub(hessian_raw, drop_rows, keep_rows),
                ).squeeze(1)
            system = _sub(hessian_system, keep_rows, keep_rows)
            delta[row_slice] = -solve_psd(system, rhs.unsqueeze(-1)).squeeze(-1)

    retained_values = weight.gather(1, keep) + delta
    return PruningCompensationState(
        keep=keep,
        drop=drop,
        delta=delta,
        retained=retained_values,
        boundary=boundary,
        drift_weight=drift_weight,
        hessian_raw=hessian_raw,
        hessian_system=hessian_system,
    )


@torch.no_grad()
def compensate_quantization_error(
    weight: torch.Tensor,
    pruning_state: PruningCompensationState,
    scale: torch.Tensor,
    qparams: tuple[int, int, float],
    grid: str,
    alpha: float = 0.5,
    chunk: int = 16,
) -> torch.Tensor:
    """Compensate the error introduced by quantizing retained columns."""
    keep = pruning_state.keep
    drop = pruning_state.drop
    hessian_raw = pruning_state.hessian_raw
    hessian_system = pruning_state.hessian_system
    boundary = pruning_state.boundary
    retained_width = keep.shape[1]
    retained_values = pruning_state.retained.clone()
    quantized = fake_quantize(retained_values, scale, qparams, grid)
    early_error = quantized - retained_values

    if drop.shape[1] == 0:
        # A dense mask has one shared partition point for every row, so the
        # late-column system can be factorized once.
        split = int(boundary[0].item()) if boundary.numel() else 0
        if split <= 0 or split >= retained_width:
            return retained_values
        early = keep[:, :split].contiguous()
        late = keep[:, split:].contiguous()
        late_row = late[0]
        system = hessian_system[late_row][:, late_row]
        rhs = torch.zeros(
            weight.shape[0],
            late.shape[1],
            device=weight.device,
            dtype=weight.dtype,
        )
        if pruning_state.drift_weight is not None:
            # DriftOBR: retain the explicit w @ dX term.  The pruning and
            # dropped-column errors are already folded into ``retained`` and
            # must not be added a second time here.
            rhs += pruning_state.drift_weight.gather(1, late)
        early_matrix = _sub(hessian_raw, early[:1], late[:1]).squeeze(0)
        rhs += early_error[:, :split] @ early_matrix
        retained_values[:, split:] += -solve_psd_many(system, rhs)
        return retained_values

    for boundary_value in torch.unique(boundary):
        split = int(boundary_value.item())
        if split <= 0 or split >= retained_width:
            continue
        rows_idx = torch.nonzero(boundary == boundary_value).squeeze(1)
        if rows_idx.numel() == 0:
            continue
        step = _auto_chunk(
            max(retained_width, drop.shape[1], split, retained_width - split),
            chunk,
            retained_width,
            cap=2.0e8,
        )
        for start in range(0, rows_idx.numel(), step):
            batch = rows_idx[start : start + step]
            keep_batch = keep[batch]
            early_batch = keep_batch[:, :split].contiguous()
            late_batch = keep_batch[:, split:].contiguous()
            rhs = torch.zeros(
                batch.numel(),
                late_batch.shape[1],
                device=weight.device,
                dtype=weight.dtype,
            )
            if pruning_state.drift_weight is not None:
                # DriftOBR: keep only w @ dX here.  ``retained_values`` already
                # contains the pruning delta and the dropped columns contribute
                # zero, so W_drop and delta must not be applied again.
                rhs += pruning_state.drift_weight[batch].gather(1, late_batch)
            rhs += torch.bmm(
                early_error[batch, :split].unsqueeze(1),
                _sub(hessian_raw, early_batch, late_batch),
            ).squeeze(1)
            system = _sub(hessian_system, late_batch, late_batch)
            correction = -solve_psd(system, rhs.unsqueeze(-1)).squeeze(-1)
            target = retained_values[batch, split:]
            target += correction
            retained_values[batch, split:] = target

    return retained_values


def _factor_psd(matrix: torch.Tensor, eps: float) -> torch.Tensor | None:
    factor, info = torch.linalg.cholesky_ex(matrix)
    if bool((info != 0).any()):
        diag = torch.diagonal(matrix, dim1=-2, dim2=-1).clamp_min(1e-12)
        mean = diag.mean(dim=-1, keepdim=True)
        eye = torch.eye(
            matrix.shape[-1],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        damped = matrix + eps * mean.unsqueeze(-1) * eye
        factor, info = torch.linalg.cholesky_ex(damped)
        if bool((info != 0).any()):
            return None
    return factor


def solve_psd(
    matrix: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Solve ``matrix @ x = rhs`` for SPD ``matrix``.

    Supports batched matrices (``[..., n, n]``).  If Cholesky fails, retry with
    a small relative diagonal damping, then fall back to ``torch.linalg.solve``.
    """
    factor = _factor_psd(matrix, eps)
    if factor is None:
        return torch.linalg.solve(matrix, rhs)
    return torch.cholesky_solve(rhs, factor)


@torch.no_grad()
def solve_psd_many(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    chunk: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Solve one SPD system against many right-hand sides.

    ``matrix`` is ``[n, n]`` and ``rhs`` is ``[rows, n]``.  The matrix is
    factorized once; rows are solved in bounded batches so the broadcast
    factorization is never materialized for all rows at once.
    """
    factor = _factor_psd(matrix, eps)
    if factor is None:
        return torch.linalg.solve(matrix, rhs.t()).t()
    rows, width = rhs.shape
    per_row = max(width * width * 4, 1)
    step = max(1, min(chunk, int(1.0e9 / per_row)))
    result = torch.empty_like(rhs)
    factor_batch = factor.unsqueeze(0)
    for start in range(0, rows, step):
        result[start : start + step] = torch.cholesky_solve(
            rhs[start : start + step].unsqueeze(-1),
            factor_batch,
        ).squeeze(-1)
    return result
