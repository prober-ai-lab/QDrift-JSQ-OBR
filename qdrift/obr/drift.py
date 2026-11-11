# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DriftOBR compressed-prefix capture and drift terms."""

from __future__ import annotations

import random
from collections.abc import Callable

import torch

from qdrift.obr.calibrate import capture_inputs

__all__ = [
    "RIDGE_SCALES",
    "capture_compressed_input",
    "drift_weight",
    "heldout_errors",
    "ridge_drift_weight",
    "shrink_drift_weight",
]

# Candidate ridges of the drift regression, in units of the compensation damping;
# 1 is the original DAC and inf is OBR (no drift correction).
RIDGE_SCALES: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, float("inf"))


@torch.no_grad()
def capture_compressed_input(
    model: torch.nn.Module,
    capture: str,
    chunks: list[torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Capture one layer input from the partially compressed model."""
    return capture_inputs(model, [capture], chunks, device)[capture]


def drift_weight(weight: torch.Tensor, hessian_diff: torch.Tensor) -> torch.Tensor:
    """Project a weight matrix through the drift Hessian difference."""
    return weight @ hessian_diff


@torch.no_grad()
def shrink_drift_weight(
    shift: torch.Tensor,
    weight: torch.Tensor,
    fp_input: torch.Tensor,
    compressed_input: torch.Tensor,
    seqlen: int,
    mode: str,
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """James-Stein shrinkage of the drift term ``shift = W D`` toward OBR.

    ``D = E[(z_hat - x)^T z_hat]`` averages ``S`` calibration sequences, so
    ``shift`` is the mean of the per-sequence terms ``W D_s`` and its sampling
    variance is ``sum_s ||W D_s - shift||^2 / (S (S - 1))``.  Each output row
    (``row``) or the whole layer (``layer``) is scaled by
    ``max(0, 1 - variance / ||shift||^2)``: 1 keeps DAC, 0 falls back to OBR.
    ``act_quant`` maps ``z_hat`` to the deployed input first, as for ``D``.
    Returns the shrunk term and the factors (one per row).
    """
    count = compressed_input.shape[0] // seqlen
    if count < 2:
        raise ValueError("drift shrinkage needs at least two calibration sequences")
    squares = torch.zeros(shift.shape[0], device=shift.device, dtype=torch.float64)
    for index in range(count):
        rows = slice(index * seqlen, (index + 1) * seqlen)
        z_hat = compressed_input[rows].to(device=shift.device, dtype=torch.float32)
        if act_quant is not None:
            z_hat = act_quant(z_hat)
        x = fp_input[rows].to(device=shift.device, dtype=torch.float32)
        per_sequence = ((z_hat - x) @ weight.t()).t() @ z_hat / seqlen
        squares += per_sequence.double().pow(2).sum(dim=1)
        del z_hat, x, per_sequence
    signal = shift.double().pow(2).sum(dim=1)
    variance = (squares - count * signal).clamp_min(0) / (count * (count - 1))
    if mode == "layer":
        variance, signal = variance.sum().expand_as(signal), signal.sum().expand_as(signal)
    factor = (1 - variance / signal.clamp_min(1e-30)).clamp_min(0).to(shift.dtype)
    return shift * factor[:, None], factor


@torch.no_grad()
def _group_statistics(
    weight: torch.Tensor,
    fp_input: torch.Tensor,
    compressed_input: torch.Tensor,
    seqlen: int,
    groups: int,
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None,
    chunk: int = 4096,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Per sequence group (sequence index mod ``groups``): sums of z z^T,
    (z - x) z^T, ||W (z - x)||^2 and the token count."""
    device, dtype, n = weight.device, weight.dtype, compressed_input.shape[1]
    h = [torch.zeros(n, n, device=device, dtype=dtype) for _ in range(groups)]
    d = [torch.zeros(n, n, device=device, dtype=dtype) for _ in range(groups)]
    r = torch.zeros(groups, dtype=torch.float64)
    count = torch.zeros(groups, dtype=torch.float64)
    for start in range(0, compressed_input.shape[0], chunk):
        end = min(start + chunk, compressed_input.shape[0])
        z = compressed_input[start:end].to(device=device, dtype=dtype)
        if act_quant is not None:
            z = act_quant(z).to(dtype)
        dz = z - fp_input[start:end].to(device=device, dtype=dtype)
        group = ((torch.arange(start, end) // seqlen) % groups).to(device)
        for g in range(groups):
            sel = group == g
            if bool(sel.any()):
                zg, dg = z[sel], dz[sel]
                h[g] += zg.t() @ zg
                d[g] += dg.t() @ zg
                r[g] += float((dg @ weight.t()).double().pow(2).sum())
                count[g] += float(sel.sum())
        del z, dz
    return h, d, r, count


@torch.no_grad()
def heldout_errors(
    weight: torch.Tensor,
    train_h: torch.Tensor,
    train_d: torch.Tensor,
    held_h: torch.Tensor,
    held_d: torch.Tensor,
    held_r: float,
    damping: float,
    scales: tuple[float, ...] = RIDGE_SCALES,
) -> list[float]:
    """Held-out ``sum_t ||W (z_t - x_t) - W B_c z_t||^2`` per scale ``c``.

    ``B_c = D (H + c g I)^-1`` is fit on the training means ``train_h``,
    ``train_d`` (``g = damping * mean diag H``); ``held_*`` are the held-out
    sums of ``z z^T``, ``(z - x) z^T`` and ``||W (z - x)||^2``.  With
    ``H = Q diag(L) Q^T`` and ``A = W D Q``, the two traces reduce to
    ``s^T ((A^T A) * (Q^T H_held Q)) s`` and ``s^T colsum(A * (W D_held Q))``,
    ``s = 1 / (L + c g)``.
    """
    gamma = damping * float(torch.diagonal(train_h).clamp_min(0).mean())
    lam, q = torch.linalg.eigh(train_h)
    a = weight @ train_d @ q
    quad = ((a.t() @ a) * (q.t() @ held_h @ q)).double()
    lin = (a * (weight @ held_d @ q)).sum(dim=0).double()
    out = []
    for c in scales:
        if c == float("inf"):
            out.append(float(held_r))
            continue
        s = 1.0 / (lam.double() + c * gamma)
        out.append(float(held_r) - 2.0 * float(lin @ s) + float(s @ (quad @ s)))
    return out


@torch.no_grad()
def ridge_drift_weight(
    weight: torch.Tensor,
    fp_input: torch.Tensor,
    compressed_input: torch.Tensor,
    seqlen: int,
    damping: float,
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
    scales: tuple[float, ...] = RIDGE_SCALES,
    groups: int = 8,
    folds: int = 4,
    partitions: int = 3,
) -> tuple[torch.Tensor, dict[str, object]]:
    """DAC's drift term with a cross-validated ridge on the drift regression.

    DAC's target is ``W - W B``, ``B = D H^-1`` the least-squares regression of
    the input drift ``z - x`` on ``z``; its held-out prediction error is the
    excess layer objective (docs/dac-revision.md).  ``B_c = D (H + c g I)^-1``
    is evaluated by ``folds``-fold cross-validation over whole sequences
    (``groups`` groups, ``partitions`` seeded repartitions); the choice is the
    most regularized ``c`` within one standard error of the best.  Returns the
    drift term for the solver, ``W B_c H_sys`` with ``H_sys = H + g I`` (so the
    solver's target is ``W - W B_c``; ``c = 1`` is DAC's ``W D``, ``c = inf``
    is zero), and the cross-validation record.
    """
    h, d, r, count = _group_statistics(
        weight, fp_input, compressed_input, seqlen, groups, act_quant
    )
    h_tot, d_tot, n_tot = sum(h), sum(d), float(count.sum())
    info: dict[str, object] = {"scales": list(scales)}
    if float(r.sum()) <= 0.0:
        info["choice"] = None
        return weight @ (d_tot / max(n_tot, 1.0)), info
    cv_parts, se_parts = [], []
    for p in range(partitions):
        order = list(range(groups))
        random.Random(p).shuffle(order)
        errors, energies = [], []
        for k in range(folds):
            fold = order[k::folds]
            hh, dh = sum(h[g] for g in fold), sum(d[g] for g in fold)
            rh, nh = float(sum(r[g] for g in fold)), float(sum(count[g] for g in fold))
            errors.append(heldout_errors(
                weight, (h_tot - hh) / (n_tot - nh), (d_tot - dh) / (n_tot - nh),
                hh, dh, rh, damping, scales,
            ))
            energies.append(rh)
        e = torch.tensor(errors, dtype=torch.float64)
        rr = torch.tensor(energies, dtype=torch.float64)
        cv_parts.append(e.sum(0) / rr.sum())
        se_parts.append((e / rr[:, None]).std(dim=0, unbiased=True) / folds ** 0.5)
    cv = torch.stack(cv_parts).mean(0)
    se = torch.stack(se_parts).mean(0)
    best = int(cv.argmin())
    choice = max(i for i in range(len(scales)) if cv[i] <= cv[best] + se[best])
    info.update(cv=cv.tolist(), se=se.tolist(), choice=scales[choice])
    c = scales[choice]
    if c == float("inf"):
        return torch.zeros_like(weight), info
    hm = h_tot / n_tot
    gamma = damping * float(torch.diagonal(hm).clamp_min(0).mean())
    eye = torch.eye(hm.shape[0], device=hm.device, dtype=hm.dtype)
    shift = weight @ (d_tot / n_tot)
    regression = torch.linalg.solve(hm + c * gamma * eye, shift.t()).t()
    return regression @ (hm + gamma * eye), info
