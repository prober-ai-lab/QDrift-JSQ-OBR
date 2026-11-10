# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output-sensitivity-aware compensation: the Kronecker objective G (x) H.

OBR compensates each output row alone, ``min_V sum_i (w_i - v_i) H (w_i - v_i)^T``.
The loss does not weigh the output directions of a layer equally, so the
rows are coupled here through the output curvature ``G = E[g g^T]``, ``g`` the
gradient of the calibration NLL at the layer output (dense model, no
training):

    min_V tr(G (V Z - W X)(V Z - W X)^T),  V zero off the mask,

whose stationarity condition on the retained entries is
``mask * (G (T - V H)) = 0`` with ``T = W H - W D`` (``W D`` DAC's drift term,
zero for OBR).  A diagonal ``G`` decouples the rows and gives OBR's (or DAC's)
solution exactly, so only the off-diagonal coupling changes anything.  ``G``
is normalized to mean diagonal 1 and damped, ``G + gamma I``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["collect_output_curvature", "couple_rows", "normalize_curvature"]


class _Accumulator:
    def __init__(self, module: torch.nn.Module, device: torch.device) -> None:
        self.g = torch.zeros(
            module.out_features, module.out_features, device=device, dtype=torch.float32
        )
        self.tokens = 0
        self.handle = module.register_forward_hook(self.forward)

    def forward(self, module, args, out) -> None:
        if out.requires_grad:
            out.register_hook(self.backward)

    def backward(self, grad: torch.Tensor) -> None:
        g = grad.reshape(-1, grad.shape[-1]).float()
        self.g.addmm_(g.t(), g)
        self.tokens += g.shape[0]


def _normal_parameters(model: torch.nn.Module) -> None:
    """Replace inference-mode parameters (rotate_model) by normal copies.

    Inference tensors cannot be saved for backward; cloning one at a time
    keeps the peak memory at one extra parameter.
    """
    copies: dict[int, torch.nn.Parameter] = {}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not param.is_inference():
                continue
            if id(param) not in copies:
                copies[id(param)] = torch.nn.Parameter(param.clone(), requires_grad=False)
            module._parameters[name] = copies[id(param)]


def collect_output_curvature(
    model: torch.nn.Module,
    chunks: list[torch.Tensor],
    layers: dict[str, torch.nn.Module],
    device: str,
    budget_bytes: float = 16e9,
) -> dict[str, torch.Tensor]:
    """``G = E[g g^T]`` per layer on the dense model, on the host in fp32.

    One forward/backward of the calibration NLL per sequence; the layers are
    covered in passes of at most ``budget_bytes`` of accumulators on ``device``.
    """
    sequences = [row for batch in chunks for row in batch.reshape(-1, batch.shape[-1])]
    _normal_parameters(model)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    embed = model.get_input_embeddings()
    handle = embed.register_forward_hook(lambda m, a, out: out.requires_grad_())
    names = list(layers)
    curvature: dict[str, torch.Tensor] = {}
    try:
        while names:
            group, size = [], 0.0
            while names and (not group or size + 4 * layers[names[0]].out_features ** 2 <= budget_bytes):
                size += 4 * layers[names[0]].out_features ** 2
                group.append(names.pop(0))
            accumulators = {name: _Accumulator(layers[name], torch.device(device)) for name in group}
            for sequence in sequences:
                ids = sequence.unsqueeze(0).to(device)
                with torch.enable_grad():
                    logits = model(ids).logits[:, :-1].float()
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
                    )
                    loss.backward()
                del logits, loss
            for name, acc in accumulators.items():
                acc.handle.remove()
                curvature[name] = (acc.g / max(acc.tokens, 1)).cpu()
            del accumulators
            torch.cuda.empty_cache()
    finally:
        handle.remove()
        for p in params:
            p.requires_grad_(True)
    return curvature


def normalize_curvature(g: torch.Tensor, damping: float) -> torch.Tensor:
    """``G / mean(diag G) + damping I``."""
    g = g / torch.diagonal(g).mean().clamp_min(1e-30)
    return g + damping * torch.eye(g.shape[0], device=g.device, dtype=g.dtype)


@torch.no_grad()
def couple_rows(
    values: torch.Tensor,
    free: torch.Tensor,
    target: torch.Tensor,
    curvature: torch.Tensor,
    system: torch.Tensor,
    iters: int,
) -> tuple[torch.Tensor, float]:
    """Move the ``free`` entries of ``values`` toward the Kronecker optimum.

    Conjugate gradients on ``free * (G X H) = free * (G (T - V H))`` from
    ``V = values`` (OBR's or DAC's row-wise solution), preconditioned by
    ``G_ii H_jj``; the other entries stay fixed.  Returns the new values and
    the final residual relative to the first.
    """
    m = free.to(values.dtype)
    inverse = m / (torch.diagonal(curvature)[:, None] * torch.diagonal(system)[None, :])
    x = values.clone()
    r = m * (curvature @ (target - values @ system))
    first = float(r.norm())
    if first == 0.0 or iters <= 0:
        return x, 0.0
    z = inverse * r
    p = z.clone()
    rz = float((r * z).sum())
    for _ in range(iters):
        ap = m * (curvature @ p @ system)
        alpha = rz / float((p * ap).sum())
        x += alpha * p
        r -= alpha * ap
        z = inverse * r
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, float(r.norm()) / first
