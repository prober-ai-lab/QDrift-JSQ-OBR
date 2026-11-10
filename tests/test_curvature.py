# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output-curvature coupling: the Kronecker objective G (x) H.

    python -m pytest tests/test_curvature.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrift.compress.recipe import CompensationSpec, CompressionRecipe  # noqa: E402
from qdrift.compress.runs import run_name  # noqa: E402
from qdrift.mechanisms import LayerStats, get_mechanism  # noqa: E402
from qdrift.obr.curvature import couple_rows, normalize_curvature  # noqa: E402
from qdrift.obr.obr import compensate_pruning_error  # noqa: E402
from qdrift.prune.mask import mask_from_score  # noqa: E402
from qdrift.quant.quantize import qrange  # noqa: E402


def _problem(rows=12, cols=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    x = torch.randn(400, cols, generator=generator, dtype=dtype)
    x = x @ (0.3 * torch.randn(cols, cols, generator=generator, dtype=dtype) + torch.eye(cols, dtype=dtype))
    hessian = x.t() @ x / 400
    system = hessian + 0.01 * torch.diagonal(hessian).mean() * torch.eye(cols, dtype=dtype)
    y = torch.randn(400, rows, generator=generator, dtype=dtype)
    y = y @ (0.5 * torch.randn(rows, rows, generator=generator, dtype=dtype) + torch.eye(rows, dtype=dtype))
    curvature = y.t() @ y / 400
    weight = torch.randn(rows, cols, generator=generator, dtype=dtype)
    return weight, system, curvature


def _obr(weight, system, mask):
    state = compensate_pruning_error(weight, system, system, None, mask)
    return torch.zeros_like(weight).scatter_(1, state.keep, state.retained)


@pytest.mark.parametrize("structure", ["unstructured", "4:8_pairwise"])
def test_diagonal_curvature_keeps_obr(structure) -> None:
    weight, system, curvature = _problem()
    mask = mask_from_score(weight.abs(), 0.5, structure)
    start = _obr(weight, system, mask)
    got, _ = couple_rows(
        start, mask, weight @ system, torch.diag(torch.diagonal(curvature)), system, 20
    )
    assert torch.allclose(got, start, atol=1e-10)


@pytest.mark.parametrize("structure", ["unstructured", "4:8_pairwise"])
def test_coupling_reaches_the_kronecker_optimum(structure) -> None:
    weight, system, curvature = _problem()
    mask = mask_from_score(weight.abs(), 0.5, structure)
    got, _ = couple_rows(_obr(weight, system, mask), mask, weight @ system, curvature, system, 400)
    # Brute force on vec(V) (row-major): K_SS v_S = (K w)_S, K = G (x) H.
    kron = torch.kron(curvature, system)
    free = mask.reshape(-1).nonzero().squeeze(1)
    want = torch.zeros(weight.numel(), dtype=weight.dtype)
    want[free] = torch.linalg.solve(kron[free][:, free], (kron @ weight.reshape(-1))[free])
    assert torch.allclose(got, want.reshape(weight.shape), atol=1e-6)
    assert bool((got[~mask] == 0).all())


def test_mechanism_rounds_early_columns_and_keeps_them() -> None:
    weight, system, curvature = _problem(rows=8, cols=16, seed=2)
    weight, system, curvature = weight.float(), system.float(), curvature.float()
    mask = mask_from_score(weight.abs(), 0.5, "unstructured")
    scale = torch.full((8, 8), 0.25)
    qparams = qrange(4, "midtread")
    mechanism = get_mechanism("obr", 8.0)
    plain = LayerStats(mean_diag=1.0, system=system, raw=system)
    coupled = LayerStats(
        mean_diag=1.0, system=system, raw=system,
        curvature=normalize_curvature(curvature, 0.1), curvature_iters=50,
    )
    base = mechanism.compensate(weight, mask, plain, scale, qparams, "midtread", 0.5)
    got = mechanism.compensate(weight, mask, coupled, scale, qparams, "midtread", 0.5)
    assert got.values.shape == base.values.shape
    assert len(got.meta["curvature_residual"]) == 2


def test_curvature_is_named_and_round_trips() -> None:
    plain = CompressionRecipe(compensation=CompensationSpec(method="obr"))
    assert "curvature" not in plain.compensation.to_dict()
    assert run_name(plain).endswith("_seq_seed42")
    for comp, tag in (
        (CompensationSpec(method="obr", curvature="fisher"), "_seq-fisher_seed42"),
        (CompensationSpec(method="drift", drift_shrink="row", curvature="fisher"), "_qdrift-jsrow-fisher_seed42"),
        (CompensationSpec(method="obr", curvature="fisher", curvature_damping=0.01, curvature_iters=64),
         "_seq-fisher-g0.01-i64_seed42"),
    ):
        recipe = CompressionRecipe(compensation=comp)
        recipe.validate()
        assert run_name(recipe).endswith(tag)
        assert CompensationSpec.from_dict(comp.to_dict()) == comp
    with pytest.raises(ValueError):
        CompensationSpec(method="none", curvature="fisher").validate()


def test_collect_matches_manual_gradients() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from qdrift.obr.curvature import collect_output_curvature

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
    )
    model = LlamaForCausalLM(config).double().eval()
    with torch.inference_mode():  # rotate_model leaves inference tensors
        layer = model.model.layers[0].mlp.up_proj
        layer.weight = torch.nn.Parameter(layer.weight.clone())
    chunks = [torch.randint(0, 50, (2, 8))]
    layers = {"up": model.model.layers[0].mlp.up_proj, "o": model.model.layers[1].self_attn.o_proj}
    got = collect_output_curvature(model, chunks, layers, "cpu", budget_bytes=1)
    outputs = {}
    handles = [m.register_forward_hook(lambda m, a, o, k=k: outputs.__setitem__(k, o) or o.retain_grad())
               for k, m in layers.items()]
    want = {k: 0 for k in layers}
    for row in chunks[0]:
        ids = row.unsqueeze(0)
        model.zero_grad()
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        logits = model(inputs_embeds=embeds).logits[:, :-1]
        torch.nn.functional.cross_entropy(logits.reshape(-1, 50), ids[:, 1:].reshape(-1), reduction="sum").backward()
        for k, out in outputs.items():
            g = out.grad.reshape(-1, out.shape[-1])
            want[k] = want[k] + g.t() @ g
    for h in handles:
        h.remove()
    for k in layers:
        assert torch.allclose(got[k].double(), want[k] / 16, rtol=1e-4, atol=1e-8)
    assert not any(p.is_inference() for p in model.parameters())
