# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QAP code removed from the active tree (archived, not importable as-is).

Taken from commit 965f0a5.  The whole-file QAP minitests and plots sit next
to this file; see ../README.md.
"""

# ruff: noqa

# --- qdrift/prune/criteria.py ------------------------------------------------
def qap_score(
    weight: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Return the quantization-aware pruning (QAP) score.

    ``S_ij = (w_ij^2 - eps_ij^2) / [H^-1]_jj``: the OBS loss of pruning
    ``w_ij`` minus the OBS loss of the rounding error ``eps_ij = Q(w_ij) - w_ij``
    it incurs if retained, with the layer's own quantizer.  Larger means keep;
    the score may be negative, which ranks a weight as cheaper to prune than to
    quantize.
    """
    return (weight.pow(2) - error.pow(2)) / hessian_inv_diag.reshape(1, -1)


# --- qdrift/compress/pipeline.py (_score branch) -----------------------------
#     if criterion == "qap":
#         return qap_score(weight, hessian_inv_diag, error)

# --- qdrift/compress/pipeline.py (_quantize_layer) ---------------------------
#     # QAP weighs each weight against its rounding error under the layer's
#     # own quantizer.
#     error = None
#     if recipe.prune.criterion == "qap":
#         error = (
#             fake_quantize(weight_rot, full_scale.scale, qparams, recipe.weight.grid)
#             - weight_rot
#         )
#     score = _score(recipe.prune.criterion, weight_rot, diag, error, hessian_system)
#     error = None

# --- qdrift/compress/pipeline.py (pairwise 2:4/4:8 ratio) --------------------
#             # QAP scores may be negative; the loss ratio needs a non-negative one.
#             loss = score.clamp_min(0)

# --- minitests/obr_diag.py (mid-tread QAP tie test) ---------------------------
def lexicographic(score: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Rank by ``score``, ties broken by ``|w|`` (larger = keep)."""
    first = weight.abs().argsort(dim=1, stable=True)
    second = score.gather(1, first).argsort(dim=1, stable=True)
    order = first.gather(1, second)
    rank = torch.empty_like(order)
    rank.scatter_(
        1, order, torch.arange(order.shape[1], device=order.device).expand_as(order)
    )
    return rank.float()


def tie_rows(layer, geo, args) -> list[dict]:
    """Mid-tread QAP ties at W2/W4 and the effect of a ``|w|`` tie-break."""
    weight, system, evalh = geo["weight"], geo["system"], geo["eval"]
    hinv = hessian_inverse_diag(system)
    rows = []
    half = weight.shape[1] // 2
    for bits in (2, 4):
        scale = scale_of(weight, bits, args, "midtread")
        error = (
            fake_quantize(weight, scale, qrange(bits, "midtread"), "midtread") - weight
        )
        score = qap_score(weight, hinv, error)
        zero = score == 0
        per_row = zero.float().mean(1)
        group_zero = zero.reshape(weight.shape[0], -1, 4).sum(2)
        out = {
            "layer": layer,
            "bits": bits,
            "zero_share": float(per_row.mean()),
            "zero_share_row_max": float(per_row.max()),
            "rows_tie_decided": float((zero.sum(1) > half).float().mean()),
            "groups24_tie_decided": float((group_zero >= 3).float().mean()),
        }
        for structure in ("unstructured", "2:4"):
            plain = mask_from_score(score, 0.5, structure)
            broken = mask_from_score(lexicographic(score, weight), 0.5, structure)
            key = "u" if structure == "unstructured" else "24"
            out[f"mask_diff_{key}"] = float((plain != broken).float().mean() / 2 / 0.5)
            if structure == "unstructured":
                totals = []
                for mask in (plain, broken):
                    state = C.pruning_state(weight, mask, system, 0.5)
                    stages = obr_stages(
                        weight,
                        mask,
                        state,
                        system,
                        scale,
                        bits,
                        0.5,
                        "gptq",
                        "midtread",
                    )
                    totals.append(float(energy(stages["v3"] - weight, evalh).sum()))
                    del state, stages
                out["obr_total_plain"] = totals[0]
                out["obr_total_tiebreak"] = totals[1]
                out["tiebreak_ratio"] = totals[1] / totals[0]
        rows.append(out)
    return rows
