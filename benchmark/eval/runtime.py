# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real compressed runtime, kernel dispatch, timing, and hardware info."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

import torch
from safetensors.torch import load_file

from benchmark.kernels.cutlass.fp16_2to4_matmul import (
    make_semi_structured,
    semi_structured_linear,
)
from benchmark.kernels.lib.int8_linear import (
    Int8Weight,
    build_int8_weight,
    dequantize_checkpoint_weight,
)
from benchmark.kernels.lib.w4a4 import (
    QuarotInt4Weight,
    build_sparse_int4_4to8_weight,
)
from benchmark.kernels.marlin import build_marlin_weight
from benchmark.kernels.lib.registry import select_kernel
from qdrift.compress.recipe import CompressionRecipe
from qdrift.data.tokenizer import load_tokenizer
from qdrift.models.backbone import load_backbone
from qdrift.models.permutation import (
    ChannelPermutations,
    apply_channel_permutations,
)
from qdrift.models.rotation import (
    fuse_layer_norms,
    install_online_hadamards,
    rotate_model,
)
from qdrift.quant.pack import pack_codes, unpack_codes
from qdrift.quant.scale import expand_scale

__all__ = [
    "checkpoint_stats",
    "detect_sparse_backend",
    "hardware_fingerprint",
    "load_runtime_model",
    "make_layer_case",
    "make_regime_case",
    "measure_runtime",
    "precompile_runtime",
    "run_regime_recipe",
    "summarize",
]


def detect_sparse_backend(device: str = "cuda") -> dict[str, object]:
    """Return the available sparse Tensor Core backend for ``device``."""
    info: dict[str, object] = {
        "device": device,
        "available": False,
        "backend": None,
        "capability": None,
        "reason": "CUDA is not available",
    }
    if not torch.cuda.is_available():
        return info
    parsed = torch.device(device)
    if parsed.type != "cuda":
        info["reason"] = "sparse Tensor Cores require a CUDA device"
        return info

    index = parsed.index
    if index is None:
        index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    info["device"] = torch.cuda.get_device_name(index)
    info["capability"] = f"{major}.{minor}"
    if major < 8:
        info["reason"] = "sparse Tensor Cores require compute capability >= 8.0"
        return info
    if not hasattr(torch.sparse, "to_sparse_semi_structured"):
        info["reason"] = "torch.sparse has no semi-structured sparse support"
        return info

    info.update(
        {
            "available": True,
            "backend": "cutlass",
            "reason": "sparse Tensor Cores available",
        }
    )
    return info


@dataclass
class LayerCase:
    """One quantized linear layer prepared for benchmarking."""

    name: str
    packed: torch.Tensor
    scales: torch.Tensor
    mask: torch.Tensor | None
    in_features: int
    out_features: int
    scale_group_size: int | None = None
    fp16_weight: torch.Tensor | None = None
    semi_weight: torch.Tensor | None = None
    int8_weight: Int8Weight | None = None
    quarot_weight: QuarotInt4Weight | object | None = None
    marlin_weight: object | None = None
    kernel: str = "fp16_dense"


def _unpack_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    bits = torch.arange(width, device=mask.device)
    byte = mask[:, bits // 8]
    bit = (byte >> (bits % 8)[None, :]) & 1
    return bit.bool()


def _prepare_scales(
    config: dict[str, object],
    scales: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    recipe = CompressionRecipe.from_config(config)
    granularity = recipe.weight.granularity
    if granularity.kind == "tensor":
        return scales.reshape(1, 1).expand(out_features, 1).contiguous()
    if granularity.kind in ("channel", "token"):
        return scales.reshape(out_features, 1).contiguous()
    return scales.reshape(out_features, -1).contiguous()


def make_layer_case(
    config: dict[str, object],
    tensors: dict[str, torch.Tensor],
    name: str,
) -> LayerCase:
    """Prepare one layer from a checkpoint on the current CUDA device."""
    shape = tensors[f"{name}.weight_shape"]
    out_features = int(shape[0])
    in_features = int(shape[1])
    mask = tensors.get(f"{name}.weight_mask")
    if mask is not None:
        mask = mask.cuda()
    recipe = CompressionRecipe.from_config(config)
    value_layout = str(config.get("layout", {}).get("value_layout", "full"))
    scale_group_size = None
    if value_layout == "compacted":
        kept_shape = tensors[f"{name}.weight_kept_shape"]
        kept = int(kept_shape[1])
        if mask is None:
            raise ValueError(f"{name} has compacted codes but no mask")
        mask_bool = _unpack_mask(mask, in_features)
        codes_compact = unpack_codes(
            tensors[f"{name}.weight_kept_packed"].cuda(),
            recipe.weight.bits,
            kept,
        ).to(torch.int8)
        codes_full = torch.zeros(
            out_features,
            in_features,
            device="cuda",
            dtype=torch.int8,
        )
        codes_full[mask_bool] = codes_compact.reshape(-1)
        packed, _ = pack_codes(codes_full, recipe.weight.bits)
        scale_compact = tensors[f"{name}.weight_kept_scale"].to(
            device="cuda", dtype=torch.float32
        )
        kept_scale = expand_scale(
            scale_compact,
            recipe.weight.granularity.kind,
            recipe.weight.granularity.group_size,
            out_features,
            kept,
        )
        scale_full = torch.ones(
            out_features,
            in_features,
            device="cuda",
            dtype=torch.float16,
        )
        scale_full[mask_bool] = kept_scale.reshape(-1).to(torch.float16)
        scales = scale_full
        scale_group_size = 1
    else:
        packed = tensors[f"{name}.weight_packed"].cuda()
        scales = _prepare_scales(
            config,
            tensors[f"{name}.weight_scale"].to(device="cuda", dtype=torch.float16),
            out_features,
        )
    return LayerCase(
        name=name,
        packed=packed.contiguous(),
        scales=scales,
        mask=mask,
        in_features=in_features,
        out_features=out_features,
        scale_group_size=scale_group_size,
    )


def _fp16_regime_case(
    config: dict[str, object],
    tensors: dict[str, torch.Tensor],
    name: str,
    regime: str,
    allow_sparse: bool = True,
) -> LayerCase:
    """Rebuild the real fp16 weights for the dense or sparse checkpoint."""
    shape = tensors[f"{name}.weight_shape"]
    out_features = int(shape[0])
    in_features = int(shape[1])
    mask_packed = tensors.get(f"{name}.weight_mask")
    packed_mask = mask_packed.cuda() if mask_packed is not None else None
    mask = _unpack_mask(packed_mask, in_features) if packed_mask is not None else None

    kept = int(tensors[f"{name}.weight_kept_shape"][1])
    kept_values = tensors[f"{name}.weight_kept_fp16"].to(
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.zeros(
        out_features,
        in_features,
        device="cuda",
        dtype=torch.float16,
    )
    if mask is None:
        weight.copy_(kept_values.reshape(out_features, kept))
    else:
        weight[mask] = kept_values.reshape(-1)

    recipe = CompressionRecipe.from_config(config)
    layer_meta = config.get("layers", {}).get(name, {})
    layer_structure = layer_meta.get("structure") if isinstance(layer_meta, dict) else None
    if layer_structure and layer_structure != recipe.prune.structure:
        recipe = replace(recipe, prune=replace(recipe.prune, structure=str(layer_structure)))
    semi_weight = None
    if (
        regime == "sparse"
        and recipe.prune.structure in ("2:4", "2:4_pairwise")
        and allow_sparse
    ):
        semi_weight = make_semi_structured(weight)

    return LayerCase(
        name=name,
        packed=torch.empty(0, device="cuda", dtype=torch.uint8),
        scales=torch.empty(0, device="cuda", dtype=torch.float16),
        mask=packed_mask if regime == "sparse" else None,
        in_features=in_features,
        out_features=out_features,
        scale_group_size=None,
        fp16_weight=weight,
        semi_weight=semi_weight,
    )


def make_regime_case(
    config: dict[str, object],
    tensors: dict[str, torch.Tensor],
    name: str,
    regime: str,
    device: str = "cuda",
    allow_sparse: bool = True,
) -> LayerCase:
    """Build one regime case from its real checkpoint payload.

    ``dense`` / ``sparse`` reconstruct the fp16 matrix from
    ``weight_kept_fp16``.  ``quant`` / ``joint`` use the real packed codes and
    scales; the quantization-only case drops the all-ones mask so it runs the
    dense integer kernel.
    """
    if regime in ("dense", "sparse"):
        return _fp16_regime_case(config, tensors, name, regime, allow_sparse)
    if regime not in ("quant", "joint"):
        raise ValueError(f"unknown regime {regime}")
    case = make_layer_case(config, tensors, name)
    recipe = CompressionRecipe.from_config(config)
    layer_meta = config.get("layers", {}).get(name, {})
    layer_structure = layer_meta.get("structure") if isinstance(layer_meta, dict) else None
    if layer_structure and layer_structure != recipe.prune.structure:
        recipe = replace(
            recipe,
            prune=replace(recipe.prune, structure=str(layer_structure)),
        )
    # Hardware sparse Tensor Cores consume 2:4 data at pair granularity.
    # 2:4_pairwise is the legal INT4 sparse pattern; the arbitrary 2:4/4:8
    # OBR masks stay on the dense INT4/INT8 path.
    sparse_joint = (
        regime == "joint"
        and allow_sparse
        and recipe.prune.structure in ("2:4", "2:4_pairwise", "4:8_pairwise")
    )
    sparse_pairwise = (
        regime == "joint"
        and allow_sparse
        and recipe.prune.structure in ("2:4_pairwise", "4:8_pairwise")
        and recipe.weight.granularity.kind in ("tensor", "channel")
    )
    selected = select_kernel(recipe, regime, device, allow_sparse)
    mask_bool = None
    if regime == "joint":
        if case.mask is None:
            raise ValueError(f"{name} has no sparse mask")
        mask_bool = _unpack_mask(case.mask, case.in_features)
    granularity = recipe.weight.granularity
    group_size = (
        int(granularity.group_size or case.in_features)
        if granularity.kind == "group"
        else case.in_features
    )
    if recipe.activation.bits >= 16:
        fp16_weight = dequantize_checkpoint_weight(
            recipe.weight.bits,
            recipe.weight.grid,
            case.packed,
            case.scales,
            (case.out_features, case.in_features),
            group_size,
            granularity.kind,
            mask_bool,
        ).to(torch.float16)
        marlin_group = None if granularity.kind in ("tensor", "channel") else group_size
        marlin_ok = recipe.weight.bits == 4 and (
            marlin_group is None or marlin_group == 128
        )
        if marlin_ok and case.scales.shape[1] == case.in_features:
            # Compacted layouts store elementwise scales.  Recover per-group
            # scales from the already fake-quantized weights.
            if marlin_group is None:
                amax = fp16_weight.abs().amax(dim=1, keepdim=True)
            else:
                amax = (
                    fp16_weight.abs()
                    .reshape(case.out_features, -1, marlin_group)
                    .amax(dim=2)
                )
            marlin_scales = (amax / 7.5).clamp_min(1e-8).to(torch.float16)
            if marlin_group is not None:
                marlin_scales = marlin_scales.reshape(case.out_features, -1)
        elif marlin_ok:
            marlin_scales = case.scales
        else:
            marlin_scales = None
        if marlin_scales is not None:
            case.marlin_weight = build_marlin_weight(
                fp16_weight,
                marlin_scales,
                marlin_group,
                sparse=sparse_joint,
                mask=mask_bool,
            )
            case.kernel = (
                "sparse_marlin_int4_2to4" if sparse_joint else "marlin_int4_dense"
            )
            if regime == "quant":
                case.mask = None
            return case
        case.fp16_weight = fp16_weight
        if sparse_joint:
            case.semi_weight = make_semi_structured(fp16_weight)
        case.kernel = "fp16_fallback"
        if regime == "quant":
            case.mask = None
        return case
    case.int8_weight = build_int8_weight(
        recipe.weight.bits,
        recipe.weight.grid,
        case.packed,
        case.scales,
        (case.out_features, case.in_features),
        group_size,
        granularity.kind,
        mask_bool,
        sparse=sparse_joint and recipe.weight.bits > 4,
    )
    if (
        case.int8_weight is not None
        and recipe.weight.bits <= 4
        and recipe.activation.bits < 16
        and recipe.prune.structure in ("2:4", "4:8", "2:4_pairwise", "4:8_pairwise")
        and granularity.kind in ("tensor", "channel")
    ):
        if sparse_pairwise:
            if mask_bool is None:
                raise ValueError(f"{name} has no 2:4_pairwise mask")
            case.quarot_weight = build_sparse_int4_4to8_weight(
                case.int8_weight.codes,
                mask_bool,
                case.int8_weight.scales[:, 0],
                recipe.activation.clip_ratio,
            )
        else:
            case.quarot_weight = QuarotInt4Weight(
                case.int8_weight.codes,
                case.int8_weight.scales[:, 0],
                recipe.activation.clip_ratio,
            )
        case.int8_weight = None
    case.kernel = selected.name
    if regime in ("quant", "joint"):
        # The INT8 plan owns the compute representation.  Drop the temporary
        # full-width packed codes/scales so 7B checkpoints do not accumulate
        # fp16-sized scale tensors for every layer.
        case.packed = torch.empty(0, device="cuda", dtype=torch.uint8)
        case.scales = torch.empty(0, device="cuda", dtype=torch.float16)
        case.mask = None
    if regime == "quant":
        case.mask = None
    return case


def run_regime_recipe(
    recipe: CompressionRecipe,
    case: LayerCase,
    regime: str,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run one real-weight regime on ``x`` using an already parsed recipe."""
    if regime == "dense":
        if case.fp16_weight is None:
            raise ValueError(f"{case.name} has no fp16 weight")
        return torch.nn.functional.linear(x, case.fp16_weight)
    if regime in ("quant", "joint"):
        if case.marlin_weight is not None:
            return case.marlin_weight.forward(x)
        if case.quarot_weight is not None:
            return case.quarot_weight.forward(x)
        if case.int8_weight is not None:
            return case.int8_weight.forward(
                x,
                recipe.activation.bits,
                recipe.activation.grid,
                recipe.activation.granularity.kind,
                recipe.activation.granularity.group_size,
                recipe.activation.clip_ratio,
            )
        if case.fp16_weight is None:
            raise ValueError(f"{case.name} has no quantized or fp16 weight")
        if regime == "joint":
            if case.semi_weight is None:
                raise ValueError(f"{case.name} has no sparse weight")
            return semi_structured_linear(x, case.semi_weight)
        return torch.nn.functional.linear(x, case.fp16_weight)
    if regime == "sparse":
        if case.semi_weight is not None:
            return semi_structured_linear(x, case.semi_weight)
        if case.fp16_weight is None:
            raise ValueError(f"{case.name} has no fp16 sparse fallback")
        return torch.nn.functional.linear(x, case.fp16_weight)
    raise ValueError(f"unknown regime {regime}")


class CompressedLinear(torch.nn.Module):
    """Drop-in Linear that dispatches to a real compressed kernel."""

    def __init__(
        self,
        original: torch.nn.Linear,
        case: LayerCase,
        regime: str,
        recipe: CompressionRecipe,
        smooth: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(original.in_features)
        self.out_features = int(original.out_features)
        self.case = case
        self.regime = regime
        self.recipe = recipe
        self.smooth = smooth
        self.bias = (
            original.bias.detach().to(dtype=torch.float16)
            if original.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if flat.dtype != torch.float16:
            flat = flat.to(torch.float16)
        if self.smooth is not None:
            flat = flat / self.smooth.to(device=flat.device, dtype=flat.dtype).reshape(
                1, -1
            )
        if self.case.semi_weight is not None and self.smooth is None:
            bias = (
                self.bias.to(device=flat.device, dtype=flat.dtype)
                if self.bias is not None
                else None
            )
            output = semi_structured_linear(flat, self.case.semi_weight, bias)
            return output.reshape(*shape[:-1], self.out_features)
        output = run_regime_recipe(self.recipe, self.case, self.regime, flat)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.reshape(*shape[:-1], self.out_features)

    def storage_bytes(self) -> int:
        case = self.case

        def size(tensor: torch.Tensor | None) -> int:
            if tensor is None:
                return 0
            return tensor.numel() * tensor.element_size()

        total = size(case.fp16_weight) + size(case.packed) + size(case.scales)
        total += size(case.mask)
        if case.int8_weight is not None:
            total += size(case.int8_weight.codes)
            total += size(case.int8_weight.scales)
        for weight in (case.quarot_weight, case.marlin_weight):
            if weight is not None:
                total += size(getattr(weight, "packed", None))
                total += size(getattr(weight, "meta", None))
                total += size(getattr(weight, "scales", None))
                total += size(getattr(weight, "weight_scale", None))
        if case.semi_weight is not None:
            total += size(case.semi_weight.values())
            total += size(case.semi_weight.indices())
        return int(total)


@torch.no_grad()
def load_runtime_model(
    checkpoint: str | Path,
    regime: str,
    manifest_path: str = "modelzoo/manifest.yaml",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    compile_model: bool = False,
    allow_sparse: bool = True,
) -> tuple[torch.nn.Module, object, CompressionRecipe]:
    """Load a model whose Linear layers use the real regime kernels."""
    checkpoint = Path(checkpoint)
    config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    recipe = CompressionRecipe.from_config(config)

    tensors = load_file(str(checkpoint))
    model = load_backbone(
        recipe.model,
        manifest_path,
        device=device,
        dtype=dtype,
    )
    tokenizer = load_tokenizer(recipe.model, manifest_path)
    permutations = None
    if recipe.rotation.mode != "none":
        fuse_layer_norms(model)
        torch.manual_seed(recipe.rotation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe.rotation.seed)
        rotate_model(model, recipe.rotation.mode, device, recipe.rotation.seed)
    if config.get("permutation"):
        permutations = ChannelPermutations.from_dict(config["permutation"])
        apply_channel_permutations(model, permutations)

    if compile_model:
        from torch import _dynamo as torch_dynamo

        torch_dynamo.config.suppress_errors = True
        torch_dynamo.config.cache_size_limit = 256

    modules = dict(model.named_modules())
    for name in config["layers"]:
        original = modules[name]
        case = make_regime_case(config, tensors, name, regime, device, allow_sparse)
        smooth = tensors.get(f"{name}.weight_smooth_scale")
        if smooth is not None:
            smooth = smooth.to(device=device, dtype=torch.float16)
        replacement = CompressedLinear(original, case, regime, recipe, smooth)
        if compile_model:
            replacement = torch.compile(replacement, mode="max-autotune")
        parent_name, child_name = name.rsplit(".", 1)
        setattr(modules[parent_name], child_name, replacement)
        modules[name] = replacement
        del original

    del modules
    if recipe.rotation.mode != "none":
        install_online_hadamards(
            model,
            permutations.o if permutations is not None else None,
            permutations.down if permutations is not None else None,
        )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, recipe


def _event_time(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


@torch.no_grad()
def precompile_runtime(
    model: torch.nn.Module,
    m_values: tuple[int, ...] = (1, 32),
) -> float:
    """Warm every compressed Linear for decode and prefill row buckets."""
    import time

    modules = []
    for module in model.modules():
        original = getattr(module, "_orig_mod", module)
        if isinstance(original, CompressedLinear):
            modules.append((module, original))
    start = time.perf_counter()
    for module, original in modules:
        for rows in m_values:
            case = original.case
            if case.int8_weight is not None:
                device = case.int8_weight.codes.device
            elif case.fp16_weight is not None:
                device = case.fp16_weight.device
            else:
                device = case.packed.device
            sample = torch.zeros(
                rows,
                original.in_features,
                device=device,
                dtype=torch.float16,
            )
            module(sample)
    torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.no_grad()
def measure_runtime(
    model: torch.nn.Module,
    seqlen: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
    device: str,
    kv_cache: object | None = None,
    batch: int = 1,
) -> dict[str, object]:
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (batch, seqlen), device=device)
    use_quant_cache = kv_cache is not None and (
        int(kv_cache.k_bits) < 16 or int(kv_cache.v_bits) < 16
    )

    def new_cache():
        if not use_quant_cache:
            return None
        from benchmark.eval.kv_cache import QuantizedDynamicCache

        return QuantizedDynamicCache(
            model.config,
            int(kv_cache.k_bits),
            int(kv_cache.v_bits),
            str(kv_cache.grid),
        )

    def run_prefill(cache=None):
        return model(
            ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    for _ in range(warmup):
        run_prefill(new_cache())
    torch.cuda.synchronize()

    prefill_times = [
        _event_time(lambda: run_prefill(new_cache())) for _ in range(iters)
    ]
    prefill = summarize(prefill_times)

    cache = new_cache()
    outputs = run_prefill(cache)
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    def decode_step() -> torch.Tensor:
        nonlocal past, next_token
        outputs = model(
            next_token,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        return outputs.logits

    for _ in range(warmup):
        decode_step()
    decode_times = [_event_time(decode_step) for _ in range(decode_tokens)]
    decode = summarize(decode_times)

    return {
        "batch": batch,
        "prefill": {
            **prefill,
            "tokens_per_s": batch * seqlen / max(prefill["p50_ms"] * 1e-3, 1e-12),
            "tokens_per_s_per_sequence": seqlen
            / max(prefill["p50_ms"] * 1e-3, 1e-12),
        },
        "decode": {
            **decode,
            "tokens_per_s": batch / max(decode["p50_ms"] * 1e-3, 1e-12),
            "tokens_per_s_per_sequence": 1.0
            / max(decode["p50_ms"] * 1e-3, 1e-12),
        },
    }


def checkpoint_stats(path: Path, config: dict[str, object]) -> dict[str, float]:
    """Return real file size and compression ratio versus fp16 shapes."""
    original_bytes = 0
    for meta in config.get("layers", {}).values():
        rows, cols = (int(value) for value in meta["shape"])
        original_bytes += rows * cols * 2
    checkpoint_bytes = path.stat().st_size
    return {
        "checkpoint_bytes": checkpoint_bytes,
        "compression_ratio": original_bytes / max(checkpoint_bytes, 1),
    }


def summarize(timings: list[float]) -> dict[str, float]:
    """Return mean, min, max, and percentiles for latency samples."""
    if not timings:
        raise ValueError("timings must not be empty")
    ordered = sorted(timings)
    return {
        "mean_ms": mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": ordered[int(0.50 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
    }


def hardware_fingerprint() -> dict[str, str]:
    """Return a compact hardware/runtime fingerprint."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "gpu": gpu,
        "cuda": str(torch.version.cuda),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
