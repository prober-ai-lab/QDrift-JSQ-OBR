# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical compression recipe, checkpoint naming, and serialization.

A checkpoint id is built from the axes that define an experiment:

    {model}__{sparsity}[x2to4|x4to8|x2to4p]w{w}a{a}kv{K}__{criterion}_{grid}_{range}_{quant}_{wgran}
    [_{agran}][_{kvgran}]__{compensation}[__h{hash}]

Example:
    llama2-7b__0.5w4a4kv4__qwanda0.7_mr_rc_rtn_pg128_an_kn__drift

The trailing ``h{hash}`` is only added when a knob that is not encoded in the
readable part differs from its default (rotation seed, exact clip set, ...).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_CLIP_RHOS: tuple[float, ...] = (0.85, 0.9, 0.95, 1.0)
SPARSITY_VALUES: tuple[float, ...] = tuple(round(index / 10, 1) for index in range(10))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", name.lower()).strip("-")


@dataclass(frozen=True)
class GranularitySpec:
    """Quantization granularity for weights, activations, or KV cache."""

    kind: str = "group"
    group_size: int | None = 128

    def tag(self) -> str:
        if self.kind == "tensor":
            return "t"
        if self.kind == "channel":
            return "c"
        if self.kind == "token":
            return "n"
        if self.kind == "group":
            return f"g{self.group_size}"
        raise ValueError(f"unknown granularity {self.kind}")

    def validate(self, scope: str) -> None:
        allowed = {
            "weight": ("tensor", "channel", "group"),
            "activation": ("tensor", "token", "group"),
            "kv": ("tensor", "token", "group"),
        }
        if scope not in allowed:
            raise ValueError(f"unknown scope {scope}")
        if self.kind not in allowed[scope]:
            raise ValueError(
                f"granularity {self.kind!r} is not valid for {scope}; "
                f"choose from {allowed[scope]}"
            )
        if self.kind == "group" and (self.group_size is None or self.group_size <= 0):
            raise ValueError("group granularity requires a positive group_size")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "group_size": self.group_size}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GranularitySpec:
        return cls(kind=str(data["kind"]), group_size=data.get("group_size"))


@dataclass(frozen=True)
class WeightQuantSpec:
    """Weight quantization: bits, granularity, grid, range, and method."""

    bits: int = 4
    granularity: GranularitySpec = field(
        default_factory=lambda: GranularitySpec("channel", None)
    )
    grid: str = "midtread"
    range: str = "clip"
    clip_rhos: tuple[float, ...] = DEFAULT_CLIP_RHOS
    method: str = "gptq"
    gptq_alpha: float = 0.5
    gptq_blocksize: int = 128

    def effective_rhos(self) -> tuple[float, ...]:
        if self.range == "full":
            return (1.0,)
        return tuple(float(rho) for rho in self.clip_rhos)

    def validate(self) -> None:
        if not 1 <= self.bits <= 16:
            raise ValueError(f"weight bits must be in [1, 16], got {self.bits}")
        if self.grid not in ("midrise", "midtread"):
            raise ValueError(f"unknown weight grid {self.grid}")
        if self.grid == "midtread" and self.bits < 2:
            raise ValueError("mid-tread weight grid requires at least 2 bits")
        if self.range not in ("full", "clip"):
            raise ValueError(f"unknown weight range {self.range}")
        if self.method not in ("rtn", "gptq"):
            raise ValueError(f"unknown weight quantization method {self.method}")
        self.granularity.validate("weight")
        if not self.effective_rhos():
            raise ValueError("clip_rhos must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "granularity": self.granularity.to_dict(),
            "grid": self.grid,
            "range": self.range,
            "clip_rhos": list(self.clip_rhos),
            "method": self.method,
            "gptq_alpha": self.gptq_alpha,
            "gptq_blocksize": self.gptq_blocksize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightQuantSpec:
        return cls(
            bits=int(data.get("bits", 4)),
            granularity=GranularitySpec.from_dict(data["granularity"]),
            grid=str(data.get("grid", "midtread")),
            range=str(data.get("range", "clip")),
            clip_rhos=tuple(data.get("clip_rhos", DEFAULT_CLIP_RHOS)),
            method=str(data.get("method", "gptq")),
            gptq_alpha=float(data.get("gptq_alpha", 0.5)),
            gptq_blocksize=int(data.get("gptq_blocksize", 128)),
        )


@dataclass(frozen=True)
class ActivationQuantSpec:
    """Activation quantization used by fake runtime evaluation."""

    bits: int = 4
    granularity: GranularitySpec = field(
        default_factory=lambda: GranularitySpec("token", None)
    )
    grid: str = "midtread"
    clip_ratio: float = 1.0

    def validate(self) -> None:
        if not 1 <= self.bits <= 16:
            raise ValueError(f"activation bits must be in [1, 16], got {self.bits}")
        if self.grid not in ("midrise", "midtread"):
            raise ValueError(f"unknown activation grid {self.grid}")
        if self.grid == "midtread" and self.bits < 2:
            raise ValueError("mid-tread activation grid requires at least 2 bits")
        if not 0.0 < self.clip_ratio <= 1.0:
            raise ValueError("activation clip_ratio must be in (0, 1]")
        self.granularity.validate("activation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "granularity": self.granularity.to_dict(),
            "grid": self.grid,
            "clip_ratio": self.clip_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivationQuantSpec:
        return cls(
            bits=int(data.get("bits", 4)),
            granularity=GranularitySpec.from_dict(data["granularity"]),
            grid=str(data.get("grid", "midtread")),
            clip_ratio=float(data.get("clip_ratio", 1.0)),
        )


@dataclass(frozen=True)
class KVQuantSpec:
    """Key/value cache quantization used by fake runtime evaluation."""

    k_bits: int = 4
    v_bits: int = 4
    granularity: GranularitySpec = field(
        default_factory=lambda: GranularitySpec("token", None)
    )
    grid: str = "midtread"
    k_clip_ratio: float = 1.0
    v_clip_ratio: float = 1.0

    def validate(self) -> None:
        if not 1 <= self.k_bits <= 16:
            raise ValueError(f"k bits must be in [1, 16], got {self.k_bits}")
        if not 1 <= self.v_bits <= 16:
            raise ValueError(f"v bits must be in [1, 16], got {self.v_bits}")
        if self.grid not in ("midrise", "midtread"):
            raise ValueError(f"unknown KV grid {self.grid}")
        if self.grid == "midtread" and min(self.k_bits, self.v_bits) < 2:
            raise ValueError("mid-tread KV grid requires at least 2 bits")
        if not 0.0 < self.k_clip_ratio <= 1.0:
            raise ValueError("k_clip_ratio must be in (0, 1]")
        if not 0.0 < self.v_clip_ratio <= 1.0:
            raise ValueError("v_clip_ratio must be in (0, 1]")
        self.granularity.validate("kv")

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "granularity": self.granularity.to_dict(),
            "grid": self.grid,
            "k_clip_ratio": self.k_clip_ratio,
            "v_clip_ratio": self.v_clip_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KVQuantSpec:
        return cls(
            k_bits=int(data.get("k_bits", data.get("bits", 4))),
            v_bits=int(data.get("v_bits", data.get("bits", 4))),
            granularity=GranularitySpec.from_dict(data["granularity"]),
            grid=str(data.get("grid", "midtread")),
            k_clip_ratio=float(data.get("k_clip_ratio", 1.0)),
            v_clip_ratio=float(data.get("v_clip_ratio", 1.0)),
        )


@dataclass(frozen=True)
class PruneSpec:
    """Pruning criterion."""

    sparsity: float = 0.5
    structure: str = "4:8"
    criterion: str = "wanda"
    lam: float = 0.7

    def validate(self) -> None:
        if self.sparsity not in SPARSITY_VALUES:
            raise ValueError(
                f"sparsity must be one of {SPARSITY_VALUES}, got {self.sparsity}"
            )
        if self.structure not in (
            "none",
            "unstructured",
            "2:4",
            "4:8",
        ):
            raise ValueError(f"unknown structure {self.structure}")
        if self.structure in ("2:4", "4:8") and self.sparsity != 0.5:
            raise ValueError(
                f"{self.structure} structured sparsity requires sparsity 0.5"
            )
        if self.structure == "none" and self.sparsity != 0.0:
            raise ValueError("no pruning requires sparsity 0.0")
        if self.criterion not in ("wanda", "qwanda", "qap", "magnitude", "random"):
            raise ValueError(f"unknown criterion {self.criterion}")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lambda must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sparsity": self.sparsity,
            "structure": self.structure,
            "criterion": self.criterion,
            "lambda": self.lam,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PruneSpec:
        return cls(
            sparsity=round(float(data.get("sparsity", data.get("ratio", 0.5))), 1),
            structure=str(data.get("structure", "4:8")),
            criterion=str(data.get("criterion", "wanda")),
            lam=float(data.get("lambda", 0.7)),
        )


@dataclass(frozen=True)
class RotationSpec:
    """Rotation used as the weight quantization basis."""

    mode: str = "hadamard"
    seed: int = 42

    def validate(self) -> None:
        if self.mode not in ("hadamard", "none"):
            raise ValueError(f"unknown rotation mode {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "seed": self.seed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationSpec:
        return cls(
            mode=str(data.get("mode", "hadamard")),
            seed=int(data.get("seed", 42)),
        )


@dataclass(frozen=True)
class CompensationSpec:
    """OBR / DriftOBR compensation."""

    method: str = "obr"
    alpha: float = 0.5
    damping: float = 0.01

    def validate(self) -> None:
        if self.method not in ("none", "obr", "drift"):
            raise ValueError(f"unknown compensation method {self.method}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("compensation alpha must be in [0, 1]")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "damping": self.damping,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompensationSpec:
        return cls(
            method=str(data.get("method", "obr")),
            alpha=float(data.get("alpha", 0.5)),
            damping=float(data.get("damping", 0.01)),
        )


@dataclass(frozen=True)
class CompressionRecipe:
    """Everything needed to reproduce one compressed checkpoint."""

    model: str = "llama2-7b"
    weight: WeightQuantSpec = field(default_factory=WeightQuantSpec)
    activation: ActivationQuantSpec = field(default_factory=ActivationQuantSpec)
    kv_cache: KVQuantSpec = field(default_factory=KVQuantSpec)
    prune: PruneSpec = field(default_factory=PruneSpec)
    rotation: RotationSpec = field(default_factory=RotationSpec)
    compensation: CompensationSpec = field(default_factory=CompensationSpec)
    nsamples: int = 128
    seqlen: int = 2048
    batch_size: int = 16
    max_layers: int = 0
    pairwise_fraction: float = 0.4
    pairwise_threshold: float = 0.0
    pairwise_layers: tuple[str, ...] = ()

    def validate(self) -> None:
        self.weight.validate()
        self.activation.validate()
        self.kv_cache.validate()
        self.prune.validate()
        self.rotation.validate()
        self.compensation.validate()
        if self.nsamples <= 0:
            raise ValueError("nsamples must be positive")
        if self.seqlen <= 0:
            raise ValueError("seqlen must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_layers < 0:
            raise ValueError("max_layers must be non-negative")
        if not 0.0 <= self.pairwise_fraction <= 1.0:
            raise ValueError("pairwise_fraction must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "weight": self.weight.to_dict(),
            "activation": self.activation.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "prune": self.prune.to_dict(),
            "rotation": self.rotation.to_dict(),
            "compensation": self.compensation.to_dict(),
            "calibration": {
                "nsamples": self.nsamples,
                "seqlen": self.seqlen,
                "batch_size": self.batch_size,
            },
            "max_layers": self.max_layers,
            "pairwise_fraction": self.pairwise_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionRecipe:
        calib = data.get("calibration", {})
        return cls(
            model=str(data.get("model", data.get("source_model", "llama2-7b"))),
            weight=WeightQuantSpec.from_dict(data["weight"]),
            activation=ActivationQuantSpec.from_dict(data["activation"]),
            kv_cache=KVQuantSpec.from_dict(data["kv_cache"]),
            prune=PruneSpec.from_dict(data["prune"]),
            rotation=RotationSpec.from_dict(data["rotation"]),
            compensation=CompensationSpec.from_dict(data["compensation"]),
            nsamples=int(calib.get("nsamples", 128)),
            seqlen=int(calib.get("seqlen", 2048)),
            batch_size=int(calib.get("batch_size", 16)),
            max_layers=int(data.get("max_layers", 0)),
            pairwise_fraction=float(data.get("pairwise_fraction", 0.4)),
            pairwise_layers=tuple(str(name) for name in data.get("pairwise_layers", ())),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CompressionRecipe:
        recipe = config.get("recipe")
        if recipe is None:
            raise ValueError("checkpoint config has no 'recipe' section")
        return cls.from_dict(recipe)

    def recipe_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def _has_hidden_nondefault(self) -> bool:
        canonical = replace(
            self,
            rotation=RotationSpec(),
            compensation=replace(
                self.compensation,
                alpha=0.5,
                damping=0.01,
            ),
            weight=replace(
                self.weight,
                clip_rhos=(
                    self.weight.clip_rhos
                    if self.weight.range == "full"
                    else DEFAULT_CLIP_RHOS
                ),
                gptq_alpha=0.5,
                gptq_blocksize=128,
            ),
            activation=replace(self.activation, grid="midtread", clip_ratio=1.0),
            kv_cache=replace(
                self.kv_cache,
                grid="midtread",
                k_clip_ratio=1.0,
                v_clip_ratio=1.0,
            ),
            nsamples=128,
            seqlen=2048,
            batch_size=16,
            max_layers=0,
            pairwise_fraction=0.4,
            pairwise_threshold=0.0,
            pairwise_layers=(),
        )
        return self.to_dict() != canonical.to_dict()

    def _sparsity_tag(self) -> str:
        tag = f"{self.prune.sparsity:.1f}"
        if self.prune.structure == "2:4":
            tag += "x2to4"
        elif self.prune.structure == "4:8":
            tag += "x4to8"
        return tag

    def _criterion_tag(self) -> str:
        if self.prune.criterion == "qwanda":
            return f"qwanda{self.prune.lam:g}"
        return self.prune.criterion

    def _kv_bits_tag(self) -> str:
        if self.kv_cache.k_bits == self.kv_cache.v_bits:
            return f"kv{self.kv_cache.k_bits}"
        return f"k{self.kv_cache.k_bits}v{self.kv_cache.v_bits}"

    def compression_tag(self) -> str:
        """The readable, compression-only part of the checkpoint id."""
        prefix = (
            f"{self._sparsity_tag()}"
            f"w{self.weight.bits}"
            f"a{self.activation.bits}"
            f"{self._kv_bits_tag()}"
        )
        grid = "mr" if self.weight.grid == "midrise" else "mt"
        range_tag = "rf" if self.weight.range == "full" else "rc"
        parts = [
            self._criterion_tag(),
            grid,
            range_tag,
            self.weight.method,
            f"p{self.weight.granularity.tag()}",
        ]
        if self.activation.bits < 16:
            parts.append(f"a{self.activation.granularity.tag()}")
        if max(self.kv_cache.k_bits, self.kv_cache.v_bits) < 16:
            parts.append(f"k{self.kv_cache.granularity.tag()}")
        infix = "_".join(parts)
        tag = f"{prefix}__{infix}__{self.compensation.method}"
        if self.prune.structure in ("2:4", "4:8"):
            tag += f"__pw{self.pairwise_fraction:g}"
        if self._has_hidden_nondefault():
            tag = f"{tag}__h{self.recipe_hash()[:8]}"
        return tag

    def checkpoint_id(self) -> str:
        return f"{_slug(self.model)}__{self.compression_tag()}"

    def base_config(self, created_at: str | None = None) -> dict[str, Any]:
        return {
            "format": "qdrift-checkpoint",
            "schema_version": 2,
            "checkpoint_id": self.checkpoint_id(),
            "model": self.model,
            "created_at": created_at
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recipe": self.to_dict(),
        }


def quant_recipe(recipe: CompressionRecipe) -> CompressionRecipe:
    """Return the quantization-only variant of a joint recipe."""
    return replace(
        recipe,
        prune=replace(recipe.prune, sparsity=0.0, structure="none"),
        pairwise_fraction=0.0,
        pairwise_layers=(),
    )


def fp16_recipe(
    recipe: CompressionRecipe,
    sparsity: float,
    structure: str = "none",
) -> CompressionRecipe:
    """Return an fp16 baseline recipe with no weight/activation/KV quantization."""
    return replace(
        recipe,
        weight=replace(
            recipe.weight,
            bits=16,
            method="rtn",
            grid="midtread",
            range="full",
            clip_rhos=(1.0,),
        ),
        activation=replace(recipe.activation, bits=16),
        kv_cache=replace(recipe.kv_cache, k_bits=16, v_bits=16),
        prune=replace(recipe.prune, sparsity=sparsity, structure=structure),
        compensation=replace(recipe.compensation, method="none"),
        pairwise_fraction=0.0,
        pairwise_layers=(),
    )


def dense_recipe(recipe: CompressionRecipe) -> CompressionRecipe:
    """Return the unpruned fp16 variant."""
    return fp16_recipe(recipe, 0.0, "none")


def sparse_recipe(recipe: CompressionRecipe) -> CompressionRecipe:
    """Return the fp16 variant with the joint pruning mask.

    The sparse baseline applies OBR pruning compensation only.  It must not
    quantize weights/activations and must not run quantization compensation.
    """
    sparse = fp16_recipe(recipe, recipe.prune.sparsity, recipe.prune.structure)
    if sparse.checkpoint_id() == dense_recipe(recipe).checkpoint_id():
        # Nothing is pruned: keep the dense baseline id so callers reuse it.
        return sparse
    return replace(sparse, compensation=replace(recipe.compensation, method="obr"))


def variant_recipes(
    recipe: CompressionRecipe,
) -> dict[str, CompressionRecipe]:
    """Return the four comparison recipes derived from one joint recipe."""
    return {
        "dense": dense_recipe(recipe),
        "sparse": sparse_recipe(recipe),
        "quant": quant_recipe(recipe),
        "joint": recipe,
    }


def recipe_from_config(config: dict[str, Any]) -> CompressionRecipe:
    """Load a recipe from a checkpoint sidecar."""
    return CompressionRecipe.from_config(config)
