# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the minitests.

Minitests are small, self-contained measurements that back a claim of the
paper without being a main experiment.  They import only from ``qdrift`` and
write JSON under ``results/minitests/``; the matching figure scripts live in
``visualize/plots/``.

The model is kept on the CPU and decoder blocks are streamed to the GPU, with
as many blocks resident as the free memory allows.  This lets one LLaMA-7B
minitest run on a shared 24 GB card that another job already partly uses.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The manifest stores model paths relative to the repository root.
os.chdir(ROOT)

import torch  # noqa: E402
from transformers import PreTrainedTokenizerBase  # noqa: E402

from qdrift.compress.recipe import DEFAULT_CLIP_RHOS  # noqa: E402
from qdrift.data.dataloader import get_loaders, make_calibration_chunks  # noqa: E402
from qdrift.data.tokenizer import load_tokenizer  # noqa: E402
from qdrift.models.backbone import load_backbone  # noqa: E402
from qdrift.models.rotation import (  # noqa: E402
    fuse_layer_norms,
    get_orthogonal_matrix,
    install_online_hadamards,
    rotate_attention,
    rotate_embeddings,
    rotate_lm_head,
    rotate_mlp,
)
from qdrift.obr.obr import (  # noqa: E402
    PruningCompensationState,
    compensate_pruning_error,
    compensate_quantization_error,
)
from qdrift.prune.criteria import (  # noqa: E402
    magnitude_score,
    qap_score,
    qwanda_score,
    wanda_score,
)
from qdrift.quant.quantize import fake_quantize, qrange  # noqa: E402
from qdrift.quant.scale import ScaleResult, compute_scale  # noqa: E402

MANIFEST = "modelzoo/manifest.yaml"
RESULTS = ROOT / "results" / "minitests"

# Short layer-type names and the module each one addresses inside a block.
LAYER_TYPES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
# q/k/v and gate/up read the same input, so one capture serves each group.
INPUT_SOURCE = {
    "q": "q",
    "k": "q",
    "v": "q",
    "o": "o",
    "gate": "gate",
    "up": "gate",
    "down": "down",
}


# --------------------------------------------------------------------------
# Argument parsing and naming
# --------------------------------------------------------------------------


def floats(text: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(item) for item in text.split(",") if item]


def ints(text: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    return [int(item) for item in text.split(",") if item]


def names(text: str) -> list[str]:
    """Parse a comma-separated list of names."""
    return [item for item in text.split(",") if item]


def layer_types(text: str) -> list[str]:
    """Parse and validate a comma-separated list of layer types."""
    values = names(text)
    unknown = [value for value in values if value not in LAYER_TYPES]
    if unknown:
        raise ValueError(f"unknown layer types {unknown}; use {list(LAYER_TYPES)}")
    return values


def _fmt(value: float) -> str:
    """Format a sweep value without trailing zeros."""
    return f"{value:g}"


def span(values: Sequence[float], prefix: str = "") -> str:
    """Compact tag for a sweep, e.g. ``s0.1-0.9`` or ``blk0+10+20+31``."""
    values = list(values)
    if len(values) == 1:
        return prefix + _fmt(values[0])
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    if len(values) > 2 and len(steps) == 1 and steps.pop() > 0:
        return f"{prefix}{_fmt(values[0])}-{_fmt(values[-1])}"
    return prefix + "+".join(_fmt(value) for value in values)


def gran_tag(group_size: int) -> str:
    """Filename tag of a weight granularity: ``g128`` or ``ch``."""
    return f"g{group_size}" if group_size > 0 else "ch"


def result_path(stem: str, *parts: str) -> Path:
    """``results/minitests/<stem>__<part>__...json``."""
    return RESULTS / ("__".join([stem, *[part for part in parts if part]]) + ".json")


def git_rev() -> str:
    """Short commit hash of the code that produced a result."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # Remote copies are plain rsync trees without .git.
        return os.environ.get("QDRIFT_REV", "unknown")


def save_result(path: Path, stem: str, config: dict, **tables: object) -> None:
    """Atomically write one minitest result (config + row tables)."""
    payload = {
        "minitest": stem,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_rev(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "config": config,
        **tables,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def log(message: str) -> None:
    """Print a timestamped progress line."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# --------------------------------------------------------------------------
# Model, data, and block streaming
# --------------------------------------------------------------------------


@torch.no_grad()
def load_model(
    model_name: str,
    rotation: str = "hadamard",
    seed: int = 42,
    device: str = "cuda",
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """Load a CPU-resident backbone rotated exactly like ``compress``.

    The rotation is applied block by block on ``device`` so the fp64
    temporaries never need the whole model on the GPU.
    """
    model = load_backbone(model_name, MANIFEST, device="cpu")
    tokenizer = load_tokenizer(model_name, MANIFEST)
    if rotation != "none":
        fuse_layer_norms(model)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        config = model.config
        head_dim = config.hidden_size // config.num_attention_heads
        q = get_orthogonal_matrix(config.hidden_size, rotation, device, seed)
        rotate_embeddings(model, q)
        rotate_lm_head(model, q)
        for layer in model.model.layers:
            layer.to(device)
            rotate_attention(layer, q, head_dim)
            rotate_mlp(layer, q)
            layer.to("cpu")
        del q
        install_online_hadamards(model)
        torch.cuda.empty_cache()
    return model, tokenizer


def batch_chunks(chunks: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    """Stack equal-length chunks into batches."""
    return [
        torch.stack(chunks[start : start + batch_size])
        for start in range(0, len(chunks), batch_size)
    ]


def calibration_batches(
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seqlen: int,
    seed: int,
    batch: int,
) -> list[torch.Tensor]:
    """WikiText-2 train chunks, drawn with the same seed as ``compress``."""
    chunks = make_calibration_chunks(
        tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed
    )
    return batch_chunks(chunks, batch)


def eval_batches(
    tokenizer: PreTrainedTokenizerBase, nsamples: int, seqlen: int, batch: int = 1
) -> list[torch.Tensor]:
    """The first ``nsamples`` chunks of the WikiText-2 test stream."""
    chunks = get_loaders(
        "wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen, eval_mode=True
    )
    return batch_chunks(chunks, batch)


class _StopForward(Exception):
    """Raised by the first-block hook to stop the forward pass."""

    pass


class BlockStreamer:
    """Run a CPU-resident decoder block by block on one GPU."""

    def __init__(
        self, model: torch.nn.Module, device: str = "cuda", reserve_gb: float = 6.0
    ) -> None:
        """Move the embedding, norm, rotary, and head to ``device``."""
        self.model = model
        self.device = torch.device(device)
        self.blocks = list(model.model.layers)
        self.reserve = int(reserve_gb * 2**30)
        base = model.model
        for module in (base.embed_tokens, base.norm, model.lm_head):
            module.to(self.device)
        if getattr(base, "rotary_emb", None) is not None:
            base.rotary_emb.to(self.device)
        self.block_bytes = sum(
            param.numel() * param.element_size()
            for param in self.blocks[0].parameters()
        )
        self.resident: set[int] = set()

    @property
    def depth(self) -> int:
        """Number of decoder blocks."""
        return len(self.blocks)

    def fit_resident(self, preferred: Iterable[int]) -> None:
        """Keep the first blocks of ``preferred`` on the GPU, as memory allows."""
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.device)
        held = len(self.resident) * self.block_bytes
        budget = max(0, (free + held - self.reserve) // self.block_bytes)
        want = list(dict.fromkeys(preferred))[: int(budget)]
        for index in sorted(self.resident - set(want)):
            self.blocks[index].to("cpu")
        for index in want:
            if index not in self.resident:
                self.blocks[index].to(self.device)
        self.resident = set(want)
        torch.cuda.empty_cache()

    @contextmanager
    def on_device(self, index: int) -> Iterator[torch.nn.Module]:
        """Yield block ``index`` on the GPU, streaming it if not resident."""
        block = self.blocks[index]
        if index in self.resident:
            yield block
            return
        block.to(self.device)
        try:
            yield block
        finally:
            block.to("cpu")

    @torch.no_grad()
    def start(
        self, batches: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """Return the first-block hidden states and forward kwargs."""
        hidden: list[torch.Tensor] = []
        kwargs: list[dict] = []

        def hook(module: torch.nn.Module, args: tuple, kw: dict) -> None:
            """Record the block input and stop the forward pass."""
            hidden.append(args[0] if args else kw["hidden_states"])
            kwargs.append(
                {key: value for key, value in kw.items() if key != "hidden_states"}
            )
            raise _StopForward

        handle = self.blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            for ids in batches:
                try:
                    self.model.model(ids.to(self.device))
                except _StopForward:
                    pass
        finally:
            handle.remove()
        return hidden, kwargs

    @torch.no_grad()
    def run(
        self,
        index: int,
        hidden: list[torch.Tensor],
        kwargs: list[dict],
        taps: dict[torch.nn.Module, Callable[[torch.Tensor], None]] | None = None,
    ) -> list[torch.Tensor]:
        """Run block ``index``; ``taps`` see each tapped module's input."""
        handles = []
        with self.on_device(index) as block:
            for module, fn in (taps or {}).items():
                handles.append(
                    module.register_forward_pre_hook(
                        lambda _module, inputs, fn=fn: fn(inputs[0])
                    )
                )
            try:
                outputs = []
                for value, kw in zip(hidden, kwargs, strict=True):
                    out = block(value, **kw)
                    outputs.append(out[0] if isinstance(out, (tuple, list)) else out)
            finally:
                for handle in handles:
                    handle.remove()
        return outputs

    @torch.no_grad()
    def nll_from(
        self,
        index: int,
        hidden: list[torch.Tensor],
        kwargs: list[dict],
        batches: list[torch.Tensor],
    ) -> float:
        """Mean next-token NLL when the model is entered at block ``index``."""
        for block in range(index, self.depth):
            hidden = self.run(block, hidden, kwargs)
        total = 0.0
        count = 0
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        for value, ids in zip(hidden, batches, strict=True):
            logits = self.model.lm_head(self.model.model.norm(value)).float()
            labels = ids.to(self.device)[:, 1:]
            total += float(
                loss(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1))
            )
            count += labels.numel()
            del logits
        return total / max(count, 1)


def block_module(
    model: torch.nn.Module, index: int, layer_type: str
) -> torch.nn.Linear:
    """Return the linear layer of one type in block ``index``."""
    return model.get_submodule(f"model.layers.{index}.{LAYER_TYPES[layer_type]}")


def layer_name(index: int, layer_type: str) -> str:
    """Full module name of a layer type in block ``index``."""
    return f"model.layers.{index}.{LAYER_TYPES[layer_type]}"


# --------------------------------------------------------------------------
# Second-order statistics
# --------------------------------------------------------------------------


class Gram:
    """Accumulate ``E[x x^T]`` of a layer input on the GPU."""

    def __init__(self, dim: int, device: torch.device) -> None:
        """Start an empty ``dim x dim`` accumulator."""
        self.sum = torch.zeros(dim, dim, device=device, dtype=torch.float32)
        self.count = 0

    def __call__(self, value: torch.Tensor) -> None:
        """Add one batch of layer inputs."""
        flat = value.reshape(-1, value.shape[-1]).float()
        self.sum.addmm_(flat.t(), flat)
        self.count += flat.shape[0]

    def hessians(self, damping: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``H`` and the damped system matrix, as in ``compress``."""
        hessian = self.sum / max(self.count, 1)
        hessian = 0.5 * (hessian + hessian.t())
        return hessian, damped(hessian, damping)


def damped(hessian: torch.Tensor, damping: float) -> torch.Tensor:
    """Add OBR's relative diagonal damping and symmetrize."""
    damp = damping * torch.diagonal(hessian).clamp_min(0).mean()
    system = hessian + damp * torch.eye(
        hessian.shape[0], device=hessian.device, dtype=hessian.dtype
    )
    return 0.5 * (system + system.t())


def block_grams(
    streamer: BlockStreamer,
    index: int,
    hidden: list[torch.Tensor],
    kwargs: list[dict],
    sources: Iterable[str],
) -> tuple[dict[str, Gram], list[torch.Tensor]]:
    """Accumulate input Grams of ``sources`` and return the block output."""
    grams: dict[str, Gram] = {}
    taps = {}
    for source in sources:
        module = block_module(streamer.model, index, source)
        grams[source] = Gram(module.in_features, streamer.device)
        taps[module] = grams[source]
    outputs = streamer.run(index, hidden, kwargs, taps)
    return grams, outputs


def output_error(delta: torch.Tensor, hessian: torch.Tensor) -> float:
    """``E ||delta x||^2 = tr(delta H delta^T)``."""
    return float((delta @ hessian).mul_(delta).sum())


# --------------------------------------------------------------------------
# Quantization, masking, and compensation on top of qdrift
# --------------------------------------------------------------------------


def weight_scale(
    weight: torch.Tensor,
    bits: int,
    grid: str,
    group_size: int,
    clip: bool = True,
) -> ScaleResult:
    """Dense-weight scale, chosen exactly as ``_quantize_layer`` does."""
    kind = "group" if group_size > 0 else "channel"
    return compute_scale(
        weight,
        bits,
        grid,
        kind,
        group_size if group_size > 0 else None,
        DEFAULT_CLIP_RHOS if clip else (1.0,),
        mse_clip=clip and kind == "channel",
    )


def quant_error(
    weight: torch.Tensor, scale: torch.Tensor, bits: int, grid: str
) -> torch.Tensor:
    """Rounding error ``Q(w) - w`` with a fixed scale."""
    return fake_quantize(weight, scale, qrange(bits, grid), grid) - weight


def score(
    criterion: str,
    weight: torch.Tensor,
    diag: torch.Tensor,
    eps: torch.Tensor | None = None,
    lam: float = 1.0,
    step: torch.Tensor | None = None,
    hinv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pruning score of one criterion (larger means keep).

    ``diag`` is ``E[x_j^2]``; QAP needs the quantization ``step`` and
    ``hinv = diag(H^-1)`` instead.
    """
    if criterion == "wanda":
        return wanda_score(weight, diag)
    if criterion == "qwanda":
        return qwanda_score(weight, diag, eps, lam)
    if criterion == "qap":
        return qap_score(weight, hinv, step)
    if criterion == "magnitude":
        return magnitude_score(weight)
    raise ValueError(f"unknown criterion {criterion}")


def pruning_state(
    weight: torch.Tensor,
    mask: torch.Tensor,
    system: torch.Tensor,
    alpha: float = 0.5,
) -> PruningCompensationState:
    """OBR pruning compensation (symmetric objective, no drift)."""
    return compensate_pruning_error(weight, system, system, None, mask, alpha=alpha)


def from_state(
    weight: torch.Tensor,
    state: PruningCompensationState,
    scale: torch.Tensor | None,
    bits: int = 0,
    grid: str = "midrise",
    alpha: float = 0.5,
) -> torch.Tensor:
    """Dense OBR weight; ``scale=None`` stops after pruning compensation."""
    if scale is None:
        values = state.retained
    else:
        qparams = qrange(bits, grid)
        kept_scale = scale.gather(1, state.keep)
        values = compensate_quantization_error(
            weight, state, kept_scale, qparams, grid, alpha=alpha
        )
        values = fake_quantize(values, kept_scale, qparams, grid)
    dense = torch.zeros_like(weight)
    dense.scatter_(1, state.keep, values)
    return dense


def plain(
    weight: torch.Tensor,
    mask: torch.Tensor | None,
    scale: torch.Tensor | None,
    bits: int = 0,
    grid: str = "midrise",
) -> torch.Tensor:
    """Uncompensated prune and/or quantize, with the dense-weight scale."""
    out = (
        weight
        if scale is None
        else fake_quantize(weight, scale, qrange(bits, grid), grid)
    )
    return out if mask is None else out * mask


def row_subset(
    rows: int, count: int, seed: int, device: torch.device | str
) -> torch.Tensor | None:
    """Sorted random rows for row-decomposable metrics; None keeps all rows."""
    if count <= 0 or count >= rows:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(rows, generator=generator)[:count].sort().values.to(device)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every minitest."""
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument("--rotation", choices=("hadamard", "none"), default="hadamard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=6.0,
        help="GPU memory left free when choosing resident blocks",
    )
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--range", choices=("clip", "full"), default="clip")
    parser.add_argument("--group-size", type=int, default=128, help="0 = per channel")
    parser.add_argument(
        "--suffix", default="", help="extra filename part, e.g. a shard tag"
    )
