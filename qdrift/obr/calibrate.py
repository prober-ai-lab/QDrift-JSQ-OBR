# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped calibration and input capture.

Layers that consume the same tensor share one input distribution:

    q_proj / k_proj / v_proj  -> q_proj input
    gate_proj / up_proj       -> gate_proj input
    o_proj                    -> its own input
    down_proj                 -> its own input

Calibration follows OBR's sequential protocol: each decoder layer is replayed
with the weights compressed so far, so later groups and layers see compressed
upstream activations.  DriftOBR additionally keeps the full-precision
distribution from the initial pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "CalibrationStart",
    "LayerGroup",
    "SequentialCalibration",
    "capture_calibration_start",
    "capture_inputs",
    "layer_groups",
]


@dataclass
class LayerGroup:
    """A set of linears sharing one input and its capture point."""

    names: list[str]
    capture: str
    input_dim: int


def _block_index(block: str) -> tuple[int, str]:
    tail = block.split(".")[-1]
    if tail.isdigit():
        return int(tail), block
    return 10**9, block


def layer_groups(
    model: torch.nn.Module,
    layers: dict[str, torch.nn.Linear],
) -> list[LayerGroup]:
    """Build input-sharing groups for a Llama-style decoder."""
    modules = dict(model.named_modules())
    by_block: dict[str, dict[str, str]] = {}
    consumed: set[str] = set()
    groups: list[LayerGroup] = []

    for name in layers:
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            continue
        block = ".".join(parts[:3])
        by_block.setdefault(block, {})[parts[-1]] = name
        consumed.add(name)

    def add(names: list[str], capture: str) -> None:
        if not names:
            return
        module = modules[capture]
        groups.append(
            LayerGroup(
                names=names,
                capture=capture,
                input_dim=int(module.in_features),
            )
        )

    for block in sorted(by_block, key=_block_index):
        members = by_block[block]
        qkv = [members[s] for s in ("q_proj", "k_proj", "v_proj") if s in members]
        add(qkv, members.get("q_proj", qkv[0] if qkv else ""))
        if "o_proj" in members:
            add([members["o_proj"]], members["o_proj"])
        gate_up = [members[s] for s in ("gate_proj", "up_proj") if s in members]
        add(
            gate_up,
            members.get("gate_proj", gate_up[0] if gate_up else ""),
        )
        if "down_proj" in members:
            add([members["down_proj"]], members["down_proj"])
        grouped = set(qkv) | set(gate_up)
        if "o_proj" in members:
            grouped.add(members["o_proj"])
        if "down_proj" in members:
            grouped.add(members["down_proj"])
        for name in members.values():
            if name not in grouped:
                add([name], name)

    # Any layer that does not match the decoder naming scheme gets its own group.
    for name in layers:
        if name not in consumed or not any(name in group.names for group in groups):
            add([name], name)
    return groups


def _capture_block(capture: str) -> int | None:
    parts = capture.split(".")
    if (
        len(parts) >= 3
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdigit()
    ):
        return int(parts[2])
    return None


def _detach_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_value(item) for item in value)
    return value


def _detach_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    return {key: _detach_value(value) for key, value in kwargs.items()}


@dataclass
class CalibrationStart:
    """First-block activations and optional full-precision captures."""

    hidden: list[torch.Tensor]
    kwargs: list[dict[str, object]]
    inputs: dict[str, torch.Tensor]


@torch.no_grad()
def capture_calibration_start(
    model: torch.nn.Module,
    capture_names: list[str],
    chunks: list[torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> CalibrationStart:
    """Capture first-block inputs and, optionally, full-precision layer inputs.

    The first-block hidden states and forward kwargs are kept so the caller can
    replay each decoder layer with already-compressed weights, matching OBR's
    sequential calibration.  ``capture_names`` may be used to freeze the
    uncompressed input distribution needed by DriftOBR.

    Large calibrations (for example 128 x 2048) are captured into preallocated
    buffers instead of a list plus ``torch.cat``, which would temporarily double
    the host memory requirement.
    """
    modules = dict(model.named_modules())
    blocks = list(model.model.layers)
    if not blocks:
        raise ValueError("model has no decoder layers")
    total_tokens = sum(int(chunk.numel()) for chunk in chunks)
    buffers = {
        name: torch.empty(
            (total_tokens, int(modules[name].in_features)),
            dtype=dtype,
        )
        for name in capture_names
    }
    offsets = {name: 0 for name in capture_names}
    hidden: list[torch.Tensor] = []
    kwargs_list: list[dict[str, object]] = []
    handles = []

    def layer_hook(module, args, kwargs):
        value = args[0] if args else kwargs.get("hidden_states")
        if value is None:
            return
        hidden.append(value.detach())
        kwargs_list.append(_detach_kwargs(dict(kwargs)))

    def make_hook(name: str):
        def hook(module, inputs):
            if not inputs:
                return
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            start = offsets[name]
            end = start + int(value.shape[0])
            buffers[name][start:end].copy_(value.to(device="cpu", dtype=dtype))
            offsets[name] = end

        return hook

    handles.append(blocks[0].register_forward_pre_hook(layer_hook, with_kwargs=True))
    for name in capture_names:
        handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
    try:
        for ids in chunks:
            inputs = ids.to(device)
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    inputs = {name: buffers[name][: offsets[name]] for name in capture_names}
    return CalibrationStart(hidden=hidden, kwargs=kwargs_list, inputs=inputs)


class SequentialCalibration:
    """Replay decoder layers with the current, partially-compressed weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        start: CalibrationStart,
    ) -> None:
        self.blocks = list(model.model.layers)
        self.modules = dict(model.named_modules())
        self.hidden = list(start.hidden)
        self.kwargs = list(start.kwargs)
        self.index = 0

    @torch.no_grad()
    def _advance(self) -> None:
        if self.index + 1 >= len(self.blocks):
            raise ValueError("cannot advance beyond the final decoder layer")
        layer = self.blocks[self.index]
        outputs = []
        for hidden, kwargs in zip(self.hidden, self.kwargs, strict=True):
            value = layer(hidden, **kwargs)
            outputs.append(value[0] if isinstance(value, (tuple, list)) else value)
        self.hidden = outputs
        self.index += 1

    @torch.no_grad()
    def capture(self, capture: str) -> torch.Tensor:
        """Return the current group input after replaying compressed prefixes."""
        target = _capture_block(capture)
        if target is None:
            raise ValueError(f"cannot map capture point {capture!r} to a block")
        while self.index < target:
            self._advance()
        if target < self.index:
            raise ValueError(f"capture point {capture!r} is behind the cursor")
        layer = self.blocks[self.index]
        # q/k/v share the post-input-norm tensor.  Capture it directly when the
        # block exposes input_layernorm, avoiding one full layer forward.
        if capture.endswith(".self_attn.q_proj"):
            norm = getattr(layer, "input_layernorm", None)
            if norm is not None:
                values = [
                    norm(hidden)
                    .detach()
                    .reshape(-1, hidden.shape[-1])
                    .to(device="cpu", dtype=torch.float32)
                    for hidden in self.hidden
                ]
                return torch.cat(values, dim=0) if values else torch.empty(0)
        module = self.modules[capture]
        values: list[torch.Tensor] = []

        def hook(module, inputs):
            if not inputs:
                return
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            values.append(value.to(device="cpu", dtype=torch.float32))

        handle = module.register_forward_pre_hook(hook)
        try:
            for hidden, kwargs in zip(self.hidden, self.kwargs, strict=True):
                layer(hidden, **kwargs)
        finally:
            handle.remove()
        return torch.cat(values, dim=0) if values else torch.empty(0)


@torch.no_grad()
def capture_inputs(
    model: torch.nn.Module,
    capture_names: list[str],
    chunks: list[torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Capture flattened inputs at ``capture_names`` for every chunk.

    Inputs are kept on CPU to bound GPU memory.  For very large models reduce
    ``nsamples``/``seqlen`` or capture one group at a time from the caller.
    """
    modules = dict(model.named_modules())
    store: dict[str, list[torch.Tensor]] = {name: [] for name in capture_names}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            if not inputs:
                return
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            store[name].append(value.to(device="cpu", dtype=dtype))

        return hook

    for name in capture_names:
        handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
    try:
        for ids in chunks:
            inputs = ids.to(device)
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {
        name: torch.cat(values, dim=0) if values else torch.empty(0)
        for name, values in store.items()
    }
