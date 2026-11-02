# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safetensors loading helpers for sharded and single-file checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:  # pragma: no cover - dependency check is exercised on the remote host
    from safetensors import safe_open
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "safetensors is required. On the benchmark host add the cached wheel "
        "to PYTHONPATH, for example:\n"
        "  PYTHONPATH=/data1/xiaoxiao/.cache/uv/wheels-v6/pypi/"
        "safetensors/0.8.0-cp310-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64"
    ) from error

__all__ = ["SafetensorStore"]


class SafetensorStore:
    """Read tensors by name from a possibly sharded safetensors checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        files: list[Path] = []
        self._key_to_file: dict[str, str] = {}

        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self._key_to_file = dict(index["weight_map"])
            files = sorted({self.model_dir / name for name in self._key_to_file.values()})
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(
                    f"no *.safetensors files found under {self.model_dir}"
                )

        self._handles = {}
        for path in files:
            handle = safe_open(str(path), framework="pt")
            self._handles[path.name] = handle
            if not self._key_to_file:
                for key in handle.keys():
                    self._key_to_file[key] = path.name

        if not self._key_to_file:
            raise RuntimeError(f"checkpoint {self.model_dir} contains no tensors")

    def keys(self):
        return self._key_to_file.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._key_to_file

    def get(self, name: str, device: str | torch.device = "cuda") -> torch.Tensor:
        try:
            shard = self._key_to_file[name]
        except KeyError as error:
            raise KeyError(f"{name!r} is not present in {self.model_dir}") from error
        tensor = self._handles[shard].get_tensor(name)
        if torch.device(device).type != "cpu":
            tensor = tensor.to(device)
        return tensor

    def close(self) -> None:
        self._handles.clear()
