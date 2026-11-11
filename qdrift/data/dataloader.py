# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration and PPL data loading: WikiText-2 and Pile."""

from __future__ import annotations

import random

import datasets
import torch
from transformers import PreTrainedTokenizerBase

__all__ = [
    "get_loaders",
    "get_pile",
    "get_wikitext2",
    "make_calibration_chunks",
]

CALIBRATION_DATASET = "wikitext2"
PPL_DATASETS = ("wikitext2",)
# Splits used for PPL, following the GPTQ / OBR / QuaRot convention.
PPL_SPLITS = {"wikitext2": "test"}


def _join_text(dataset: datasets.Dataset, key: str = "text") -> str:
    return "\n\n".join(item for item in dataset[key] if item)


def _slice_chunks(
    ids: list[int],
    seqlen: int,
    limit: int | None = None,
) -> list[torch.Tensor]:
    count = len(ids) // seqlen
    if limit is not None:
        count = min(count, limit)
    return [
        torch.tensor(ids[index * seqlen : (index + 1) * seqlen], dtype=torch.long)
        for index in range(count)
    ]


def _random_chunks(
    ids: list[int],
    nsamples: int,
    seqlen: int,
    seed: int,
) -> list[torch.Tensor]:
    if len(ids) < seqlen:
        return []
    generator = random.Random(seed)
    upper = len(ids) - seqlen
    return [
        torch.tensor(
            ids[generator.randint(0, upper) :][:seqlen],
            dtype=torch.long,
        )
        for _ in range(nsamples)
    ]


def get_wikitext2(
    nsamples: int,
    seqlen: int,
    tokenizer: PreTrainedTokenizerBase,
    eval_mode: bool = False,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Load WikiText-2 calibration (train) or PPL (test) chunks."""
    if eval_mode:
        data = datasets.load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split=PPL_SPLITS["wikitext2"],
        )
        # Keep the empty lines: the standard test stream joins every row.
        ids = tokenizer("\n\n".join(data["text"])).input_ids
        return _slice_chunks(ids, seqlen, limit=nsamples)
    data = datasets.load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )
    data = data.filter(lambda item: len(item["text"]) > 0)
    data = data.map(lambda item: {"text": item["text"].strip()})
    ids = tokenizer(_join_text(data)).input_ids
    return _random_chunks(ids, nsamples, seqlen, seed)


def get_pile(
    nsamples: int,
    seqlen: int,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Load Pile validation chunks for calibration."""
    data = datasets.load_dataset("mit-han-lab/pile-val-backup", split="validation")
    ids = tokenizer(_join_text(data.select(range(min(len(data), 1000))))).input_ids
    return _random_chunks(ids, nsamples, seqlen, seed)


def get_loaders(
    dataset: str,
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int | None = 32,
    seqlen: int = 1024,
    eval_mode: bool = False,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Return calibration or PPL chunks for a supported dataset.

    In ``eval_mode`` ``nsamples`` caps the number of chunks; None keeps all.
    """
    if dataset == "wikitext2":
        return get_wikitext2(nsamples, seqlen, tokenizer, eval_mode, seed)
    if dataset == "pile":
        if eval_mode:
            raise ValueError("pile only supports calibration")
        return get_pile(nsamples, seqlen, tokenizer, seed)
    raise ValueError(f"unknown dataset {dataset}")


def make_calibration_chunks(
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int = 32,
    seqlen: int = 1024,
    dataset: str = CALIBRATION_DATASET,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Build calibration chunks using the FlatQuant protocol."""
    return get_loaders(
        dataset,
        tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
        eval_mode=False,
        seed=seed,
    )
