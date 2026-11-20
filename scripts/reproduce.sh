#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the headline row of the paper's main table: LLaMA-2-7B, W4A4KV4 with
# 70% unstructured sparsity, SparseGPT mask, GPTQ late half, 128x1024 WikiText-2
# calibration, seed 42.  Runs OBR and QDrift at the validation-selected mixture
# (alpha, gamma) = (0.25, 0); each writes results/v2/<run name>.json with the
# WikiText-2 test and validation perplexity (paper: 35.18 and 23.30 test).
#
# Extra flags are passed to both runs, e.g. --devices cuda:1 or --zeroshot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECIPE=(
  --model llama2-7b
  --criterion sparsegpt --structure unstructured --sparsity 0.7
  --w-bits 4 --w-gran channel --grid midtread --range clip --quant gptq
  --a-bits 4 --kv-bits 4
  --nsamples 128 --seqlen 1024 --seed 42
  --drop-checkpoint
  "$@"
)

"${PYTHON:-python}" -m main run "${RECIPE[@]}" --comp obr
"${PYTHON:-python}" -m main run "${RECIPE[@]}" --comp drift --mix-asym 0.25 --mix-clean 0
