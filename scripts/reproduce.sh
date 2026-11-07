#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compress and evaluate the headline QDrift W4A4KV4 recipe: llama2-7b, 50%
# unstructured sparsity, per-channel mid-tread weights, the QAP criterion,
# RTN and drift compensation.
#
# Extra recipe flags are passed to both steps, e.g. --model llama2-13b.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECIPE=(
  --model llama2-7b
  --sparsity 0.5 --structure unstructured --pairwise-fraction 0
  --criterion qap
  --w-bits 4 --w-gran channel --grid midtread
  --a-bits 4 --kv-bits 4
  --quant rtn --range clip --comp drift --damping 0.01
  --rotation hadamard
  --nsamples 32 --seqlen 1024 --batch-size 8
  "$@"
)

scripts/compress.sh "${RECIPE[@]}"
scripts/benchmark.sh "${RECIPE[@]}" --tasks ppl
