#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY="${PYTHON:-python}"
DEVICES="${DEVICES:-0,1,2,3}"
BENCH_DEVICES="${BENCH_DEVICES:-4}"
MODEL="${MODEL:-llama2-7b}"
LOG="${LOG:-logs/sparsequant_2to4_qwanda06_drift.log}"
mkdir -p "$(dirname "$LOG")"

RECIPE=(
  --model "$MODEL"
  --structure 2:4 --sparsity 0.5
  --criterion qwanda --lambda 0.6
  --w-bits 4 --a-bits 4 --kv-bits 4
  --w-gran channel --a-gran token --kv-gran token
  --grid midrise --range clip
  --quant rtn --comp drift --damping 0.01
  --rotation none
  --nsamples 32 --seqlen 1024 --batch-size 8
  --pairwise-fraction 0.4
)

{
  echo "[$(date -Is)] compress start"
  "$PY" -m main compress "${RECIPE[@]}" --devices "$DEVICES" --parallel auto
  echo "[$(date -Is)] compress done"
  echo "[$(date -Is)] benchmark start"
  CUDA_VISIBLE_DEVICES="$BENCH_DEVICES" "$PY" -m main benchmark "${RECIPE[@]}" \
    --tasks all \
    --engine inference \
    --e2e-regimes dense,sparse,quant,joint \
    --ppl-datasets wikitext2 --ppl-chunks 32 --ppl-seqlen 1024 \
    --e2e-seqlen 1024 --e2e-decode-tokens 64 \
    --e2e-warmup 2 --e2e-iters 5 \
    --replace
  echo "[$(date -Is)] benchmark done"
} >> "$LOG" 2>&1
