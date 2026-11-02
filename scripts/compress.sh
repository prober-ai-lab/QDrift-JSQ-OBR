#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One compression run.  By default it writes four corresponding checkpoints:
#   dense  = fp16, no pruning, no quantization
#   sparse = fp16 + joint mask
#   quant  = quantization only
#   joint  = sparse + quantization
# Add --single to write only the joint checkpoint.
#
# Edit the flags below directly; extra CLI flags are forwarded, e.g.
#
#   scripts/compress.sh --criterion qwanda --comp drift
#   scripts/compress.sh --range full --grid midtread --no-smooth
#   scripts/compress.sh --structure 2:4_pairwise --w-gran channel \
#     --w-bits 4 --a-bits 4 --k-bits 4 --v-bits 4 --single
#
# Multi-GPU examples:
#   scripts/compress.sh --devices 1,2,3,4
#   scripts/compress.sh --devices 1,2,3,4 --parallel intra
#   scripts/compress.sh --devices 1,2,3,4 --single
#
# Do not run multi-GPU with
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (the code rejects it because concurrent cuSOLVER can corrupt that allocator).
#
# Flag groups:
#   pruning       --sparsity (0.0, 0.1, ..., 0.9)
#                 --structure 2:4|4:8|2:4_pairwise --criterion --lambda
#   weight quant  --w-bits --w-gran --group-size --grid --range [--clip-rhos]
#                 --quant --gptq-alpha
#                 range=clip uses MSE shrink for channel/tensor scales
#   compensation  --comp --comp-alpha --damping
#   activation    --a-bits --a-gran --a-grid --a-clip
#   KV cache      --k-bits --v-bits --kv-gran --kv-grid --k-clip --v-clip
#   rotation      --rotation
#   smoothing     --smooth-alpha / --no-smooth
#   calibration   --nsamples --seqlen --seed
#   multi-GPU     --gpus N / --devices 0,1,... / --parallel auto|workers|intra
#
# rotation:
#   hadamard = fast global Hadamard
#   none     = identity

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-qwen2.5-0.5b}"

"${PYTHON:-python}" -m main compress \
  --model "$MODEL" \
  --w-bits 4 \
  --a-bits 16 \
  --k-bits 16 \
  --v-bits 16 \
  --sparsity 0.5 \
  --structure 2:4 \
  --criterion wanda \
  --lambda 0.4 \
  --w-gran channel \
  --grid midrise \
  --range clip \
  --clip-rhos 0.85,0.9,0.95,1.0 \
  --quant rtn \
  --gptq-alpha 0.5 \
  --comp obr \
  --comp-alpha 0.5 \
  --damping 0.1 \
  --rotation hadamard \
  --smooth-alpha 0.5 \
  --nsamples 32 \
  --seqlen 1024 \
  --device cuda \
  --seed 0 \
  --checkpoint-dir checkpoints \
  "$@"
