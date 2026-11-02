#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified benchmark entry point.  All tasks merge into results/benchmarks.json.
#
# Tasks: all | ppl | zeroshot | e2e (comma combinations are allowed).
#
# The recipe flags below select the four checkpoints under --checkpoint-dir.
# Missing checkpoints are an error; run scripts/compress.sh with the same
# recipe first.
#
# Examples:
#   scripts/bench.sh
#   scripts/bench.sh --comp drift --grid midtread --damping 0.1
#   scripts/bench.sh --tasks ppl
#   scripts/bench.sh --structure 2:4 --tasks e2e --compile
#   scripts/bench.sh --structure 2:4_pairwise --tasks ppl,e2e
#   scripts/bench.sh --device cuda:1 --checkpoint-dir /tmp/my-checkpoints
#
# Note: --seqlen is the calibration length used in the checkpoint id;
#       --ppl-seqlen is the PPL evaluation length.  Benchmarking is
#       single-device; use --device to pick the GPU.
#
# PPL and zero-shot report dense / sparse / quant / joint.  For W4A4 sparse
# acceleration use --structure 2:4_pairwise; arbitrary 2:4/4:8 use dense INT4.
# Add --no-sparse-accel to force dense fallback on sparse-capable GPUs.
#
# Accuracy and efficiency share one runtime pass per regime: the same real
# compressed model is used for PPL/zero-shot and for prefill/decode timing.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-qwen2.5-0.5b}"

# The recipe flags below are the same ones used by compress.sh.
"${PYTHON:-python}" -m main benchmark \
  --model "$MODEL" \
  --sparsity 0.5 \
  --w-bits 4 \
  --a-bits 16 \
  --k-bits 16 \
  --v-bits 16 \
  --structure 2:4 \
  --criterion wanda \
  --lambda 0.4 \
  --w-gran channel \
  --grid midrise \
  --range clip \
  --clip-rhos 0.85,0.9,0.95,1.0 \
  --quant rtn \
  --comp obr \
  --comp-alpha 0.5 \
  --damping 0.1 \
  --rotation hadamard \
  --smooth-alpha 0.5 \
  --nsamples 32 \
  --seqlen 1024 \
  --device cuda \
  --checkpoint-dir checkpoints \
  --tasks all \
  --ppl-datasets wikitext2 \
  --ppl-seqlen 1024 \
  --ppl-chunks 32 \
  --lm-eval-batch-size 16 \
  --output results/benchmarks.json \
  "$@"
