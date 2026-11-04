# QDrift

QDrift compresses decoder-only LLMs with **pruning + quantization + compensation**
and writes both the compressed checkpoints and a unified benchmark result file.

Supported axes:

- sparsity: `2:4` / `4:8` require `0.5`; `unstructured` uses `--sparsity`; dense/quant baselines use no mask
- pruning structure: `2:4`, `4:8` (hybrid dense/pair) and `unstructured`
- pruning criterion: WANDA, Q-WANDA, magnitude, random
- weight/activation/KV granularity: per-tensor, per-channel, per-token, per-group
- WAKV: weight/activation/KV 1-4 bit, with INT8 Tensor Core GEMM
- 2:4 + INT4 weights (W4A16): Sparse-Marlin FP16xINT4 sparse kernel
- `2:4` / `4:8` + INT4 weights/activations (W4A4, channel/tensor scales):
  dense CUTLASS INT4 for the dense fraction, CUTLASS INT4 sparse for the
  pairwise fraction selected by `--pairwise-fraction`
- 2:4 + INT1-3/8: CUTLASS INT8 sparse Tensor Cores (standard 2:4)

Terminology:

- **2:4 / 4:8** = OBR-style arbitrary N:M. The same names also select the hybrid mask policy: a `--pairwise-fraction` of their layers use the hardware pair layout, the rest use dense INT4.
- **pairwise fraction** = fraction of least-sensitive layers using the hardware pair mask. `1.0` means all sparse pair layers; `0.0` means all dense N:M layers.
- weight bits: 1-4 (1-3 stored as a little-endian bitstream, 4 as packed nibbles)
- grid: mid-rise / mid-tread
- range: full / clip-search
- weight quantizer: RTN / GPTQ
- compensation: none / OBR / drift-OBR
- rotation: fast Hadamard / none

Default recipe follows OBR's setup: GPTQ, W4A4KV4, mid-tread, range-clip
weights, per-channel weights and per-token activations/KV, WANDA, damping
`0.1`, and `128×2048` calibration. Channel permutation is enabled by default.
For structured `2:4` / `4:8`, `--pairwise-fraction` controls the fraction of
least-sensitive layers using hardware pair masks:
`1.0` = all pairwise/sparse, `0.0` = all dense N:M.

For a pure OBR accuracy run, use:

```bash
scripts/compress.sh --structure 4:8 --pairwise-fraction 0.0 --no-channel-permutation
```

## Sparse structures

| structure | mask semantics | W4A4 kernel |
|---|---|---|
| `2:4` | hybrid of arbitrary 2:4 and 2:4 pairwise | dense INT4 / CUTLASS sparse |
| `4:8` | hybrid of arbitrary 4:8 and 4:8 pairwise | dense INT4 / CUTLASS sparse |
| `unstructured` | global top-k per output row | dense INT4/INT8 |

`2:4` and `4:8` follow OBR's accuracy semantics. They are not representable by
INT4 sparse Tensor Cores. Use `--pairwise-fraction 1.0` when all structured
layers should use the sparse pair layout.

For `range=clip`, per-channel/per-tensor weight scales use OBR's MSE shrink
search (100-grid, `maxshrink=0.8`, `L2.4`).

## Quick start

```bash
# Install dependencies in your Python environment.
pip install -r requirements.txt

# Compress.  By default this writes four corresponding checkpoints:
#   dense, sparse, quant, joint
scripts/compress.sh

# Unified benchmark: four-regime accuracy + prefill/decode efficiency.
scripts/bench.sh --tasks all

# Hardware-sparse joint checkpoint only (CUTLASS INT4 sparse).
scripts/compress.sh --structure 2:4 --pairwise-fraction 1.0 --w-gran channel \
  --w-bits 4 --a-bits 4 --kv-bits 4 --single
```

`scripts/compress.sh` and `scripts/bench.sh` are thin wrappers around
`python -m main`; all recipe defaults and argument construction live in
`main.py`.  Pass `--model llama2-7b` or set `PYTHON=/path/to/python` when needed.

Multi-GPU:

```bash
# Independent variant workers on four GPUs (default auto schedule).
scripts/compress.sh --devices 0,1,2,3

# One variant at a time, using all four GPUs inside each layer.
scripts/compress.sh --devices 0,1,2,3 --parallel intra
```

The joint worker publishes per-layer masks so the sparse baseline can run
concurrently; `q/k/v` and `gate/up` are layer-parallel, while `o/down` are
row-parallel.

## Checkpoints

One `compress` run writes:

| regime | weights | meaning |
|---|---|---|
| `dense`  | fp16 | no pruning, no quantization |
| `sparse` | fp16 + mask | pruning only |
| `quant`  | intN | quantization only |
| `joint`  | intN + mask | pruning + quantization |

Example names:

```text
qwen2.5-0.5b__0.0w16a16kv16__wanda_mt_rf_rtn_pc__none__cp__hc1d389d5
qwen2.5-0.5b__0.5x4to8w16a16kv16__wanda_mt_rf_rtn_pc__none__cp__pw0__h73418bef
qwen2.5-0.5b__0.0w4a4kv4__wanda_mt_rc_gptq_pc_an_kn__obr__cp__h3e67ec5e
qwen2.5-0.5b__0.5x4to8w4a4kv4__wanda_mt_rc_gptq_pc_an_kn__obr__cp__pw0.4
```

Use `--single` to write only the joint checkpoint.

## Results

All benchmark tasks merge into one file under:

```text
results/benchmarks/<checkpoint_id>__<tasks>.json
```

The unified benchmark loads each of the four real checkpoints once and reports:

- `accuracy`: PPL and zero-shot
- `efficiency`: real-kernel prefill/decode latency, throughput, load time,
  peak memory, checkpoint size and compression ratio

## Documentation

- [Compression](docs/compression.md)
- [Benchmark](docs/benchmark.md)
- [Architecture](docs/architecture.md)


## Standalone sparse/quant inference benchmark

`main benchmark --engine inference` loads the four QDrift checkpoints and
measures them with the graph-friendly runtime:

| regime | weights | kernel |
|---|---|---|
| `dense` | FP16 | cuBLAS + SDPA |
| `sparse_fp16` | 2:4 sparse FP16 | PyTorch CUTLASS sparse |
| `int4_marlin` | INT4 W4A16 | Marlin |
| `joint_sparse_int4` | standard 2:4 + INT4 W4A16 | Sparse-Marlin |

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m main benchmark \
  --model qwen2.5-0.5b \
  --structure 2:4 --sparsity 0.5 --criterion qwanda --lambda 0.4 \
  --w-bits 4 --w-gran channel \
  --a-bits 4 --kv-bits 4 \
  --quant gptq --comp none --rotation none --no-channel-permutation \
  --pairwise-fraction 1.0 \
  --tasks e2e --engine inference --batch-size 8 \
  --e2e-seqlen 1024 --e2e-decode-tokens 64 \
  --e2e-warmup 2 --e2e-iters 5 --replace
```

If structured sparse acceleration is unavailable, the benchmark does not abort:
`sparse`/`joint` e2e are skipped, but their PPL/zero-shot are still evaluated
with dense/int fallbacks, and `dense`/`quant` e2e keeps running.  Use
`--no-sparse-accel` to force this mode on a sparse-capable GPU.

The inference engine consumes the four QDrift checkpoints and labels them
`dense`, `sparse_fp16`, `int4_marlin`, `joint_sparse_int4`.  Use `--a-bits 4`
checkpoints for the W4A4/joint-sparse-W4A4 path.  KV precision comes from the
checkpoint recipe (`--kv-bits`); the inference engine falls back to
KV16 if they differ.  The default
`--e2e-regimes dense,quant,joint` keeps only the effective paths.  The engine
requires `--rotation none`; rotated checkpoints use `--engine hf`.  Reference
results: `results/inference/REPORT.md`.
