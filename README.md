# QDrift

QDrift compresses decoder-only LLMs with **pruning + quantization + compensation**
and writes both the compressed checkpoints and a unified benchmark result file.

Supported axes:

- sparsity: `2:4` / `4:8` / `2:4_pairwise` require `0.5`; `unstructured` uses `--sparsity`; dense/quant baselines use no mask
- pruning structure: `2:4`, `4:8` (OBR standard arbitrary N:M), `2:4_pairwise` (hardware sparse), `unstructured`
- pruning criterion: WANDA, Q-WANDA, magnitude, random
- weight/activation/KV granularity: per-tensor, per-channel, per-token, per-group
- WAKV: weight/activation/KV 1-4 bit, with INT8 Tensor Core GEMM
- 2:4 + INT4 weights (W4A16): Sparse-Marlin FP16xINT4 sparse kernel
- 2:4_pairwise + INT4 weights/activations (W4A4, channel/tensor scales):
  CUTLASS INT4 sparse Tensor Cores
- 2:4 / 4:8 + INT4 weights/activations (W4A4, channel/tensor scales):
  dense CUTLASS INT4 (OBR arbitrary N:M semantics; not representable by sparse INT4 hardware)
- 2:4 + INT1-3/8: CUTLASS INT8 sparse Tensor Cores (standard 2:4)

Terminology:

- **2:4** = any two of every four columns. This is the OBR accuracy semantics; arbitrary 2:4 is not representable by INT4 sparse Tensor Cores, but is the standard 2:4 layout for FP16/BF16/INT8 sparse MMA.
- **4:8** = OBR-style arbitrary N:M: any four of every eight columns. Also accuracy-only for INT4 sparse hardware.
- **2:4_pairwise** = one aligned 2-element pair per four columns; this is the INT4 sparse MMA layout. It is a subset of arbitrary 2:4, and CUTLASS packs two 4-column groups as 4:8 metadata.
- weight bits: 1-4 (1-3 stored as a little-endian bitstream, 4 as packed nibbles)
- grid: mid-rise / mid-tread
- range: full / clip-search
- weight quantizer: RTN / GPTQ
- compensation: none / OBR / drift-OBR
- rotation: fast Hadamard / none

## Sparse structures

| structure | mask semantics | W4A4 kernel |
|---|---|---|
| `2:4` | any two of four columns | dense INT4 |
| `4:8` | any four of eight columns | dense INT4 |
| `2:4_pairwise` | one aligned pair per four columns | CUTLASS INT4 sparse |
| `unstructured` | global top-k per output row | dense INT4/INT8 |

`2:4` and `4:8` follow OBR's accuracy semantics. They are not representable by
INT4 sparse Tensor Cores. Use `2:4_pairwise` when real sparse acceleration is
required.

For `range=clip`, per-channel/per-tensor weight scales use OBR's MSE shrink
search (100-grid, `maxshrink=0.8`, `L2.4`). Group scales still use
`--clip-rhos`.

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
scripts/compress.sh --structure 2:4_pairwise --w-gran channel \
  --w-bits 4 --a-bits 4 --k-bits 4 --v-bits 4 --single
```

`scripts/compress.sh` and `scripts/bench.sh` are readable command templates.
Edit the flags directly or append overrides on the command line.  Use
`MODEL=llama2-7b` or `PYTHON=/path/to/python` to switch the model or interpreter
without editing the scripts.

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
qwen2.5-0.5b__0.0w16a16kv16__wanda_mt_rf_rtn_pc__none__h13e5adcc
qwen2.5-0.5b__0.5x2to4w16a16kv16__wanda_mt_rf_rtn_pc__none__hd0d8bd3a
qwen2.5-0.5b__0.0w4a16kv16__wanda_mr_rc_rtn_pc__obr
qwen2.5-0.5b__0.5x2to4w4a16kv16__wanda_mr_rc_rtn_pc__obr
```

Use `--single` to write only the joint checkpoint.

## Results

All benchmark tasks merge into one file:

```text
results/benchmarks.json
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
  --a-bits 16 --k-bits 16 --v-bits 16 \
  --quant gptq --comp none --no-smooth --rotation none \
  --checkpoint-dir checkpoints \
  --tasks e2e --engine inference --e2e-batch 8 \
  --e2e-seqlen 512 --e2e-decode-tokens 16 \
  --e2e-warmup 2 --e2e-iters 5 \
  --output results/benchmarks.json --replace
```

If structured sparse acceleration is unavailable, the benchmark does not abort:
`sparse`/`joint` e2e are skipped, but their PPL/zero-shot are still evaluated
with dense/int fallbacks, and `dense`/`quant` e2e keeps running.  Use
`--no-sparse-accel` to force this mode on a sparse-capable GPU.

The inference engine consumes the four QDrift checkpoints and labels them
`dense`, `sparse_fp16`, `int4_marlin`, `joint_sparse_int4`.  Use `--a-bits 4`
checkpoints for the W4A4/joint-sparse-W4A4 path, and `--e2e-kv-bits 8|4` to
enable fused quantized KV attention.  The default
`--e2e-regimes dense,quant,joint` keeps only the effective paths.  The engine
requires `--rotation none`; rotated checkpoints use `--engine hf`.  Reference
results: `results/inference/REPORT.md`.
