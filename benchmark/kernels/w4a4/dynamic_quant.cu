// Copyright (c) 2026 QDrift Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Fused per-token scale + INT4 packing for the QuaRot CUTLASS GEMM.
#include <common.h>

__global__ void sym_quant_dynamic_kernel(
    const half* __restrict__ x,
    uint32_t rows,
    uint32_t cols,
    float clip_ratio,
    half* __restrict__ scale_out,
    Int4Storage* __restrict__ q
) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    __shared__ float cache[256];
    __shared__ float shared_scale;

    float amax = 0.0f;
    for (uint32_t col = tid; col < cols; col += stride) {
        amax = fmaxf(amax, fabsf(__half2float(x[row * cols + col])));
    }
    cache[tid] = amax;
    __syncthreads();
    for (uint32_t step = stride / 2; step > 0; step >>= 1) {
        if (tid < step) {
            cache[tid] = fmaxf(cache[tid], cache[tid + step]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        float scale = fmaxf(cache[0] * clip_ratio / 7.0f, 1e-8f);
        shared_scale = scale;
        scale_out[row] = __float2half_rn(scale);
    }
    __syncthreads();
    const float inv_scale = 1.0f / shared_scale;
    const uint32_t packed_cols = (cols + 1) / 2;
    for (uint32_t col = tid; col < packed_cols; col += stride) {
        Int4Storage storage = 0;
        const uint32_t k0 = col * 2;
        const int c0 = clamp(__float2int_rn(__half2float(x[row * cols + k0]) * inv_scale), qmin, qmax);
        Int4Subbyte{reinterpret_cast<cutlass::int4b_t*>(&storage), 0}.set(c0);
        if (k0 + 1 < cols) {
            const int c1 = clamp(__float2int_rn(__half2float(x[row * cols + k0 + 1]) * inv_scale), qmin, qmax);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t*>(&storage), 1}.set(c1);
        }
        q[row * packed_cols + col] = storage;
    }
}

void sym_quant_dynamic_host(
    const half* x,
    uint32_t rows,
    uint32_t cols,
    float clip_ratio,
    half* scale,
    Int4Storage* q
) {
    sym_quant_dynamic_kernel<<<rows, 256>>>(x, rows, cols, clip_ratio, scale, q);
}
