// Copyright (c) 2026 QDrift Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 4:8 pairwise sparse INT4 CUTLASS GEMM used by the joint W4A4 benchmark regime.
//
// The sparse operand is the weight matrix W [out_features, K].  CUTLASS
// sparse Tensor Cores only support sparsity on operand A, so the kernel
// computes
//
//     D[out_features, rows] = W_sparse[out_features, K] * x^T[K, rows]
//
// with a row-major output.  The fused dequantization kernel transposes D
// while scaling so the caller receives the usual ``[rows, out_features]``
// layout.
//
// Metadata follows the PTX ``mma.sp::ordered_metadata`` INT4 rule: one
// nibble per eight original columns, holding two increasing 2-element pair
// indices (4-bit values 0x4/0x8/0x9/0xc/0xd/0xe).  Nibbles are packed into
// uint32 words (eight groups per word) and reordered with CUTLASS
// ``reorder_meta`` for the ldmatrix metadata layout.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_uncompress.h>

using SparseGemm = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

namespace {

using ElementE = typename SparseGemm::GemmKernel::ElementE;
using LayoutE = typename SparseGemm::GemmKernel::LayoutE;

void check(cutlass::Status status, const char* what) {
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cutlassGetStatusString(status));
    }
}

}  // namespace

void sparse_matmul_host(
    const uint8_t* A,
    const uint32_t* E,
    const uint8_t* B,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int32_t* C)
{
    // LayoutE is ColumnMajorInterleaved<2>: the leading dimension is
    // 2 * M, matching the reordered HostTensor stride used by CUTLASS
    // example 15.
    cutlass::TensorRef<ElementE, LayoutE> e_ref(
        reinterpret_cast<ElementE*>(const_cast<uint32_t*>(E)), M * 2);

    typename SparseGemm::Arguments arguments{
        {M, N, K},
        {reinterpret_cast<const cutlass::int4b_t*>(A), lda},
        {reinterpret_cast<const cutlass::int4b_t*>(B), ldb},
        {C, ldc},
        {C, ldc},
        e_ref,
        {1, 0},
        1,
    };

    SparseGemm gemm_op;
    check(gemm_op.can_implement(arguments), "sparse GEMM unsupported shape");
    size_t workspace_size = SparseGemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    check(gemm_op.initialize(arguments, workspace.get()), "sparse GEMM init");
    check(gemm_op(), "sparse GEMM run");
}

// Reorder row-major [rows, K/64] metadata into the ldmatrix layout.  The
// destination is column-major, matching SparseGemm::LayoutE.
void reorder_meta_host(const uint32_t* src, uint32_t* dst, int rows, int groups) {
    cutlass::HostTensor<uint32_t, cutlass::layout::RowMajor> source(
        {rows, groups}, false);
    cutlass::HostTensor<uint32_t, LayoutE> dest({rows, groups}, false);
    std::memcpy(source.host_data(), src, sizeof(uint32_t) * size_t(rows) * groups);
    cutlass::reorder_meta(
        dest.host_ref(),
        source.host_ref(),
        {rows, 1, groups});
    std::memcpy(dst, dest.host_data(), sizeof(uint32_t) * size_t(rows) * groups);
}

constexpr int kDequantTile = 32;

// D is [out_features, rows] row-major.  The consumer wants
// out [rows, out_features] row-major, so transpose and scale in one pass.
// A shared-memory tile keeps both the INT32 reads and the FP16 writes
// coalesced; the naive elementwise transpose is memory-bound.
__global__ void dequant_sparse_kernel(
    const int32_t* __restrict__ acc,
    const half* __restrict__ a_scale,
    const half* __restrict__ w_scale,
    half* __restrict__ out,
    int rows,
    int out_features)
{
    __shared__ int32_t tile[kDequantTile][kDequantTile + 1];
    const int out0 = blockIdx.x * kDequantTile;
    const int row0 = blockIdx.y * kDequantTile;

    for (int i = threadIdx.y; i < kDequantTile; i += blockDim.y) {
        const int o = out0 + i;
        for (int j = threadIdx.x; j < kDequantTile; j += blockDim.x) {
            const int r = row0 + j;
            tile[i][j] =
                (o < out_features && r < rows) ? acc[size_t(o) * rows + r] : 0;
        }
    }
    __syncthreads();

    for (int i = threadIdx.y; i < kDequantTile; i += blockDim.y) {
        const int r = row0 + i;
        for (int j = threadIdx.x; j < kDequantTile; j += blockDim.x) {
            const int o = out0 + j;
            if (r < rows && o < out_features) {
                float value = static_cast<float>(tile[j][i]) *
                              __half2float(a_scale[r]) *
                              __half2float(w_scale[o]);
                out[size_t(r) * out_features + o] = __float2half_rn(value);
            }
        }
    }
}

void sparse_uncompress_host(
    const uint8_t* A,
    const uint32_t* E,
    int rows,
    int K,
    int8_t* out)
{
    cutlass::HostTensor<cutlass::int4b_t, cutlass::layout::RowMajor> a(
        {rows, K / 2}, false);
    cutlass::HostTensor<cutlass::int4b_t, cutlass::layout::RowMajor> dense(
        {rows, K}, false);
    cutlass::HostTensor<uint32_t, cutlass::layout::RowMajor> e(
        {rows, K / 64}, false);
    std::memcpy(a.host_data(), A, size_t(rows) * (K / 4));
    std::memcpy(e.host_data(), E, size_t(rows) * (K / 64) * sizeof(uint32_t));
    cutlass::uncompress(dense.host_ref(), a.host_ref(), e.host_ref(), rows, K);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < K; ++c) {
            out[r * K + c] = static_cast<int8_t>(int(dense.at({r, c})));
        }
    }
}

void sparse_dequant_host(
    const int32_t* acc,
    const half* a_scale,
    const half* w_scale,
    half* out,
    int rows,
    int out_features)
{
    dim3 threads(32, 8);
    dim3 blocks(
        (out_features + kDequantTile - 1) / kDequantTile,
        (rows + kDequantTile - 1) / kDequantTile);
    dequant_sparse_kernel<<<blocks, threads>>>(
        acc, a_scale, w_scale, out, rows, out_features);
}
