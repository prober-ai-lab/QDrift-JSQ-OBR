// SPDX-License-Identifier: Apache-2.0
// Adapted from: https://github.com/spcl/QuaRot (commit 5008669b08c1f11f9b64d52d16fddd47ca754c5a)
#include <cutlass/gemm/device/gemm.h>
#include <gemm.h>
#include <c10/cuda/CUDAStream.h>




void matmul_host(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
)
{
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::int4b_t,                // ElementA
            cutlass::layout::RowMajor,       // LayoutA
            cutlass::int4b_t,                // ElementB
            cutlass::layout::ColumnMajor,    // LayoutB
            int32_t,                         // ElementOutput
            cutlass::layout::RowMajor,       // LayoutOutput
            int32_t,                         // ElementAccumulator
            cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
            cutlass::arch::Sm80  // tag indicating target GPU compute architecture  // TODO: This is just for compiling on my laptop temporarily. Should be higher when doing benchmarking.
    >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
            {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
            {(cutlass::int4b_t *) A,           K},
            {(cutlass::int4b_t *) B,           K},
            {C,                                N},
            {C,                                N},
            {1,                                0}
    };

    auto status = gemmOp(arguments, nullptr, c10::cuda::getCurrentCUDAStream());

    ensure(status == cutlass::Status::kSuccess,
           cutlassGetStatusString(status));

}
