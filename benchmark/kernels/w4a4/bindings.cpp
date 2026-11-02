// Copyright (c) 2026 QDrift Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from: https://github.com/spcl/QuaRot (commit 5008669b08c1f11f9b64d52d16fddd47ca754c5a)
// Minimal bindings for the QuaRot INT4 CUTLASS GEMM and symmetric quantization.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <gemm.h>
#include <quant.h>

void sparse_matmul_host(const uint8_t* A, const uint32_t* E, const uint8_t* B,
                        int M, int N, int K, int lda, int ldb, int ldc,
                        int32_t* C);
void reorder_meta_host(const uint32_t* src, uint32_t* dst, int rows, int groups);
void sparse_dequant_host(const int32_t* acc, const half* a_scale,
                         const half* w_scale, half* out, int rows,
                         int out_features);
void sparse_uncompress_host(const uint8_t* A, const uint32_t* E, int rows,
                            int K, int8_t* out);

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kUInt8, "A must be packed uint8");
    TORCH_CHECK(B.scalar_type() == torch::kUInt8, "B must be packed uint8");
    const uint32_t M = A.size(0);
    const uint32_t N = B.size(0);
    const uint32_t K = A.size(1) * kElementsPerVector;
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));
    matmul_host(
        A.data_ptr<Int4Storage>(),
        B.data_ptr<Int4Storage>(),
        M,
        N,
        K,
        C.data_ptr<int32_t>()
    );
    return C;
}

void sym_quant_dynamic_host(const half* x, uint32_t rows, uint32_t cols, float clip_ratio, half* scale, Int4Storage* q);

torch::Tensor quarot_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor weight_scale, double clip_ratio) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weight_scale.is_cuda() && weight_scale.is_contiguous() && weight_scale.scalar_type() == torch::kFloat16);
    const uint32_t rows = x.size(0);
    const uint32_t cols = x.size(1);
    const uint32_t out_features = weight.size(0);
    const uint32_t packed_cols = (cols + 1) / 2;
    auto scale = torch::empty({rows}, torch::dtype(torch::kHalf).device(x.device()));
    auto q = torch::empty({rows, packed_cols}, torch::dtype(torch::kUInt8).device(x.device()));
    auto acc = torch::empty({rows, out_features}, torch::dtype(torch::kInt32).device(x.device()));
    auto out = torch::empty({rows, out_features}, torch::dtype(torch::kHalf).device(x.device()));
    sym_quant_dynamic_host(
        reinterpret_cast<half*>(x.data_ptr()), rows, cols,
        static_cast<float>(clip_ratio),
        reinterpret_cast<half*>(scale.data_ptr()),
        q.data_ptr<Int4Storage>()
    );
    matmul_host(
        q.data_ptr<Int4Storage>(), weight.data_ptr<Int4Storage>(),
        rows, out_features, cols,
        acc.data_ptr<int32_t>()
    );
    sym_dequant_host(
        acc.data_ptr<int32_t>(),
        reinterpret_cast<half*>(scale.data_ptr()),
        reinterpret_cast<half*>(weight_scale.data_ptr()),
        rows, out_features,
        reinterpret_cast<half*>(out.data_ptr())
    );
    return out;
}

std::vector<torch::Tensor> sym_quant_dynamic(torch::Tensor x, double clip_ratio) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16);
    const uint32_t rows = x.size(0);
    const uint32_t cols = x.size(1);
    const uint32_t packed_cols = (cols + 1) / 2;
    auto scale = torch::empty({rows}, torch::dtype(torch::kHalf).device(x.device()));
    auto q = torch::empty({rows, packed_cols}, torch::dtype(torch::kUInt8).device(x.device()));
    sym_quant_dynamic_host(
        reinterpret_cast<half*>(x.data_ptr()),
        rows,
        cols,
        static_cast<float>(clip_ratio),
        reinterpret_cast<half*>(scale.data_ptr()),
        q.data_ptr<Int4Storage>()
    );
    return {q, scale};
}

torch::Tensor sym_quant(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda() && scale.is_cuda(), "x and scale must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && scale.is_contiguous(), "x and scale must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
    TORCH_CHECK(scale.numel() == x.size(0), "scale must have one value per row");
    const uint32_t rows = x.size(0);
    const uint32_t cols_src = x.size(1);
    const uint32_t cols_dst = cdiv(cols_src, kElementsPerVector);
    auto q = torch::empty(
        {rows, cols_dst},
        torch::dtype(torch::kUInt8).device(x.device())
    );
    sym_quant_host(
        reinterpret_cast<half*>(x.data_ptr()),
        reinterpret_cast<half*>(scale.data_ptr()),
        rows,
        cols_src,
        cols_dst,
        q.data_ptr<Int4Storage>()
    );
    return q;
}

torch::Tensor sym_dequant(
    torch::Tensor q,
    torch::Tensor scale_row,
    torch::Tensor scale_col,
    int64_t bits
) {
    TORCH_CHECK(bits == 32, "only INT32 accumulation is supported");
    TORCH_CHECK(q.is_cuda() && scale_row.is_cuda() && scale_col.is_cuda());
    TORCH_CHECK(q.is_contiguous() && scale_row.is_contiguous() && scale_col.is_contiguous());
    const uint32_t rows = q.size(0);
    const uint32_t cols = q.size(1);
    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));
    sym_dequant_host(
        q.data_ptr<int32_t>(),
        reinterpret_cast<half*>(scale_row.data_ptr()),
        reinterpret_cast<half*>(scale_col.data_ptr()),
        rows,
        cols,
        reinterpret_cast<half*>(x.data_ptr())
    );
    return x;
}

torch::Tensor sparse_linear(torch::Tensor x, torch::Tensor weight,
                            torch::Tensor meta, torch::Tensor weight_scale,
                            double clip_ratio) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kUInt8);
    TORCH_CHECK(meta.is_cuda() && meta.is_contiguous() && meta.scalar_type() == torch::kInt32);
    TORCH_CHECK(weight_scale.is_cuda() && weight_scale.is_contiguous() && weight_scale.scalar_type() == torch::kHalf);
    const int rows = int(x.size(0));
    const int in_features = int(x.size(1));
    const int out_features = int(weight.size(0));
    const int packed_cols = (in_features + 1) / 2;
    auto scale = torch::empty({rows}, torch::dtype(torch::kHalf).device(x.device()));
    auto q = torch::empty({rows, packed_cols}, torch::dtype(torch::kUInt8).device(x.device()));
    auto acc = torch::empty({int64_t(out_features) * rows}, torch::dtype(torch::kInt32).device(x.device()));
    auto out = torch::empty({rows, out_features}, torch::dtype(torch::kHalf).device(x.device()));
    sym_quant_dynamic_host(
        reinterpret_cast<half*>(x.data_ptr()), rows, in_features,
        static_cast<float>(clip_ratio),
        reinterpret_cast<half*>(scale.data_ptr()),
        q.data_ptr<Int4Storage>());
    sparse_matmul_host(
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(meta.data_ptr<int32_t>()),
        q.data_ptr<uint8_t>(),
        out_features, rows, in_features,
        in_features / 2, in_features, rows,
        acc.data_ptr<int32_t>());
    sparse_dequant_host(
        acc.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(scale.data_ptr()),
        reinterpret_cast<const half*>(weight_scale.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        rows, out_features);
    return out;
}

torch::Tensor sparse_matmul(torch::Tensor A, torch::Tensor E, torch::Tensor B,
                            int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda() && E.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.is_contiguous() && E.is_contiguous() && B.is_contiguous());
    TORCH_CHECK(A.scalar_type() == torch::kUInt8, "A must be packed uint8");
    TORCH_CHECK(B.scalar_type() == torch::kUInt8, "B must be packed uint8");
    TORCH_CHECK(E.scalar_type() == torch::kInt32, "E must be uint32 words");
    auto C = torch::empty({M * N}, torch::dtype(torch::kInt32).device(A.device()));
    sparse_matmul_host(
        A.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(E.data_ptr<int32_t>()),
        B.data_ptr<uint8_t>(),
        int(M), int(N), int(K),
        int(K / 2), int(K), int(N),
        C.data_ptr<int32_t>());
    return C;
}

torch::Tensor sparse_reorder_meta(torch::Tensor src) {
    TORCH_CHECK(src.is_cpu() && src.is_contiguous());
    TORCH_CHECK(src.scalar_type() == torch::kInt32, "metadata must be int32 words");
    auto dst = torch::empty_like(src);
    reorder_meta_host(
        reinterpret_cast<const uint32_t*>(src.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(dst.data_ptr<int32_t>()),
        int(src.size(0)),
        int(src.size(1)));
    return dst;
}

torch::Tensor sparse_uncompress(torch::Tensor A, torch::Tensor E, int64_t rows, int64_t K) {
    TORCH_CHECK(A.is_cpu() && E.is_cpu() && A.is_contiguous() && E.is_contiguous());
    TORCH_CHECK(A.scalar_type() == torch::kUInt8);
    TORCH_CHECK(E.scalar_type() == torch::kInt32);
    auto out = torch::empty({rows, K}, torch::dtype(torch::kInt8));
    sparse_uncompress_host(
        A.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(E.data_ptr<int32_t>()),
        int(rows), int(K), out.data_ptr<int8_t>());
    return out;
}

torch::Tensor sparse_dequant(torch::Tensor acc, torch::Tensor a_scale,
                             torch::Tensor w_scale, int64_t rows,
                             int64_t out_features) {
    TORCH_CHECK(acc.is_cuda() && a_scale.is_cuda() && w_scale.is_cuda());
    TORCH_CHECK(acc.is_contiguous() && a_scale.is_contiguous() && w_scale.is_contiguous());
    TORCH_CHECK(acc.scalar_type() == torch::kInt32);
    TORCH_CHECK(a_scale.scalar_type() == torch::kHalf);
    TORCH_CHECK(w_scale.scalar_type() == torch::kHalf);
    auto out = torch::empty({rows, out_features}, torch::dtype(torch::kHalf).device(acc.device()));
    sparse_dequant_host(
        acc.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(a_scale.data_ptr()),
        reinterpret_cast<const half*>(w_scale.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        int(rows), int(out_features));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul);
    m.def("sym_quant", &sym_quant);
    m.def("sym_quant_dynamic", &sym_quant_dynamic);
    m.def("quarot_linear", &quarot_linear);
    m.def("sym_dequant", &sym_dequant);
    m.def("sparse_linear", &sparse_linear);
    m.def("sparse_matmul", &sparse_matmul);
    m.def("sparse_reorder_meta", &sparse_reorder_meta);
    m.def("sparse_dequant", &sparse_dequant);
    m.def("sparse_uncompress", &sparse_uncompress);
}
