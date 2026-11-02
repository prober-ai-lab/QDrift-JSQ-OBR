// SPDX-License-Identifier: Apache-2.0
// Adapted from: https://github.com/spcl/QuaRot (commit 5008669b08c1f11f9b64d52d16fddd47ca754c5a)
#pragma once

#include <common.h>


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
);


void sym_dequant_host(
        const int32_t *q,
        const half *scale_row,
        const half *scale_col,
        uint32_t rows,
        uint32_t cols,
        half *x
);