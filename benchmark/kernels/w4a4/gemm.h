// SPDX-License-Identifier: Apache-2.0
// Adapted from: https://github.com/spcl/QuaRot (commit 5008669b08c1f11f9b64d52d16fddd47ca754c5a)
#pragma once

#include <common.h>


void matmul_host(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);