/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Persistent reduction kernels for the CUDA op component.
 *
 * Each kernel runs one block of 256 threads and loops indefinitely,
 * sleeping between polls to reduce power consumption.  The host posts
 * a command by writing src/dst/count into the managed-memory slot and
 * then setting status=1.  The kernel executes the reduction, then sets
 * status=2.  The host spins on status until it sees 2, then resets it
 * to 0 for the next call.  A separate shutdown flag terminates the loop
 * at session end.
 */

#include <stdint.h>
#include <cuda_runtime.h>

#include "ompi/mca/op/op.h"
#include "ompi/mca/op/cuda/op_cuda.h"

/* -------------------------------------------------------------------------
 * PERSISTENT_KERNEL(name, ctype, op_expr)
 *
 * Generates __global__ void ompi_op_cuda_persistent_<name>(...).
 * op_expr must be a statement that updates dst[i] in-place using src[i],
 * e.g. "dst[i] += src[i]" or "dst[i] = dst[i] > src[i] ? dst[i] : src[i]".
 * ------------------------------------------------------------------------- */
#define PERSISTENT_KERNEL(kname, ctype, op_expr)                               \
__global__ void ompi_op_cuda_persistent_##kname(                               \
        ompi_op_gpu_cmd_t *cmd, volatile int32_t *shutdown)                    \
{                                                                               \
    while (!*shutdown) {                                                        \
        /* Spin-wait for work; sleep 1 µs between polls to save power */        \
        while (cmd->status != 1 && !*shutdown) { __nanosleep(1000); }          \
        if (*shutdown) break;                                                   \
        const ctype * __restrict__ src = (const ctype *) cmd->src;             \
              ctype * __restrict__ dst = (      ctype *) cmd->dst;             \
        int64_t n = cmd->count;                                                 \
        for (int64_t i = (int64_t)threadIdx.x; i < n; i += blockDim.x) {      \
            op_expr;                                                            \
        }                                                                       \
        __syncthreads();                                                        \
        if (threadIdx.x == 0) {                                                 \
            __threadfence_system();   /* ensure dst writes reach host */        \
            cmd->status = 2;          /* signal done */                         \
        }                                                                       \
    }                                                                           \
}

/* =========================================================================
 * Kernel instantiations
 * ========================================================================= */

/* --- MAX --- */
PERSISTENT_KERNEL(max_int8,   int8_t,   dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_uint8,  uint8_t,  dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_int16,  int16_t,  dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_uint16, uint16_t, dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_int32,  int32_t,  dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_uint32, uint32_t, dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_int64,  int64_t,  dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_uint64, uint64_t, dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_float,  float,    dst[i] = dst[i] > src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(max_double, double,   dst[i] = dst[i] > src[i] ? dst[i] : src[i])

/* --- MIN --- */
PERSISTENT_KERNEL(min_int8,   int8_t,   dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_uint8,  uint8_t,  dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_int16,  int16_t,  dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_uint16, uint16_t, dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_int32,  int32_t,  dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_uint32, uint32_t, dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_int64,  int64_t,  dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_uint64, uint64_t, dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_float,  float,    dst[i] = dst[i] < src[i] ? dst[i] : src[i])
PERSISTENT_KERNEL(min_double, double,   dst[i] = dst[i] < src[i] ? dst[i] : src[i])

/* --- SUM --- */
PERSISTENT_KERNEL(sum_int8,   int8_t,   dst[i] += src[i])
PERSISTENT_KERNEL(sum_uint8,  uint8_t,  dst[i] += src[i])
PERSISTENT_KERNEL(sum_int16,  int16_t,  dst[i] += src[i])
PERSISTENT_KERNEL(sum_uint16, uint16_t, dst[i] += src[i])
PERSISTENT_KERNEL(sum_int32,  int32_t,  dst[i] += src[i])
PERSISTENT_KERNEL(sum_uint32, uint32_t, dst[i] += src[i])
PERSISTENT_KERNEL(sum_int64,  int64_t,  dst[i] += src[i])
PERSISTENT_KERNEL(sum_uint64, uint64_t, dst[i] += src[i])
PERSISTENT_KERNEL(sum_float,  float,    dst[i] += src[i])
PERSISTENT_KERNEL(sum_double, double,   dst[i] += src[i])

/* --- PROD --- */
PERSISTENT_KERNEL(prod_int8,   int8_t,   dst[i] *= src[i])
PERSISTENT_KERNEL(prod_uint8,  uint8_t,  dst[i] *= src[i])
PERSISTENT_KERNEL(prod_int16,  int16_t,  dst[i] *= src[i])
PERSISTENT_KERNEL(prod_uint16, uint16_t, dst[i] *= src[i])
PERSISTENT_KERNEL(prod_int32,  int32_t,  dst[i] *= src[i])
PERSISTENT_KERNEL(prod_uint32, uint32_t, dst[i] *= src[i])
PERSISTENT_KERNEL(prod_int64,  int64_t,  dst[i] *= src[i])
PERSISTENT_KERNEL(prod_uint64, uint64_t, dst[i] *= src[i])
PERSISTENT_KERNEL(prod_float,  float,    dst[i] *= src[i])
PERSISTENT_KERNEL(prod_double, double,   dst[i] *= src[i])

/* --- BAND (bitwise AND, integer types only) --- */
PERSISTENT_KERNEL(band_int8,   int8_t,   dst[i] &= src[i])
PERSISTENT_KERNEL(band_uint8,  uint8_t,  dst[i] &= src[i])
PERSISTENT_KERNEL(band_int16,  int16_t,  dst[i] &= src[i])
PERSISTENT_KERNEL(band_uint16, uint16_t, dst[i] &= src[i])
PERSISTENT_KERNEL(band_int32,  int32_t,  dst[i] &= src[i])
PERSISTENT_KERNEL(band_uint32, uint32_t, dst[i] &= src[i])
PERSISTENT_KERNEL(band_int64,  int64_t,  dst[i] &= src[i])
PERSISTENT_KERNEL(band_uint64, uint64_t, dst[i] &= src[i])

/* --- BOR (bitwise OR) --- */
PERSISTENT_KERNEL(bor_int8,   int8_t,   dst[i] |= src[i])
PERSISTENT_KERNEL(bor_uint8,  uint8_t,  dst[i] |= src[i])
PERSISTENT_KERNEL(bor_int16,  int16_t,  dst[i] |= src[i])
PERSISTENT_KERNEL(bor_uint16, uint16_t, dst[i] |= src[i])
PERSISTENT_KERNEL(bor_int32,  int32_t,  dst[i] |= src[i])
PERSISTENT_KERNEL(bor_uint32, uint32_t, dst[i] |= src[i])
PERSISTENT_KERNEL(bor_int64,  int64_t,  dst[i] |= src[i])
PERSISTENT_KERNEL(bor_uint64, uint64_t, dst[i] |= src[i])

/* --- BXOR (bitwise XOR) --- */
PERSISTENT_KERNEL(bxor_int8,   int8_t,   dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_uint8,  uint8_t,  dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_int16,  int16_t,  dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_uint16, uint16_t, dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_int32,  int32_t,  dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_uint32, uint32_t, dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_int64,  int64_t,  dst[i] ^= src[i])
PERSISTENT_KERNEL(bxor_uint64, uint64_t, dst[i] ^= src[i])

/* =========================================================================
 * Host-side launcher wrappers — one per kernel, 1 block × 256 threads.
 * ========================================================================= */
#define LAUNCHER(kname)                                                        \
static void launch_##kname(ompi_op_gpu_cmd_t *cmd,                            \
                            volatile int32_t  *sd,                             \
                            cudaStream_t       stream)                         \
{                                                                               \
    ompi_op_cuda_persistent_##kname<<<1, 256, 0, stream>>>(cmd, sd);          \
}

LAUNCHER(max_int8)    LAUNCHER(max_uint8)
LAUNCHER(max_int16)   LAUNCHER(max_uint16)
LAUNCHER(max_int32)   LAUNCHER(max_uint32)
LAUNCHER(max_int64)   LAUNCHER(max_uint64)
LAUNCHER(max_float)   LAUNCHER(max_double)

LAUNCHER(min_int8)    LAUNCHER(min_uint8)
LAUNCHER(min_int16)   LAUNCHER(min_uint16)
LAUNCHER(min_int32)   LAUNCHER(min_uint32)
LAUNCHER(min_int64)   LAUNCHER(min_uint64)
LAUNCHER(min_float)   LAUNCHER(min_double)

LAUNCHER(sum_int8)    LAUNCHER(sum_uint8)
LAUNCHER(sum_int16)   LAUNCHER(sum_uint16)
LAUNCHER(sum_int32)   LAUNCHER(sum_uint32)
LAUNCHER(sum_int64)   LAUNCHER(sum_uint64)
LAUNCHER(sum_float)   LAUNCHER(sum_double)

LAUNCHER(prod_int8)   LAUNCHER(prod_uint8)
LAUNCHER(prod_int16)  LAUNCHER(prod_uint16)
LAUNCHER(prod_int32)  LAUNCHER(prod_uint32)
LAUNCHER(prod_int64)  LAUNCHER(prod_uint64)
LAUNCHER(prod_float)  LAUNCHER(prod_double)

LAUNCHER(band_int8)   LAUNCHER(band_uint8)
LAUNCHER(band_int16)  LAUNCHER(band_uint16)
LAUNCHER(band_int32)  LAUNCHER(band_uint32)
LAUNCHER(band_int64)  LAUNCHER(band_uint64)

LAUNCHER(bor_int8)    LAUNCHER(bor_uint8)
LAUNCHER(bor_int16)   LAUNCHER(bor_uint16)
LAUNCHER(bor_int32)   LAUNCHER(bor_uint32)
LAUNCHER(bor_int64)   LAUNCHER(bor_uint64)

LAUNCHER(bxor_int8)   LAUNCHER(bxor_uint8)
LAUNCHER(bxor_int16)  LAUNCHER(bxor_uint16)
LAUNCHER(bxor_int32)  LAUNCHER(bxor_uint32)
LAUNCHER(bxor_int64)  LAUNCHER(bxor_uint64)

/* =========================================================================
 * 2D launcher table [op_index][type_index]
 *
 * Indexed by OMPI_OP_BASE_FORTRAN_* (rows) × OMPI_OP_BASE_TYPE_* (columns).
 * Zero/NULL entries mean "not supported on GPU" → host fallback.
 * ========================================================================= */
ompi_op_cuda_launcher_fn_t
ompi_op_cuda_kernel_fns[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] = {

    [OMPI_OP_BASE_FORTRAN_MAX] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_max_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_max_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_max_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_max_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_max_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_max_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_max_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_max_uint64,
        [OMPI_OP_BASE_TYPE_FLOAT]    = launch_max_float,
        [OMPI_OP_BASE_TYPE_DOUBLE]   = launch_max_double,
    },

    [OMPI_OP_BASE_FORTRAN_MIN] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_min_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_min_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_min_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_min_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_min_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_min_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_min_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_min_uint64,
        [OMPI_OP_BASE_TYPE_FLOAT]    = launch_min_float,
        [OMPI_OP_BASE_TYPE_DOUBLE]   = launch_min_double,
    },

    [OMPI_OP_BASE_FORTRAN_SUM] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_sum_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_sum_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_sum_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_sum_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_sum_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_sum_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_sum_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_sum_uint64,
        [OMPI_OP_BASE_TYPE_FLOAT]    = launch_sum_float,
        [OMPI_OP_BASE_TYPE_DOUBLE]   = launch_sum_double,
    },

    [OMPI_OP_BASE_FORTRAN_PROD] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_prod_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_prod_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_prod_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_prod_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_prod_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_prod_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_prod_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_prod_uint64,
        [OMPI_OP_BASE_TYPE_FLOAT]    = launch_prod_float,
        [OMPI_OP_BASE_TYPE_DOUBLE]   = launch_prod_double,
    },

    [OMPI_OP_BASE_FORTRAN_BAND] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_band_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_band_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_band_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_band_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_band_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_band_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_band_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_band_uint64,
    },

    [OMPI_OP_BASE_FORTRAN_BOR] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_bor_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_bor_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_bor_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_bor_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_bor_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_bor_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_bor_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_bor_uint64,
    },

    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        [OMPI_OP_BASE_TYPE_INT8_T]   = launch_bxor_int8,
        [OMPI_OP_BASE_TYPE_UINT8_T]  = launch_bxor_uint8,
        [OMPI_OP_BASE_TYPE_INT16_T]  = launch_bxor_int16,
        [OMPI_OP_BASE_TYPE_UINT16_T] = launch_bxor_uint16,
        [OMPI_OP_BASE_TYPE_INT32_T]  = launch_bxor_int32,
        [OMPI_OP_BASE_TYPE_UINT32_T] = launch_bxor_uint32,
        [OMPI_OP_BASE_TYPE_INT64_T]  = launch_bxor_int64,
        [OMPI_OP_BASE_TYPE_UINT64_T] = launch_bxor_uint64,
    },

    /* LAND, LOR, LXOR, MAXLOC, MINLOC, REPLACE, NO_OP: all NULL → host path */
};
