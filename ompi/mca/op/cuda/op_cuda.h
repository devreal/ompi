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

#ifndef OMPI_MCA_OP_CUDA_H
#define OMPI_MCA_OP_CUDA_H

#include "ompi_config.h"
#include <cuda_runtime.h>

#include "ompi/mca/op/op.h"
#include "ompi/op/op_gpu_session.h"  /* defines ompi_op_gpu_cmd_t */

BEGIN_C_DECLS

/**
 * CUDA-specific cmd_queue.  Inherits ompi_op_gpu_cmd_queue_t by placing it
 * as the first member named "super".  The CUDA stream and shutdown flag are
 * stored directly here rather than in a separate priv allocation.
 * Allocated with OBJ_NEW; the OBJ destructor chain releases GPU resources.
 */
typedef struct ompi_op_cuda_cmd_queue_t {
    ompi_op_gpu_cmd_queue_t  super;       /* MUST be first */
    volatile int32_t        *shutdown;    /* managed-memory shutdown flag */
    cudaStream_t             stream;      /* private CUDA stream */
} ompi_op_cuda_cmd_queue_t;
OBJ_CLASS_DECLARATION(ompi_op_cuda_cmd_queue_t);

/**
 * Host-side launcher function type.
 * Launches the persistent kernel for one (op, type) combination.
 */
typedef void (*ompi_op_cuda_launcher_fn_t)(ompi_op_gpu_cmd_t *cmd,
                                           volatile int32_t  *shutdown,
                                           cudaStream_t       stream);

/**
 * 2D table [op_index][type_index] of launcher functions.
 * NULL entries indicate unsupported (op, type) combinations; the session
 * machinery returns NULL for those and the caller falls back to the host path.
 *
 * Indexed by OMPI_OP_BASE_FORTRAN_* × OMPI_OP_BASE_TYPE_*.
 * Defined (and initialized) in op_cuda_kernels.cu.
 */
OMPI_DECLSPEC extern ompi_op_cuda_launcher_fn_t
ompi_op_cuda_kernel_fns[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];

/* Defined in op_cuda_kernels.cu (extern "C") */
void ompi_op_cuda_kernel_fns_init(void);

/* Defined in op_cuda_session.c */
ompi_op_gpu_cmd_queue_t *ompi_op_cuda_cmd_queue_alloc(int dev_id);
ompi_op_gpu_session_t *ompi_op_cuda_session_begin(ompi_op_gpu_cmd_queue_t *queue,
                                                   struct ompi_op_t *op,
                                                   struct ompi_datatype_t *dtype);

END_C_DECLS

#endif /* OMPI_MCA_OP_CUDA_H */
