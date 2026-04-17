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
 * Private per-session state owned by the cuda component.
 * Stored in ompi_op_gpu_session_t.backend.
 */
typedef struct {
    ompi_op_gpu_cmd_t   *cmd;       /* managed-memory command slot        */
    volatile int32_t    *shutdown;  /* managed-memory shutdown flag        */
    cudaStream_t         stream;    /* private CUDA stream for this session */
} ompi_op_cuda_session_priv_t;

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
 * Declared in op_cuda_kernels.cu; populated by ompi_op_cuda_kernel_fns_init().
 */
OMPI_DECLSPEC extern ompi_op_cuda_launcher_fn_t
ompi_op_cuda_kernel_fns[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];

/** Populate ompi_op_cuda_kernel_fns. Called once from cuda_component_open(). */
OMPI_DECLSPEC void ompi_op_cuda_kernel_fns_init(void);

END_C_DECLS

#endif /* OMPI_MCA_OP_CUDA_H */
