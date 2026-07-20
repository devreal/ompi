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

#ifndef OMPI_MCA_OP_ROCM_H
#define OMPI_MCA_OP_ROCM_H

#include "ompi_config.h"
#include <hip/hip_runtime.h>

#include "ompi/mca/op/op.h"
#include "ompi/op/op_gpu_session.h"  /* defines ompi_op_gpu_cmd_t */

BEGIN_C_DECLS

/**
 * ROCm-specific cmd_queue.  Inherits ompi_op_gpu_cmd_queue_t by placing it
 * as the first member named "super".  The HIP stream and shutdown flag are
 * stored directly here rather than in a separate priv allocation.
 * Allocated with OBJ_NEW; the OBJ destructor chain releases GPU resources.
 *
 * super.cmd and shutdown both live in plain device memory (hipMalloc) --
 * they're what the persistent kernel polls, so they need to be fast for the
 * GPU to access, not host-dereferenceable. host_cmd/host_shutdown are
 * registered (page-locked) host mirrors that the host writes/reads
 * directly; transfers between the host and device copies happen via
 * explicit hipMemcpyAsync on ctrl_stream. This avoids hipMallocManaged for
 * both slots: the cmd_queue pool (ompi_op_gpu_session.c) relaunches the
 * persistent kernel on every single collective call and kills it again at
 * session end, so shutdown sees the same host/device read-write ping-pong
 * on every call that cmd sees on every reduction -- managed memory would
 * fault on both.
 */
typedef struct ompi_op_rocm_cmd_queue_t {
    ompi_op_gpu_cmd_queue_t  super;       /* MUST be first; super.cmd is device memory */
    volatile int32_t        *shutdown;    /* device-resident shutdown flag, polled by the kernel */
    hipStream_t              stream;      /* private HIP stream running the persistent kernel */
    ompi_op_gpu_cmd_t       *host_cmd;    /* registered host mirror of super.cmd */
    int32_t                 *host_shutdown; /* registered host mirror of shutdown */
    hipStream_t              ctrl_stream; /* dedicated stream for host<->device cmd transfers */
} ompi_op_rocm_cmd_queue_t;
OBJ_CLASS_DECLARATION(ompi_op_rocm_cmd_queue_t);

/**
 * Host-side launcher function type.
 * Launches the persistent kernel for one (op, type) combination.
 */
typedef void (*ompi_op_rocm_launcher_fn_t)(ompi_op_gpu_cmd_t *cmd,
                                           volatile int32_t  *shutdown,
                                           hipStream_t        stream);

/**
 * 2D table [op_index][type_index] of launcher functions.
 * NULL entries indicate unsupported (op, type) combinations; the session
 * machinery returns NULL for those and the caller falls back to the host path.
 *
 * Indexed by OMPI_OP_BASE_FORTRAN_* × OMPI_OP_BASE_TYPE_*.
 * Defined (and initialized) in op_rocm_kernels.cpp.
 */
OMPI_DECLSPEC extern ompi_op_rocm_launcher_fn_t
ompi_op_rocm_kernel_fns[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];

/* Defined in op_rocm_kernels.cpp (extern "C") */
void ompi_op_rocm_kernel_fns_init(void);

/* Defined in op_rocm_session.c */
ompi_op_gpu_cmd_queue_t *ompi_op_rocm_cmd_queue_alloc(int dev_id);
ompi_op_gpu_session_t *ompi_op_rocm_session_begin(ompi_op_gpu_cmd_queue_t *queue,
                                                   struct ompi_op_t *op,
                                                   struct ompi_datatype_t *dtype);

END_C_DECLS

#endif /* OMPI_MCA_OP_ROCM_H */
