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

#include "ompi_config.h"

#include <cuda_runtime.h>

#include "ompi/constants.h"
#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/mca/op/cuda/op_cuda.h"

/* Forward declarations of session hooks (implemented in op_cuda_session.c) */
ompi_op_gpu_session_t *ompi_op_cuda_session_begin(struct ompi_op_t *op,
                                                   struct ompi_datatype_t *dtype,
                                                   int dev_id);
void ompi_op_cuda_session_reduce(ompi_op_gpu_session_t *session,
                                 const void *src, void *dst, size_t count);
void ompi_op_cuda_session_stop(ompi_op_gpu_session_t *session);
bool ompi_op_cuda_session_restart(ompi_op_gpu_session_t *session,
                                   struct ompi_op_t *op,
                                   struct ompi_datatype_t *dtype);
void ompi_op_cuda_session_free(ompi_op_gpu_session_t *session);

static int cuda_component_open(void);
static int cuda_component_close(void);
static int cuda_component_init_query(bool enable_progress_threads,
                                     bool enable_mpi_thread_multiple);
static struct ompi_op_base_module_1_0_0_t *
    cuda_component_op_query(struct ompi_op_t *op, int *priority);

/*
 * Public component descriptor.
 *
 * This component does not provide per-op/per-type function pointers
 * (opc_op_query returns NULL).  Its sole contribution is the three session
 * hooks that enable persistent GPU reduction kernels.
 */
ompi_op_base_component_1_0_0_t mca_op_cuda_component = {
    .opc_version = {
        OMPI_OP_BASE_VERSION_1_0_0,

        .mca_component_name = "cuda",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),
        .mca_open_component  = cuda_component_open,
        .mca_close_component = cuda_component_close,
    },
    .opc_data = {
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },

    .opc_init_query = cuda_component_init_query,
    .opc_op_query   = cuda_component_op_query,

    /* GPU session hooks */
    .opc_session_begin   = ompi_op_cuda_session_begin,
    .opc_session_reduce  = ompi_op_cuda_session_reduce,
    .opc_session_stop    = ompi_op_cuda_session_stop,
    .opc_session_restart = ompi_op_cuda_session_restart,
    .opc_session_free    = ompi_op_cuda_session_free,
};
MCA_BASE_COMPONENT_INIT(ompi, op, cuda)

static int
cuda_component_open(void)
{
    return OMPI_SUCCESS;
}

static int
cuda_component_close(void)
{
    return OMPI_SUCCESS;
}

/*
 * Only activate this component when at least one CUDA-capable device is
 * present in the system.
 */
static int
cuda_component_init_query(bool enable_progress_threads,
                          bool enable_mpi_thread_multiple)
{
    int device_count = 0;
    cudaError_t err  = cudaGetDeviceCount(&device_count);
    if (cudaSuccess != err || device_count <= 0) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    return OMPI_SUCCESS;
}

/*
 * We do not provide per-op function pointers, only session hooks, so
 * always return NULL here.
 */
static struct ompi_op_base_module_1_0_0_t *
cuda_component_op_query(struct ompi_op_t *op, int *priority)
{
    (void) op;
    (void) priority;
    return NULL;
}
