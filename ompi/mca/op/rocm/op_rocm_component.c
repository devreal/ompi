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

#include <hip/hip_runtime.h>

#include "ompi/constants.h"
#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/mca/op/rocm/op_rocm.h"

/* Forward declarations of session hooks (implemented in op_rocm_session.c) */
ompi_op_gpu_session_t *ompi_op_rocm_session_begin(struct ompi_op_t *op,
                                                   struct ompi_datatype_t *dtype,
                                                   int dev_id);
void ompi_op_rocm_session_reduce(ompi_op_gpu_session_t *session,
                                  const void *src, void *dst, size_t count);
void ompi_op_rocm_session_stop(ompi_op_gpu_session_t *session);
bool ompi_op_rocm_session_restart(ompi_op_gpu_session_t *session,
                                   struct ompi_op_t *op,
                                   struct ompi_datatype_t *dtype);
void ompi_op_rocm_session_free(ompi_op_gpu_session_t *session);

static int rocm_component_open(void);
static int rocm_component_close(void);
static int rocm_component_init_query(bool enable_progress_threads,
                                      bool enable_mpi_thread_multiple);
static struct ompi_op_base_module_1_0_0_t *
    rocm_component_op_query(struct ompi_op_t *op, int *priority);

/*
 * Public component descriptor.
 */
ompi_op_base_component_1_0_0_t mca_op_rocm_component = {
    .opc_version = {
        OMPI_OP_BASE_VERSION_1_0_0,

        .mca_component_name = "rocm",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),
        .mca_open_component  = rocm_component_open,
        .mca_close_component = rocm_component_close,
    },
    .opc_data = {
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },

    .opc_init_query = rocm_component_init_query,
    .opc_op_query   = rocm_component_op_query,

    /* GPU session hooks */
    .opc_session_begin   = ompi_op_rocm_session_begin,
    .opc_session_reduce  = ompi_op_rocm_session_reduce,
    .opc_session_stop    = ompi_op_rocm_session_stop,
    .opc_session_restart = ompi_op_rocm_session_restart,
    .opc_session_free    = ompi_op_rocm_session_free,
};
MCA_BASE_COMPONENT_INIT(ompi, op, rocm)

static int
rocm_component_open(void)
{
    return OMPI_SUCCESS;
}

static int
rocm_component_close(void)
{
    return OMPI_SUCCESS;
}

/*
 * Only activate this component when at least one ROCm-capable device is
 * present in the system.
 */
static int
rocm_component_init_query(bool enable_progress_threads,
                           bool enable_mpi_thread_multiple)
{
    int device_count = 0;
    hipError_t err   = hipGetDeviceCount(&device_count);
    if (hipSuccess != err || device_count <= 0) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    return OMPI_SUCCESS;
}

/*
 * We do not provide per-op function pointers, only session hooks, so
 * always return NULL here.
 */
static struct ompi_op_base_module_1_0_0_t *
rocm_component_op_query(struct ompi_op_t *op, int *priority)
{
    (void) op;
    (void) priority;
    return NULL;
}
