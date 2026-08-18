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


static int rocm_component_open(void);
static int rocm_component_close(void);
static int rocm_component_init_query(bool enable_progress_threads,
                                      bool enable_mpi_thread_multiple);
static struct ompi_op_base_module_1_0_0_t *
    rocm_component_op_query(struct ompi_op_t *op, int *priority);
static int rocm_component_query_bandwidth(int dev_id, double *device_bw_bytes_per_sec,
                                          double *link_bw_bytes_per_sec);

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
    .opc_cmd_queue_alloc = ompi_op_rocm_cmd_queue_alloc,
    .opc_session_begin   = ompi_op_rocm_session_begin,
    .opc_query_bandwidth = rocm_component_query_bandwidth,
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
    // called here because component_open() is never called
    ompi_op_rocm_kernel_fns_init();
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

/*
 * Report dev_id's device memory bandwidth and its PCIe link bandwidth, used
 * by coll/tuned to analytically estimate a device-vs-host reduction
 * crossover size. Device bandwidth is derived from the memory clock/bus
 * width reported by the driver (GDDR/HBM is double-data-rate: 2 transfers
 * per clock); link bandwidth comes from the device's current PCIe link
 * generation/width via sysfs. Infinity Fabric is not detected -- if the
 * device sits behind Infinity Fabric rather than plain PCIe, this
 * underestimates link bandwidth.
 */
static int
rocm_component_query_bandwidth(int dev_id, double *device_bw_bytes_per_sec,
                               double *link_bw_bytes_per_sec)
{
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev_id);
    if (hipSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    *device_bw_bytes_per_sec = 2.0 * ((double) prop.memoryClockRate * 1000.0)
                               * ((double) prop.memoryBusWidth / 8.0);

    /* Function 0 is the GPU's primary (compute/display) function; this
     * assumption holds for the conventional single-GPU-per-slot case. */
    return ompi_op_gpu_query_pcie_link_bandwidth(prop.pciDomainID, prop.pciBusID,
                                                 prop.pciDeviceID, 0,
                                                 link_bw_bytes_per_sec);
}
