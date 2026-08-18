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


static int cuda_component_open(void);
static int cuda_component_close(void);
static int cuda_component_init_query(bool enable_progress_threads,
                                     bool enable_mpi_thread_multiple);
static struct ompi_op_base_module_1_0_0_t *
    cuda_component_op_query(struct ompi_op_t *op, int *priority);
static int cuda_component_query_bandwidth(int dev_id, double *device_bw_bytes_per_sec,
                                          double *link_bw_bytes_per_sec);

/*
 * Public component descriptor.
 *
 * This component does not provide per-op/per-type function pointers
 * (opc_op_query returns NULL).  Its sole contribution is the three GPU
 * hooks that enable persistent-kernel GPU reduction sessions.
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
    .opc_cmd_queue_alloc = ompi_op_cuda_cmd_queue_alloc,
    .opc_session_begin   = ompi_op_cuda_session_begin,
    .opc_query_bandwidth = cuda_component_query_bandwidth,
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
    // register launchers here, component_open seems to be never called
    ompi_op_cuda_kernel_fns_init();
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

/*
 * Report dev_id's device memory bandwidth and its PCIe link bandwidth, used
 * by coll/tuned to analytically estimate a device-vs-host reduction
 * crossover size. Device bandwidth is derived from the memory clock/bus
 * width reported by the driver (GDDR/HBM is double-data-rate: 2 transfers
 * per clock); link bandwidth comes from the device's current PCIe link
 * generation/width via sysfs. NVLink is not detected -- if the device sits
 * behind NVLink rather than plain PCIe, this underestimates link bandwidth.
 *
 * Queried via cudaDeviceGetAttribute rather than cudaGetDeviceProperties:
 * cudaDeviceProp's memoryClockRate/memoryBusWidth fields were removed in
 * CUDA 12 (they don't map cleanly onto Hopper+'s memory subsystem, so NVIDIA
 * dropped them from the struct rather than leave them meaningless) -- the
 * equivalent cudaDeviceAttr enum values are stable across CUDA versions and
 * already the query style used elsewhere in this component (see
 * ompi_op_cuda_compute_coop_grid_size in op_cuda_kernels.cu). If a newer
 * architecture also renders these particular attributes meaningless, a
 * device_bw_bytes_per_sec <= 0 is treated by the caller (coll_tuned_gpu.c)
 * as "estimate unavailable" and falls back to the compiled-in default
 * threshold, not a crash.
 */
static int
cuda_component_query_bandwidth(int dev_id, double *device_bw_bytes_per_sec,
                               double *link_bw_bytes_per_sec)
{
    int memory_clock_khz = 0, bus_width_bits = 0;
    int pci_domain = 0, pci_bus = 0, pci_device = 0;

    cudaError_t err = cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate,
                                             dev_id);
    if (cudaSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    err = cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev_id);
    if (cudaSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    *device_bw_bytes_per_sec = 2.0 * ((double) memory_clock_khz * 1000.0)
                               * ((double) bus_width_bits / 8.0);

    err = cudaDeviceGetAttribute(&pci_domain, cudaDevAttrPciDomainId, dev_id);
    if (cudaSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    err = cudaDeviceGetAttribute(&pci_bus, cudaDevAttrPciBusId, dev_id);
    if (cudaSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    err = cudaDeviceGetAttribute(&pci_device, cudaDevAttrPciDeviceId, dev_id);
    if (cudaSuccess != err) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /* Function 0 is the GPU's primary (compute/display) function; this
     * assumption holds for the conventional single-GPU-per-slot case. */
    return ompi_op_gpu_query_pcie_link_bandwidth(pci_domain, pci_bus, pci_device, 0,
                                                 link_bw_bytes_per_sec);
}
