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
 * Dispatcher and cmd_queue pool for GPU reduction sessions.
 *
 * The expensive GPU resources — managed-memory command slot, shutdown flag,
 * and private GPU stream — are bundled into an ompi_op_gpu_cmd_queue_t and
 * pooled by dev_id.  Sessions themselves are lightweight structs (function
 * pointers + a pointer to the cmd_queue) and are allocated fresh for each
 * collective.
 *
 * Pool implementation:
 *   cmd_queue_pool  — opal_lifo_t providing lock-free thread-safe push/pop.
 *   cmd_queue_pool_count — atomic counter tracking current pool depth;
 *                          used to enforce CMD_QUEUE_POOL_MAX without a mutex.
 *
 * Pool lifecycle:
 *   session_end()   — stops the persistent kernel (cmd_queue resources remain
 *                     allocated), then pushes the cmd_queue into the lifo pool.
 *   session_begin() — pops from the lifo looking for a matching dev_id entry
 *                     and calls queue->session_begin_fn(queue, op, dtype) to
 *                     configure and relaunch the kernel; no cudaMalloc overhead.
 *                     On pool miss, iterates op components to allocate a fresh
 *                     cmd_queue, then calls session_begin.
 *                     On pool hit with no matching dev_id, the queue is pushed
 *                     back and a fresh allocation is attempted.
 *                     On pool hit with no kernel for (op, dtype), the queue is
 *                     returned to the pool and NULL is returned.
 *
 * CMD_QUEUE_POOL_MAX caps the total number of idle cmd_queues to bound GPU
 * resource accumulation.
 */

#include "ompi_config.h"

#include <stdlib.h>
#include <stdio.h>

#include "opal/class/opal_lifo.h"
#include "opal/class/opal_list.h"
#include "opal/mca/accelerator/base/base.h"
#include "opal/mca/base/base.h"
#include "opal/sys/atomic.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/op/op.h"

OBJ_CLASS_INSTANCE(ompi_op_gpu_cmd_queue_t, opal_list_item_t, NULL, NULL);

/* Maximum number of idle cmd_queues kept in the pool. */
#define CMD_QUEUE_POOL_MAX 16

static opal_lifo_t          cmd_queue_pool;
static opal_atomic_int32_t  cmd_queue_pool_count = 0;

/* --------------------------------------------------------------------------
 * cmd_queue_destroy — permanently release a cmd_queue's GPU resources.
 * OBJ_RELEASE dispatches through the concrete class destructor chain
 * (e.g. ompi_op_cuda_cmd_queue_t) and frees the allocation.
 * -------------------------------------------------------------------------- */
static void
cmd_queue_destroy(ompi_op_gpu_cmd_queue_t *queue)
{
    OBJ_RELEASE(queue);
}

/* --------------------------------------------------------------------------
 * cmd_queue_pool_push — return a cmd_queue to the pool.
 * Destroys the queue instead if the pool is already at capacity.
 * -------------------------------------------------------------------------- */
static void
cmd_queue_pool_push(ompi_op_gpu_cmd_queue_t *queue)
{
    if (opal_atomic_add_fetch_32(&cmd_queue_pool_count, 1) <= CMD_QUEUE_POOL_MAX) {
        opal_lifo_push(&cmd_queue_pool, &queue->super);
    } else {
        opal_atomic_add_fetch_32(&cmd_queue_pool_count, -1);
        cmd_queue_destroy(queue);
    }
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_pool_init
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_pool_init(void)
{
    OBJ_CONSTRUCT(&cmd_queue_pool, opal_lifo_t);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_begin
 *
 * 1. Pop one entry from the lifo pool.
 * 2. If dev_id matches: call queue->session_begin_fn to configure and
 *    relaunch the kernel.  On success return the session.  On failure
 *    (no kernel for this op/dtype), push the queue back and return NULL.
 * 3. If dev_id doesn't match: push the queue back and fall through to
 *    fresh allocation.
 * 4. Pool miss: iterate op components to allocate a fresh cmd_queue and
 *    call opc_session_begin.
 * -------------------------------------------------------------------------- */
ompi_op_gpu_session_t *
ompi_op_gpu_session_begin(struct ompi_op_t *op,
                          struct ompi_datatype_t *dtype,
                          int dev_id)
{
    /* Check pool for a reusable cmd_queue. */
    opal_list_item_t *item = opal_lifo_pop(&cmd_queue_pool);
    if (NULL != item) {
        opal_atomic_add_fetch_32(&cmd_queue_pool_count, -1);
        ompi_op_gpu_cmd_queue_t *q = (ompi_op_gpu_cmd_queue_t *) item;

        if (q->dev_id == dev_id) {
            ompi_op_gpu_session_t *s = q->session_begin_fn(q, op, dtype);
            if (NULL != s) {
                return s;
            }
            /* No GPU kernel for this (op, dtype).  Return the cmd_queue to
             * the pool so it can be reused for a future combination that does
             * have a kernel.  Caller falls back to ompi_op_reduce(). */
            cmd_queue_pool_push(q);
            return NULL;
        }

        /* Wrong device — push back and fall through to fresh allocation. */
        cmd_queue_pool_push(q);
    }

    /* Pool miss (or wrong device) — allocate a fresh cmd_queue. */
    mca_base_component_list_item_t *cli;
    OPAL_LIST_FOREACH(cli, &ompi_op_base_framework.framework_components,
                      mca_base_component_list_item_t) {
        const mca_base_component_t *bc = cli->cli_component;

        if (1 != bc->mca_type_major_version ||
            0 != bc->mca_type_minor_version ||
            0 != bc->mca_type_release_version) {
            continue;
        }

        const ompi_op_base_component_1_0_0_t *opc =
            (const ompi_op_base_component_1_0_0_t *) bc;

        if (NULL == opc->opc_cmd_queue_alloc ||
            NULL == opc->opc_session_begin) {
            continue;
        }

        ompi_op_gpu_cmd_queue_t *q = opc->opc_cmd_queue_alloc(dev_id);
        if (NULL == q) {
            continue;
        }

        /* Wire session_begin_fn into the cmd_queue. */
        q->session_begin_fn = opc->opc_session_begin;

        ompi_op_gpu_session_t *session = opc->opc_session_begin(q, op, dtype);
        if (NULL == session) {
            /* This component has no kernel for (op, dtype); discard the queue. */
            cmd_queue_destroy(q);
            continue;
        }

        return session;
    }

    return NULL;
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_reduce
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_reduce(ompi_op_gpu_session_t *session,
                           const void *src1, const void *src2,
                           void *dst, size_t count)
{
    session->reduce_fn(session, src1, src2, dst, count);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_end
 *
 * Stop the persistent kernel and return the cmd_queue to the pool.
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_end(ompi_op_gpu_session_t *session)
{
    if (NULL == session) {
        return;
    }

    /* Signal the kernel to exit and wait for the stream to drain. */
    session->stop_fn(session);

    ompi_op_gpu_cmd_queue_t *q = session->queue;
    free(session);

    cmd_queue_pool_push(q);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_query_bandwidth
 *
 * Iterate loaded op components looking for one that can answer a bandwidth
 * query for dev_id (mirrors the component-iteration in session_begin's pool
 * miss path, but has no cmd_queue/session side effects -- just a lookup).
 * -------------------------------------------------------------------------- */
int
ompi_op_gpu_session_query_bandwidth(int dev_id,
                                    double *device_bw_bytes_per_sec,
                                    double *link_bw_bytes_per_sec)
{
    mca_base_component_list_item_t *cli;
    OPAL_LIST_FOREACH(cli, &ompi_op_base_framework.framework_components,
                      mca_base_component_list_item_t) {
        const mca_base_component_t *bc = cli->cli_component;

        if (1 != bc->mca_type_major_version ||
            0 != bc->mca_type_minor_version ||
            0 != bc->mca_type_release_version) {
            continue;
        }

        const ompi_op_base_component_1_0_0_t *opc =
            (const ompi_op_base_component_1_0_0_t *) bc;

        if (NULL == opc->opc_query_bandwidth) {
            continue;
        }

        if (OMPI_SUCCESS == opc->opc_query_bandwidth(dev_id, device_bw_bytes_per_sec,
                                                      link_bw_bytes_per_sec)) {
            return OMPI_SUCCESS;
        }
    }

    return OMPI_ERR_NOT_SUPPORTED;
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_query_pcie_link_bandwidth
 * -------------------------------------------------------------------------- */
int
ompi_op_gpu_query_pcie_link_bandwidth(int pci_domain, int pci_bus, int pci_device,
                                      int pci_function, double *link_bw_bytes_per_sec)
{
    char path[256];
    char buf[64];
    FILE *f;
    double gt_per_sec;
    int width;

    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%d/current_link_speed",
             pci_domain, pci_bus, pci_device, pci_function);
    f = fopen(path, "r");
    if (NULL == f) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    if (NULL == fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return OMPI_ERR_NOT_SUPPORTED;
    }
    fclose(f);
    /* Format is e.g. "16.0 GT/s PCIe". */
    if (1 != sscanf(buf, "%lf", &gt_per_sec) || gt_per_sec <= 0.0) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%d/current_link_width",
             pci_domain, pci_bus, pci_device, pci_function);
    f = fopen(path, "r");
    if (NULL == f) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    if (NULL == fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return OMPI_ERR_NOT_SUPPORTED;
    }
    fclose(f);
    if (1 != sscanf(buf, "%d", &width) || width <= 0) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /* Gen1 (2.5 GT/s) and Gen2 (5.0 GT/s) use 8b/10b line coding (80%
     * efficient); Gen3+ (8.0 GT/s and up) use 128b/130b (~98.46% efficient). */
    double efficiency = (gt_per_sec < 6.0) ? 0.80 : (128.0 / 130.0);
    double bytes_per_sec_per_lane = (gt_per_sec * 1.0e9 / 8.0) * efficiency;

    *link_bw_bytes_per_sec = bytes_per_sec_per_lane * (double) width;
    return OMPI_SUCCESS;
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_pool_finalize
 *
 * Drain the pool, release all GPU resources, and destroy the lifo.
 * Called once from ompi_op_base_close() during MPI_Finalize.
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_pool_finalize(void)
{
    opal_list_item_t *item;
    while (NULL != (item = opal_lifo_pop(&cmd_queue_pool))) {
        cmd_queue_destroy((ompi_op_gpu_cmd_queue_t *) item);
    }
    OBJ_DESTRUCT(&cmd_queue_pool);
}
