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
 * Session lifecycle for the ROCm persistent-kernel op component.
 * Mirrors op_cuda_session.c with hip* API calls in place of cuda*.
 *
 * cmd_queue_alloc: allocate managed-memory command slot + shutdown flag
 *                  and create a private HIP stream.
 *
 * cmd_queue_free:  release the HIP stream, managed memory, and
 *                  component-private state.
 *
 * session_begin:   look up the kernel for (op, dtype), reset the cmd_queue
 *                  state, and launch the persistent kernel on the existing
 *                  stream.  Wires all session dispatch hooks and returns the
 *                  session.  Returns NULL if no kernel exists.
 *
 * session_reduce:  write src/dst/count to the command slot, set status=1
 *                  to wake the kernel, and spin until status==2.
 *
 * session_stop:    signal the persistent kernel to exit and synchronize the
 *                  stream.  The cmd_queue's HIP stream and managed memory
 *                  remain allocated for reuse.
 */

#include "ompi_config.h"
#include <stdbool.h>
#include <stdlib.h>
#include <sched.h>

#include <hip/hip_runtime.h>

#include "opal/mca/accelerator/base/base.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/rocm/op_rocm.h"

/* ompi_op_ddt_map[] maps dtype->id → OMPI_OP_BASE_TYPE_* (-1 if none) */
extern int ompi_op_ddt_map[OMPI_DATATYPE_MAX_PREDEFINED];

/* Forward declarations of static session hooks referenced from session_begin. */
static void ompi_op_rocm_session_reduce(ompi_op_gpu_session_t *session,
                                         const void *src1, const void *src2,
                                         void *dst, size_t count);
static void ompi_op_rocm_session_stop(ompi_op_gpu_session_t *session);

/* --------------------------------------------------------------------------
 * ompi_op_rocm_cmd_queue_alloc
 * -------------------------------------------------------------------------- */
ompi_op_gpu_cmd_queue_t *
ompi_op_rocm_cmd_queue_alloc(int dev_id)
{
    ompi_op_gpu_cmd_queue_t *queue =
        (ompi_op_gpu_cmd_queue_t *) malloc(sizeof(ompi_op_gpu_cmd_queue_t));
    if (NULL == queue) {
        return NULL;
    }
    OBJ_CONSTRUCT(&queue->super, opal_list_item_t);

    ompi_op_rocm_cmd_queue_priv_t *priv =
        (ompi_op_rocm_cmd_queue_priv_t *) malloc(sizeof(ompi_op_rocm_cmd_queue_priv_t));
    if (NULL == priv) {
        free(queue);
        return NULL;
    }

    hipError_t err;

    /* Allocate managed-memory command slot (accessible by both CPU and GPU) */
    err = hipMallocManaged((void **) &queue->cmd,
                           sizeof(ompi_op_gpu_cmd_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        free(priv);
        free(queue);
        return NULL;
    }
    queue->cmd->src1   = NULL;
    queue->cmd->src2   = NULL;
    queue->cmd->dst    = NULL;
    queue->cmd->count  = 0;
    queue->cmd->status = 0;

    /* Allocate managed-memory shutdown flag */
    err = hipMallocManaged((void **) &priv->shutdown,
                           sizeof(int32_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        hipFree(queue->cmd);
        free(priv);
        free(queue);
        return NULL;
    }
    *priv->shutdown = 0;

    /* Create a dedicated non-blocking stream for this cmd_queue */
    err = hipStreamCreateWithFlags(&priv->stream, hipStreamNonBlocking);
    if (hipSuccess != err) {
        hipFree(priv->shutdown);
        hipFree(queue->cmd);
        free(priv);
        free(queue);
        return NULL;
    }

    queue->dev_id    = dev_id;
    queue->allocator = opal_accelerator_base_get_device_allocator(dev_id);
    queue->priv      = priv;
    return queue;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_cmd_queue_free
 *
 * Release the HIP stream, managed memory, and component-private state.
 * Does NOT free the ompi_op_gpu_cmd_queue_t struct itself.
 * -------------------------------------------------------------------------- */
void
ompi_op_rocm_cmd_queue_free(ompi_op_gpu_cmd_queue_t *queue)
{
    ompi_op_rocm_cmd_queue_priv_t *priv =
        (ompi_op_rocm_cmd_queue_priv_t *) queue->priv;
    if (NULL == priv) {
        return;
    }

    hipStreamDestroy(priv->stream);
    hipFree((void *) priv->shutdown);
    hipFree(queue->cmd);
    free(priv);
    queue->priv = NULL;
    queue->cmd  = NULL;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_begin
 * -------------------------------------------------------------------------- */
ompi_op_gpu_session_t *
ompi_op_rocm_session_begin(ompi_op_gpu_cmd_queue_t *queue,
                            struct ompi_op_t *op,
                            struct ompi_datatype_t *dtype)
{
    int op_idx   = op->o_f_to_c_index;
    int type_idx = (dtype->id < OMPI_DATATYPE_MAX_PREDEFINED)
                   ? ompi_op_ddt_map[dtype->id] : -1;

    if (op_idx  < 0 || op_idx  >= OMPI_OP_BASE_FORTRAN_OP_MAX ||
        type_idx < 0 || type_idx >= OMPI_OP_BASE_TYPE_MAX) {
        return NULL;
    }

    ompi_op_rocm_launcher_fn_t launcher = ompi_op_rocm_kernel_fns[op_idx][type_idx];
    if (NULL == launcher) {
        return NULL;
    }

    ompi_op_rocm_cmd_queue_priv_t *priv =
        (ompi_op_rocm_cmd_queue_priv_t *) queue->priv;

    /* Reset queue state for the new kernel */
    *priv->shutdown    = 0;
    queue->cmd->src1   = NULL;
    queue->cmd->src2   = NULL;
    queue->cmd->dst    = NULL;
    queue->cmd->count  = 0;
    queue->cmd->status = 0;

    /* Launch the persistent kernel (1 block, 256 threads) */
    launcher(queue->cmd, priv->shutdown, priv->stream);
    hipError_t err = hipGetLastError();
    if (hipSuccess != err) {
        return NULL;
    }

    ompi_op_gpu_session_t *session =
        (ompi_op_gpu_session_t *) malloc(sizeof(ompi_op_gpu_session_t));
    if (NULL == session) {
        return NULL;
    }

    session->queue     = queue;
    session->allocator = queue->allocator;
    session->reduce_fn = ompi_op_rocm_session_reduce;
    session->stop_fn   = ompi_op_rocm_session_stop;
    return session;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_reduce
 * -------------------------------------------------------------------------- */
static void
ompi_op_rocm_session_reduce(ompi_op_gpu_session_t *session,
                             const void *src1, const void *src2,
                             void *dst, size_t count)
{
    ompi_op_gpu_cmd_t *cmd = session->queue->cmd;

    /* Write operands before signalling the kernel */
    cmd->src1  = src1;
    cmd->src2  = src2;
    cmd->dst   = dst;
    cmd->count = (int64_t) count;

    __atomic_thread_fence(__ATOMIC_SEQ_CST);   /* ensure writes visible to GPU */
    cmd->status = 1;                           /* wake the kernel */

    /* Spin-wait for the kernel to signal completion */
    while (2 != cmd->status) {
        sched_yield();   /* relinquish CPU timeslice while waiting */
    }

    /* Reset for the next call */
    cmd->status = 0;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_stop
 *
 * Signal the persistent kernel to exit and wait for the stream to drain.
 * The cmd_queue's stream and managed memory remain allocated for reuse.
 * -------------------------------------------------------------------------- */
static void
ompi_op_rocm_session_stop(ompi_op_gpu_session_t *session)
{
    ompi_op_rocm_cmd_queue_priv_t *priv =
        (ompi_op_rocm_cmd_queue_priv_t *) session->queue->priv;

    /* Signal the kernel to exit its loop */
    *priv->shutdown = 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Wait for the kernel to finish; stream remains valid after this */
    hipStreamSynchronize(priv->stream);
}
