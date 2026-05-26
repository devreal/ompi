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
 * ompi_op_rocm_cmd_queue_t inherits ompi_op_gpu_cmd_queue_t.  OBJ_NEW
 * allocates the object; the OBJ destructor releases the HIP stream and
 * managed memory.  The component returns the base pointer from alloc so
 * callers need no knowledge of the concrete type.
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
 * OBJ constructor / destructor for ompi_op_rocm_cmd_queue_t
 * -------------------------------------------------------------------------- */
static void
ompi_op_rocm_cmd_queue_construct(ompi_op_rocm_cmd_queue_t *q)
{
    q->shutdown       = NULL;
    q->stream         = NULL;
    q->super.cmd      = NULL;
    q->super.dev_id   = -1;
    q->super.allocator = NULL;
    q->super.session_begin_fn = NULL;
}

static void
ompi_op_rocm_cmd_queue_destruct(ompi_op_rocm_cmd_queue_t *q)
{
    if (NULL != q->stream) {
        hipStreamDestroy(q->stream);
        q->stream = NULL;
    }
    if (NULL != q->shutdown) {
        hipFree((void *) q->shutdown);
        q->shutdown = NULL;
    }
    if (NULL != q->super.cmd) {
        hipFree(q->super.cmd);
        q->super.cmd = NULL;
    }
}

OBJ_CLASS_INSTANCE(ompi_op_rocm_cmd_queue_t,
                   ompi_op_gpu_cmd_queue_t,
                   ompi_op_rocm_cmd_queue_construct,
                   ompi_op_rocm_cmd_queue_destruct);

/* --------------------------------------------------------------------------
 * ompi_op_rocm_cmd_queue_alloc
 * -------------------------------------------------------------------------- */
ompi_op_gpu_cmd_queue_t *
ompi_op_rocm_cmd_queue_alloc(int dev_id)
{
    ompi_op_rocm_cmd_queue_t *q = OBJ_NEW(ompi_op_rocm_cmd_queue_t);
    if (NULL == q) {
        return NULL;
    }

    hipError_t err;

    /* Allocate managed-memory command slot (accessible by both CPU and GPU) */
    err = hipMallocManaged((void **) &q->super.cmd,
                           sizeof(ompi_op_gpu_cmd_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }
    q->super.cmd->src1   = NULL;
    q->super.cmd->src2   = NULL;
    q->super.cmd->dst    = NULL;
    q->super.cmd->count  = 0;
    q->super.cmd->status = 0;

    /* Allocate managed-memory shutdown flag */
    err = hipMallocManaged((void **) &q->shutdown,
                           sizeof(int32_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }
    *q->shutdown = 0;

    /* Create a dedicated non-blocking stream for this cmd_queue */
    err = hipStreamCreateWithFlags(&q->stream, hipStreamNonBlocking);
    if (hipSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }

    q->super.dev_id    = dev_id;
    q->super.allocator = opal_accelerator_base_get_device_allocator(dev_id);
    return &q->super;
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

    ompi_op_rocm_cmd_queue_t *cq = (ompi_op_rocm_cmd_queue_t *) queue;

    /* Reset queue state for the new kernel */
    *cq->shutdown      = 0;
    queue->cmd->src1   = NULL;
    queue->cmd->src2   = NULL;
    queue->cmd->dst    = NULL;
    queue->cmd->count  = 0;
    queue->cmd->status = 0;

    /* Launch the persistent kernel (1 block, 256 threads) */
    launcher(queue->cmd, cq->shutdown, cq->stream);
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
    ompi_op_rocm_cmd_queue_t *cq = (ompi_op_rocm_cmd_queue_t *) session->queue;

    /* Signal the kernel to exit its loop */
    *cq->shutdown = 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Wait for the kernel to finish; stream remains valid after this */
    hipStreamSynchronize(cq->stream);
}
