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
 * Session lifecycle for the CUDA persistent-kernel op component.
 *
 * ompi_op_cuda_cmd_queue_t inherits ompi_op_gpu_cmd_queue_t.  OBJ_NEW
 * allocates the object; the OBJ destructor releases the CUDA stream and
 * device/host memory.  The component returns the base pointer from alloc so
 * callers need no knowledge of the concrete type.
 *
 * The command slot the persistent kernel polls (queue->cmd) lives in plain
 * device memory, not cudaMallocManaged: it's touched by the host and the
 * kernel on every single reduction call, and that back-and-forth drove
 * constant page migration faults under Unified Memory. Instead the host
 * stages each command in a registered (page-locked) mirror, cq->host_cmd,
 * and moves it to/from the device slot with explicit cudaMemcpyAsync on a
 * dedicated ctrl_stream. ctrl_stream must be different from the persistent
 * kernel's own stream: that stream is occupied indefinitely by the kernel,
 * so anything enqueued there would queue behind it and never run.
 *
 * There is no separate shutdown flag/allocation: cmd->status doubles as the
 * shutdown signal (a negative value requests it), posted through the exact
 * same host_cmd/ctrl_stream channel as a normal reduction. The kernel
 * resets status back to 0 itself before exiting (see op_cuda_kernels.cu),
 * so a cmd_queue pulled back out of the pool is always already idle --
 * session_begin never needs to push a reset.
 *
 * The persistent kernel is also launched lazily, on the first
 * session_reduce() call rather than in session_begin(): not every rank
 * that creates a session ends up reducing anything (e.g. leaf ranks in a
 * reduction tree only forward data upward), so deferring the actual launch
 * avoids paying kernel launch/teardown cost for sessions that turn out to
 * do zero reductions. ompi_op_gpu_session_t::started tracks whether this
 * session's launch has happened yet.
 *
 * session_begin:   look up the kernel for (op, dtype) -- returns NULL if
 *                  none exists -- and stash the resolved launcher for the
 *                  deferred launch.  Does not touch the cmd_queue or launch
 *                  anything.
 *
 * session_reduce:  on the first call, launch the persistent kernel.  Then
 *                  stage src/dst/count and status=1 into host_cmd, push it
 *                  to the device slot, and poll the device slot's status by
 *                  copying it back until it reads 2.
 *
 * session_stop:    a no-op if the kernel was never launched (session never
 *                  reduced anything).  Otherwise push the shutdown sentinel
 *                  and synchronize the kernel's stream to wait for it to
 *                  exit.  The cmd_queue's GPU stream and device/host memory
 *                  remain allocated for reuse.
 */

#include "ompi_config.h"
#include <stdbool.h>
#include <stdlib.h>
#include <sched.h>

#include <cuda_runtime.h>

#include "opal/util/output.h"
#include "opal/mca/accelerator/base/base.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/cuda/op_cuda.h"

/* ompi_op_ddt_map[] maps dtype->id → OMPI_OP_BASE_TYPE_* (-1 if none) */
extern int ompi_op_ddt_map[OMPI_DATATYPE_MAX_PREDEFINED];

/* Forward declarations of static session hooks referenced from session_begin. */
static void ompi_op_cuda_session_reduce(ompi_op_gpu_session_t *session,
                                         const void *src1, const void *src2,
                                         void *dst, size_t count);
static void ompi_op_cuda_session_stop(ompi_op_gpu_session_t *session);

/* --------------------------------------------------------------------------
 * OBJ constructor / destructor for ompi_op_cuda_cmd_queue_t
 * -------------------------------------------------------------------------- */
static void
ompi_op_cuda_cmd_queue_construct(ompi_op_cuda_cmd_queue_t *q)
{
    q->stream         = NULL;
    q->host_cmd       = NULL;
    q->ctrl_stream    = NULL;
    q->super.cmd      = NULL;
    q->super.dev_id   = -1;
    q->super.allocator = NULL;
    q->super.session_begin_fn = NULL;
}

static void
ompi_op_cuda_cmd_queue_destruct(ompi_op_cuda_cmd_queue_t *q)
{
    if (NULL != q->stream) {
        cudaStreamDestroy(q->stream);
        q->stream = NULL;
    }
    if (NULL != q->ctrl_stream) {
        cudaStreamDestroy(q->ctrl_stream);
        q->ctrl_stream = NULL;
    }
    if (NULL != q->super.cmd) {
        cudaFree(q->super.cmd);
        q->super.cmd = NULL;
    }
    if (NULL != q->host_cmd) {
        cudaFreeHost(q->host_cmd);
        q->host_cmd = NULL;
    }
}

OBJ_CLASS_INSTANCE(ompi_op_cuda_cmd_queue_t,
                   ompi_op_gpu_cmd_queue_t,
                   ompi_op_cuda_cmd_queue_construct,
                   ompi_op_cuda_cmd_queue_destruct);

/* --------------------------------------------------------------------------
 * ompi_op_cuda_cmd_queue_alloc
 *
 * Allocate the expensive GPU resources for one device: a device-resident
 * command slot (polled by the persistent kernel), a registered host mirror
 * of it (staged and pushed to the device slot by the host on every
 * reduction call, and to signal shutdown), and two private CUDA streams --
 * one for the persistent kernel, one for host<->device transfers. Returns
 * the base pointer (ompi_op_gpu_cmd_queue_t *); NULL on failure.
 * -------------------------------------------------------------------------- */
ompi_op_gpu_cmd_queue_t *
ompi_op_cuda_cmd_queue_alloc(int dev_id)
{
    ompi_op_cuda_cmd_queue_t *q = OBJ_NEW(ompi_op_cuda_cmd_queue_t);
    if (NULL == q) {
        return NULL;
    }

    cudaError_t err;

    /* Device-resident command slot: what the persistent kernel polls. */
    err = cudaMalloc((void **) &q->super.cmd, sizeof(ompi_op_gpu_cmd_t));
    if (cudaSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }

    /* Registered (page-locked) host mirror: staged by the host with plain
     * stores, and used as the source/destination of the explicit
     * cudaMemcpyAsync calls that move commands to/from the device slot. */
    err = cudaMallocHost((void **) &q->host_cmd, sizeof(ompi_op_gpu_cmd_t));
    if (cudaSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }
    q->host_cmd->src1   = NULL;
    q->host_cmd->src2   = NULL;
    q->host_cmd->dst    = NULL;
    q->host_cmd->count  = 0;
    q->host_cmd->status = 0;

    /* Push the initial (idle) state down to the device slot. No kernel is
     * running yet at this point (the persistent kernel is launched lazily,
     * on the first reduction), so this just establishes status == 0 for
     * whichever launch eventually happens first. */
    err = cudaMemcpy(q->super.cmd, q->host_cmd, sizeof(ompi_op_gpu_cmd_t),
                     cudaMemcpyHostToDevice);
    if (cudaSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }

    /* Persistent-kernel compute stream. */
    err = cudaStreamCreateWithFlags(&q->stream, cudaStreamNonBlocking);
    if (cudaSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }

    /* Dedicated stream for host<->device cmd transfers -- must differ from
     * q->stream, which the persistent kernel occupies indefinitely. */
    err = cudaStreamCreateWithFlags(&q->ctrl_stream, cudaStreamNonBlocking);
    if (cudaSuccess != err) {
        OBJ_RELEASE(q);
        return NULL;
    }

    q->super.dev_id    = dev_id;
    q->super.allocator = opal_accelerator_base_get_device_allocator(dev_id);
    return &q->super;
}

/* --------------------------------------------------------------------------
 * ompi_op_cuda_session_begin
 *
 * Look up the GPU kernel for (op, dtype) and wire the session's dispatch
 * hooks. Returns NULL if no GPU kernel exists for this combination. The
 * persistent kernel itself is launched lazily by session_reduce() on its
 * first call, not here -- see the file-level comment above.
 * -------------------------------------------------------------------------- */
ompi_op_gpu_session_t *
ompi_op_cuda_session_begin(ompi_op_gpu_cmd_queue_t *queue,
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

    ompi_op_cuda_launcher_fn_t launcher = ompi_op_cuda_kernel_fns[op_idx][type_idx];
    if (NULL == launcher) {
        return NULL;
    }

    /* This only validates that a kernel exists for (op, dtype) and stashes
     * it for the deferred launch -- it does not touch the cmd_queue or
     * launch anything. cmd_queue_alloc() already left queue->cmd idle
     * (status == 0), and the kernel resets it back to idle itself before
     * exiting on shutdown, so a reused queue is always ready without a
     * reset here. See ompi_op_cuda_session_reduce() for the actual launch. */
    ompi_op_gpu_session_t *session =
        (ompi_op_gpu_session_t *) malloc(sizeof(ompi_op_gpu_session_t));
    if (NULL == session) {
        return NULL;
    }

    session->queue     = queue;
    session->allocator = queue->allocator;
    session->reduce_fn = ompi_op_cuda_session_reduce;
    session->stop_fn   = ompi_op_cuda_session_stop;
    session->started   = false;
    session->launcher  = (void *) launcher;
    return session;
}

/* --------------------------------------------------------------------------
 * ompi_op_cuda_session_reduce
 * -------------------------------------------------------------------------- */
static void
ompi_op_cuda_session_reduce(ompi_op_gpu_session_t *session,
                             const void *src1, const void *src2,
                             void *dst, size_t count)
{
    ompi_op_cuda_cmd_queue_t *cq = (ompi_op_cuda_cmd_queue_t *) session->queue;
    ompi_op_gpu_cmd_t *host_cmd  = cq->host_cmd;
    ompi_op_gpu_cmd_t *dev_cmd   = session->queue->cmd;

    if (!session->started) {
        /* Lazy launch: this is the first (and possibly only) reduction
         * this session performs. Many sessions never reach this point at
         * all (e.g. leaf ranks in a reduction tree only forward data
         * upward), so deferring the launch to here avoids paying
         * persistent-kernel launch/teardown cost for those. */
        ompi_op_cuda_launcher_fn_t launcher = (ompi_op_cuda_launcher_fn_t) session->launcher;
        launcher(dev_cmd, cq->stream, cq->super.dev_id);
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            /* session_begin already validated that a kernel exists for this
             * (op, dtype), so a failure here is a genuine CUDA runtime/driver
             * error, not a "no kernel" case. There is no viable host-side
             * fallback for a reduction over device-resident buffers, and
             * silently continuing would just hang forever in the poll loop
             * below waiting for a status update a kernel that never started
             * will never produce -- so treat this as fatal. */
            opal_output(0, "op/cuda: persistent kernel launch failed on device %d: %s",
                       cq->super.dev_id, cudaGetErrorString(err));
            abort();
        }
        session->started = true;
    }

    /* Stage the command in registered host memory, then push it to the
     * device-resident slot the persistent kernel polls, on the dedicated
     * ctrl_stream (cq->stream is occupied indefinitely by the kernel
     * itself, so anything enqueued there would never run). */
    host_cmd->src1   = src1;
    host_cmd->src2   = src2;
    host_cmd->dst    = dst;
    host_cmd->count  = (int64_t) count;
    host_cmd->status = 1;
    cudaMemcpyAsync(dev_cmd, host_cmd, sizeof(*host_cmd),
                    cudaMemcpyHostToDevice, cq->ctrl_stream);

    /* Poll for completion by copying the status word back. Stream ordering
     * guarantees each of these copies lands only after the post above (and
     * after the kernel's own write once it runs), so no extra
     * synchronization is needed beyond waiting for each individual copy.
     * Spin on cudaStreamQuery rather than cudaStreamSynchronize so a
     * pending wait busy-polls instead of taking the driver's blocking/sleep
     * path, matching the low-latency spin this replaced. */
    do {
        cudaMemcpyAsync((void *) &host_cmd->status, (const void *) &dev_cmd->status,
                        sizeof(dev_cmd->status), cudaMemcpyDeviceToHost, cq->ctrl_stream);
        while (cudaSuccess != cudaStreamQuery(cq->ctrl_stream)) {
            sched_yield();   /* relinquish CPU timeslice while waiting */
        }
    } while (2 != host_cmd->status);

    /* No separate reset-to-0 round trip is needed: the next call's post
     * overwrites the device status with 1 directly, and the kernel's own
     * wait loop only ever tests for == 1, so a stale 2 left behind here is
     * harmless. */
}

/* --------------------------------------------------------------------------
 * ompi_op_cuda_session_stop
 *
 * A no-op if the persistent kernel was never launched (this session never
 * reduced anything). Otherwise push the shutdown sentinel (a negative
 * status) through the host_cmd/ctrl_stream channel and wait for the
 * kernel's own stream to drain -- the kernel only returns once it observes
 * the update, so this one synchronize is sufficient without a separate
 * wait on ctrl_stream. The cmd_queue's stream and device/host memory
 * remain allocated for reuse.
 * -------------------------------------------------------------------------- */
static void
ompi_op_cuda_session_stop(ompi_op_gpu_session_t *session)
{
    if (!session->started) {
        return;
    }

    ompi_op_cuda_cmd_queue_t *cq = (ompi_op_cuda_cmd_queue_t *) session->queue;
    ompi_op_gpu_cmd_t *host_cmd  = cq->host_cmd;
    ompi_op_gpu_cmd_t *dev_cmd   = cq->super.cmd;

    /* Signal the kernel to exit its loop */
    host_cmd->status = -1;
    cudaMemcpyAsync((void *) &dev_cmd->status, (const void *) &host_cmd->status,
                    sizeof(host_cmd->status), cudaMemcpyHostToDevice, cq->ctrl_stream);

    /* Wait for the kernel to finish; stream remains valid after this */
    cudaStreamSynchronize(cq->stream);
}
