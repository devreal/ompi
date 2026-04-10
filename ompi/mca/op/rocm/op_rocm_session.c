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
 * session_begin:   look up the kernel in the 2D launcher table, allocate
 *                  managed-memory command slot + shutdown flag, create a
 *                  private HIP stream, and launch the persistent kernel.
 *
 * session_reduce:  write src/dst/count to the command slot, set status=1
 *                  to wake the kernel, and spin until status==2.
 *
 * session_stop:    signal the persistent kernel to exit and synchronize the
 *                  stream.  GPU stream and managed memory remain allocated
 *                  so the session can be reused via session_restart.
 *
 * session_restart: reconfigure an idle (stopped) session for a new (op, dtype)
 *                  combination and relaunch the appropriate persistent kernel.
 *                  Returns false if no GPU kernel exists for the combination.
 *
 * session_free:    release the HIP stream, managed memory, and backend
 *                  private state when a session is permanently discarded.
 *                  Does NOT free the ompi_op_gpu_session_t struct.
 */

#include "ompi_config.h"
#include <stdbool.h>
#include <stdlib.h>
#include <sched.h>

#include <hip/hip_runtime.h>

#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/rocm/op_rocm.h"

/* ompi_op_ddt_map[] maps dtype->id → OMPI_OP_BASE_TYPE_* (-1 if none) */
extern int ompi_op_ddt_map[OMPI_DATATYPE_MAX_PREDEFINED];

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_begin
 * -------------------------------------------------------------------------- */
ompi_op_gpu_session_t *
ompi_op_rocm_session_begin(struct ompi_op_t *op,
                            struct ompi_datatype_t *dtype,
                            int dev_id)
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
        return NULL;   /* no GPU kernel for this (op, type) combination */
    }

    /* Allocate the public session struct returned to the caller */
    ompi_op_gpu_session_t *session =
        (ompi_op_gpu_session_t *) malloc(sizeof(ompi_op_gpu_session_t));
    if (NULL == session) {
        return NULL;
    }

    /* Allocate component-private state */
    ompi_op_rocm_session_priv_t *priv =
        (ompi_op_rocm_session_priv_t *) malloc(sizeof(ompi_op_rocm_session_priv_t));
    if (NULL == priv) {
        free(session);
        return NULL;
    }

    hipError_t err;

    /* Allocate managed-memory command slot (accessible by both CPU and GPU) */
    err = hipMallocManaged((void **) &priv->cmd,
                           sizeof(ompi_op_gpu_cmd_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        free(priv);
        free(session);
        return NULL;
    }
    priv->cmd->src    = NULL;
    priv->cmd->dst    = NULL;
    priv->cmd->count  = 0;
    priv->cmd->status = 0;

    /* Allocate managed-memory shutdown flag */
    err = hipMallocManaged((void **) &priv->shutdown,
                           sizeof(int32_t),
                           hipMemAttachGlobal);
    if (hipSuccess != err) {
        hipFree(priv->cmd);
        free(priv);
        free(session);
        return NULL;
    }
    *priv->shutdown = 0;

    /* Create a dedicated non-blocking stream for this session */
    err = hipStreamCreateWithFlags(&priv->stream, hipStreamNonBlocking);
    if (hipSuccess != err) {
        hipFree(priv->shutdown);
        hipFree(priv->cmd);
        free(priv);
        free(session);
        return NULL;
    }

    /* Launch the persistent kernel (1 block, 256 threads) */
    launcher(priv->cmd, priv->shutdown, priv->stream);
    err = hipGetLastError();
    if (hipSuccess != err) {
        hipStreamDestroy(priv->stream);
        hipFree(priv->shutdown);
        hipFree(priv->cmd);
        free(priv);
        free(session);
        return NULL;
    }

    session->dev_id    = dev_id;
    session->allocator = NULL;   /* scratch allocator wired in Phase 4 */
    session->backend   = priv;

    return session;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_reduce
 * -------------------------------------------------------------------------- */
void
ompi_op_rocm_session_reduce(ompi_op_gpu_session_t *session,
                             const void *src, void *dst, size_t count)
{
    ompi_op_rocm_session_priv_t *priv =
        (ompi_op_rocm_session_priv_t *) session->backend;

    /* Write operands before signalling the kernel */
    priv->cmd->src   = src;
    priv->cmd->dst   = dst;
    priv->cmd->count = (int64_t) count;

    __atomic_thread_fence(__ATOMIC_SEQ_CST);   /* ensure writes visible to GPU */
    priv->cmd->status = 1;                     /* wake the kernel */

    /* Spin-wait for the kernel to signal completion */
    while (2 != priv->cmd->status) {
        sched_yield();   /* relinquish CPU timeslice while waiting */
    }

    /* Reset for the next call */
    priv->cmd->status = 0;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_stop
 *
 * Signal the persistent kernel to exit and wait for the stream to drain.
 * The HIP stream and managed memory remain allocated so the session can be
 * recycled via ompi_op_rocm_session_restart.
 * -------------------------------------------------------------------------- */
void
ompi_op_rocm_session_stop(ompi_op_gpu_session_t *session)
{
    ompi_op_rocm_session_priv_t *priv =
        (ompi_op_rocm_session_priv_t *) session->backend;

    /* Signal the kernel to exit its loop */
    *priv->shutdown = 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    /* Wait for the kernel to finish; stream remains valid after this */
    hipStreamSynchronize(priv->stream);
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_restart
 *
 * Reconfigure an idle (stopped) session for a new (op, dtype) combination
 * and relaunch the appropriate persistent kernel.  Returns false if no GPU
 * kernel exists for this combination.
 * -------------------------------------------------------------------------- */
bool
ompi_op_rocm_session_restart(ompi_op_gpu_session_t *session,
                              struct ompi_op_t *op,
                              struct ompi_datatype_t *dtype)
{
    int op_idx   = op->o_f_to_c_index;
    int type_idx = (dtype->id < OMPI_DATATYPE_MAX_PREDEFINED)
                   ? ompi_op_ddt_map[dtype->id] : -1;

    if (op_idx  < 0 || op_idx  >= OMPI_OP_BASE_FORTRAN_OP_MAX ||
        type_idx < 0 || type_idx >= OMPI_OP_BASE_TYPE_MAX) {
        return false;
    }

    ompi_op_rocm_launcher_fn_t launcher = ompi_op_rocm_kernel_fns[op_idx][type_idx];
    if (NULL == launcher) {
        return false;
    }

    ompi_op_rocm_session_priv_t *priv =
        (ompi_op_rocm_session_priv_t *) session->backend;

    /* Reset state for the new kernel */
    *priv->shutdown   = 0;
    priv->cmd->src    = NULL;
    priv->cmd->dst    = NULL;
    priv->cmd->count  = 0;
    priv->cmd->status = 0;

    /* Launch the persistent kernel for the new (op, dtype) */
    launcher(priv->cmd, priv->shutdown, priv->stream);
    hipError_t err = hipGetLastError();
    if (hipSuccess != err) {
        return false;
    }

    return true;
}

/* --------------------------------------------------------------------------
 * ompi_op_rocm_session_free
 *
 * Free the HIP stream, managed memory, and backend private state.
 * Does NOT free the ompi_op_gpu_session_t struct (that is the caller's
 * responsibility, done by session_destroy in op_gpu_session.c).
 * -------------------------------------------------------------------------- */
void
ompi_op_rocm_session_free(ompi_op_gpu_session_t *session)
{
    ompi_op_rocm_session_priv_t *priv =
        (ompi_op_rocm_session_priv_t *) session->backend;
    if (NULL == priv) {
        return;
    }

    hipStreamDestroy(priv->stream);
    hipFree((void *) priv->shutdown);
    hipFree(priv->cmd);
    free(priv);
    session->backend = NULL;
}
