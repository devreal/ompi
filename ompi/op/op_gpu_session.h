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

#ifndef OMPI_OP_GPU_SESSION_H
#define OMPI_OP_GPU_SESSION_H

#include "ompi_config.h"
#include <stdbool.h>
#include <stdint.h>
#include "opal/mca/allocator/allocator.h"

BEGIN_C_DECLS

struct ompi_op_t;
struct ompi_datatype_t;

/**
 * Managed-memory command slot shared between the host and the persistent
 * reduction kernel (accessible by both CPU and GPU via managed/unified memory).
 *
 * status lifecycle (per reduction call):
 *   0 = idle       (initial; host resets after kernel signals done)
 *   1 = work_ready (host → kernel: pointers and count are valid)
 *   2 = done       (kernel → host: reduction complete)
 */
typedef struct {
    const void      *src;
    void            *dst;
    int64_t          count;
    volatile int32_t status;
} ompi_op_gpu_cmd_t;

/**
 * Per-collective GPU reduction session.  Created by ompi_op_gpu_session_begin()
 * before a collective algorithm's reduction loop starts, and returned to the
 * session pool by ompi_op_gpu_session_end() for reuse by a future collective.
 *
 * Pool lifecycle: session_end() stops the persistent kernel (GPU resources
 * remain allocated) and pushes the session onto a freelist.  A future
 * session_begin() for the same dev_id pops the idle session and calls
 * restart_fn to reconfigure and relaunch the appropriate kernel — no
 * cudaMalloc/hipMalloc or stream creation overhead on the reuse path.
 *
 * When no GPU op component supports the (op, dtype) combination, begin()
 * returns NULL and all callers fall back to ompi_op_reduce().
 *
 * reduce_fn, stop_fn, restart_fn, free_fn, and pool_next are managed by
 * op_gpu_session.c — callers must not set them directly.
 */
typedef struct ompi_op_gpu_session_t {
    int                          dev_id;
    mca_allocator_base_module_t *allocator;  /* GPU scratch allocator for this session */
    void                        *backend;    /* opaque: cuda or rocm session state */
    /* Dispatch hooks wired at session_begin time. */
    void (*reduce_fn)(struct ompi_op_gpu_session_t *session,
                      const void *src, void *dst, size_t count);
    /* Signal the persistent kernel to exit and synchronize the stream.
     * GPU stream and managed memory remain allocated for reuse. */
    void (*stop_fn)(struct ompi_op_gpu_session_t *session);
    /* Reconfigure an idle session for a new (op, dtype) and relaunch the
     * persistent kernel.  Returns false if no GPU kernel exists for this
     * combination (caller must then free the session and return NULL). */
    bool (*restart_fn)(struct ompi_op_gpu_session_t *session,
                       struct ompi_op_t *op,
                       struct ompi_datatype_t *dtype);
    /* Release managed memory, GPU stream, and backend private state.
     * Must NOT free the ompi_op_gpu_session_t struct itself. */
    void (*free_fn)(struct ompi_op_gpu_session_t *session);
    /* Pool bookkeeping — do not access directly. */
    struct ompi_op_gpu_session_t *pool_next;
} ompi_op_gpu_session_t;

/**
 * Create a GPU reduction session and launch a persistent reduction kernel.
 * Returns NULL if no GPU op component supports this (op, dtype) combination
 * or if no GPU op component is loaded — the caller must then use ompi_op_reduce.
 */
OMPI_DECLSPEC ompi_op_gpu_session_t *ompi_op_gpu_session_begin(struct ompi_op_t *op,
                                                                struct ompi_datatype_t *dtype,
                                                                int dev_id);

/**
 * Post one reduction command (src op dst → dst) to the persistent kernel and
 * wait for completion.  Behavior is undefined if session is NULL.
 */
OMPI_DECLSPEC void ompi_op_gpu_session_reduce(ompi_op_gpu_session_t *session,
                                               const void *src, void *dst, size_t count);

/**
 * Stop the persistent kernel and return the session to the pool for reuse.
 * GPU stream and managed memory remain allocated; a future begin() call for
 * the same dev_id will relaunch the kernel without allocating new resources.
 * NULL-safe.
 */
OMPI_DECLSPEC void ompi_op_gpu_session_end(ompi_op_gpu_session_t *session);

/**
 * Drain and permanently destroy all pooled sessions.  Must be called once
 * during MPI finalization (from ompi_op_base_close).
 */
OMPI_DECLSPEC void ompi_op_gpu_session_pool_finalize(void);

END_C_DECLS

#endif /* OMPI_OP_GPU_SESSION_H */
