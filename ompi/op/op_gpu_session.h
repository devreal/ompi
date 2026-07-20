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
#include "opal/class/opal_list.h"
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
    const void      *src1;
    const void      *src2;
    void            *dst;
    int64_t          count;
    volatile int32_t status;
} ompi_op_gpu_cmd_t;

/**
 * The expensive-to-create GPU resources needed by a persistent reduction
 * kernel: a managed-memory command slot and a private GPU stream.  Pooled
 * by dev_id so they can be reused across collectives without paying
 * cudaMallocManaged/hipMallocManaged overhead on every call.
 *
 * GPU components (cuda, rocm) inherit from this base by placing it as the
 * first member named "super" in their own cmd_queue struct, then allocate
 * with OBJ_NEW and return the base pointer.  Destruction (including GPU
 * resource cleanup) is dispatched automatically through the OBJ class chain.
 *
 * session_begin_fn is wired at cmd_queue_alloc time by the component.
 */
typedef struct ompi_op_gpu_cmd_queue_t {
    opal_list_item_t             super;       /* MUST be first: used by opal_lifo_t pool */
    int                          dev_id;
    mca_allocator_base_module_t *allocator;  /* GPU scratch allocator for this device */
    ompi_op_gpu_cmd_t           *cmd;        /* managed memory — shared with GPU */
    /* Session creation hook — wired at cmd_queue_alloc time by the component. */
    struct ompi_op_gpu_session_t *(*session_begin_fn)(
        struct ompi_op_gpu_cmd_queue_t *queue,
        struct ompi_op_t *op,
        struct ompi_datatype_t *dtype);
} ompi_op_gpu_cmd_queue_t;
OBJ_CLASS_DECLARATION(ompi_op_gpu_cmd_queue_t);

/**
 * Per-collective GPU reduction session.  Created by ompi_op_gpu_session_begin()
 * before a collective algorithm's reduction loop, and destroyed (with its
 * cmd_queue recycled to the pool) by ompi_op_gpu_session_end().
 *
 * Sessions are lightweight: all expensive GPU resources (managed memory,
 * GPU stream) live in the cmd_queue, which is pooled separately.  The session
 * holds only a pointer to the cmd_queue and the dispatch function pointers.
 *
 * The component's opc_session_begin wires queue, allocator, reduce_fn, and
 * stop_fn.  Callers must not set these fields directly.
 *
 * When no GPU op component supports the (op, dtype) combination, begin()
 * returns NULL and all callers fall back to ompi_op_reduce().
 *
 * The persistent kernel itself is launched lazily: opc_session_begin only
 * validates that a kernel exists for (op, dtype) and stashes what's needed
 * to launch it in `launcher`; reduce_fn performs the actual launch on the
 * first call, using `started` to track whether that's happened yet. Not
 * every session ends up reducing anything -- e.g. leaf ranks in a
 * reduction tree only ever forward data upward -- so this avoids paying
 * persistent-kernel launch/teardown cost for sessions that turn out to do
 * zero reductions. stop_fn checks `started` and is a no-op if the kernel
 * was never launched.
 */
typedef struct ompi_op_gpu_session_t {
    ompi_op_gpu_cmd_queue_t     *queue;
    mca_allocator_base_module_t *allocator;  /* GPU scratch allocator (= queue->allocator) */
    /* Dispatch hooks wired by the component's opc_session_begin. */
    void (*reduce_fn)(struct ompi_op_gpu_session_t *session,
                      const void *src1, const void *src2, void *dst, size_t count);
    /* Signal the persistent kernel to exit and synchronize the stream.
     * The cmd_queue's resources remain valid for reuse after this call. */
    void (*stop_fn)(struct ompi_op_gpu_session_t *session);
    bool  started;   /* true once reduce_fn has actually launched the persistent kernel */
    void *launcher;  /* component-owned resolved launcher fn pointer for the deferred launch */
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
 * Post one reduction command (src1 op src2 → dst) to the persistent kernel and
 * wait for completion.  src2 may alias dst for in-place operations.
 * Behavior is undefined if session is NULL.
 */
OMPI_DECLSPEC void ompi_op_gpu_session_reduce(ompi_op_gpu_session_t *session,
                                               const void *src1, const void *src2,
                                               void *dst, size_t count);

/**
 * Stop the persistent kernel and return the session's cmd_queue to the pool
 * for reuse.  GPU stream and managed memory remain allocated; a future begin()
 * call for the same dev_id will relaunch the kernel without allocating new
 * resources.  NULL-safe.
 */
OMPI_DECLSPEC void ompi_op_gpu_session_end(ompi_op_gpu_session_t *session);

/**
 * Initialize the cmd_queue pool.  Must be called once before any session
 * operations (from ompi_op_base_open via the framework open hook).
 */
OMPI_DECLSPEC void ompi_op_gpu_session_pool_init(void);

/**
 * Drain and permanently destroy all pooled cmd_queues.  Must be called once
 * during MPI finalization (from ompi_op_base_close).
 */
OMPI_DECLSPEC void ompi_op_gpu_session_pool_finalize(void);

END_C_DECLS

#endif /* OMPI_OP_GPU_SESSION_H */
