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
 * Dispatcher and freelist pool for GPU reduction sessions.
 *
 * Sessions are expensive to create: each one allocates managed memory and
 * creates a private GPU stream.  Rather than destroy a session at the end
 * of every collective and recreate it at the start of the next, we keep a
 * flat pool of idle sessions keyed by dev_id.
 *
 * Pool lifecycle:
 *   session_end()  — stops the persistent kernel (GPU stream and managed
 *                    memory remain allocated), then pushes the session onto
 *                    the freelist.
 *   session_begin() — if a matching dev_id entry is found, pops it and calls
 *                    restart_fn(session, op, dtype) to reconfigure and relaunch
 *                    the appropriate kernel; no cudaMalloc / hipMalloc overhead.
 *                    On pool miss, iterates op components to allocate fresh.
 *
 * Pool layout:
 *   session_pool_head — singly-linked freelist, linked through session->pool_next
 *   session_pool_count — current freelist length (global cap = SESSION_POOL_MAX)
 *   session_pool_lock  — single mutex protecting all pool state
 *
 * SESSION_POOL_MAX caps the total number of idle sessions.  Sessions beyond
 * the cap are permanently destroyed rather than pooled to bound GPU resource
 * accumulation.
 */

#include "ompi_config.h"

#include <stdlib.h>

#include "opal/class/opal_list.h"
#include "opal/mca/base/base.h"
#include "opal/mca/threads/mutex.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/op/op.h"

/* Maximum number of idle sessions kept in the pool. */
#define SESSION_POOL_MAX 8

static ompi_op_gpu_session_t *session_pool_head  = NULL;
static int                    session_pool_count  = 0;
static opal_mutex_t           session_pool_lock   = OPAL_MUTEX_STATIC_INIT;

/* --------------------------------------------------------------------------
 * session_destroy — permanently shut down a session and free all resources.
 * Called when the pool is at capacity or at finalization.
 * -------------------------------------------------------------------------- */
static void
session_destroy(ompi_op_gpu_session_t *session)
{
    session->free_fn(session);   /* component frees stream, managed mem, priv */
    free(session);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_begin
 *
 * 1. Walk the pool freelist for a matching dev_id entry.
 * 2. On hit: pop the idle session, call restart_fn to reconfigure for the
 *    new (op, dtype) and relaunch the kernel.  If restart fails (no kernel
 *    for this combination), destroy the session and return NULL.
 * 3. On pool miss: iterate op components to create a new session; wire
 *    dispatch hooks before returning.
 * -------------------------------------------------------------------------- */
ompi_op_gpu_session_t *
ompi_op_gpu_session_begin(struct ompi_op_t *op,
                          struct ompi_datatype_t *dtype,
                          int dev_id)
{
    /* Check pool for a reusable idle session on this device. */
    OPAL_THREAD_LOCK(&session_pool_lock);
    ompi_op_gpu_session_t **pp = &session_pool_head;
    while (NULL != *pp) {
        if ((*pp)->dev_id == dev_id) {
            /* Found a matching idle session — remove from freelist. */
            ompi_op_gpu_session_t *s = *pp;
            *pp = s->pool_next;
            session_pool_count--;
            OPAL_THREAD_UNLOCK(&session_pool_lock);
            s->pool_next = NULL;

            /* Reconfigure the session for the new (op, dtype). */
            if (!s->restart_fn(s, op, dtype)) {
                /* No GPU kernel for this combination; release and return NULL. */
                session_destroy(s);
                return NULL;
            }
            return s;
        }
        pp = &(*pp)->pool_next;
    }
    OPAL_THREAD_UNLOCK(&session_pool_lock);

    /* Pool miss — create a fresh session via the first matching component. */
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

        if (NULL == opc->opc_session_begin   ||
            NULL == opc->opc_session_reduce  ||
            NULL == opc->opc_session_stop    ||
            NULL == opc->opc_session_restart ||
            NULL == opc->opc_session_free) {
            continue;
        }

        ompi_op_gpu_session_t *session = opc->opc_session_begin(op, dtype, dev_id);
        if (NULL == session) {
            continue;
        }

        /* Wire dispatch hooks and pool bookkeeping. */
        session->reduce_fn  = opc->opc_session_reduce;
        session->stop_fn    = opc->opc_session_stop;
        session->restart_fn = opc->opc_session_restart;
        session->free_fn    = opc->opc_session_free;
        session->pool_next  = NULL;
        return session;
    }

    return NULL;
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_reduce
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_reduce(ompi_op_gpu_session_t *session,
                           const void *src, void *dst, size_t count)
{
    session->reduce_fn(session, src, dst, count);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_end
 *
 * Stop the persistent kernel and return the session to the pool so its GPU
 * stream and managed memory can be reused by the next collective on the same
 * device.  If the pool is already at SESSION_POOL_MAX, destroy immediately.
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_end(ompi_op_gpu_session_t *session)
{
    if (NULL == session) {
        return;
    }

    /* Signal the kernel to exit and wait for the stream to drain.
     * GPU stream and managed memory remain allocated for reuse. */
    session->stop_fn(session);

    OPAL_THREAD_LOCK(&session_pool_lock);
    if (session_pool_count < SESSION_POOL_MAX) {
        session->pool_next = session_pool_head;
        session_pool_head  = session;
        session_pool_count++;
        OPAL_THREAD_UNLOCK(&session_pool_lock);
        return;
    }
    OPAL_THREAD_UNLOCK(&session_pool_lock);

    /* Pool full — destroy immediately. */
    session_destroy(session);
}

/* --------------------------------------------------------------------------
 * ompi_op_gpu_session_pool_finalize
 *
 * Drain the pool, release all GPU resources, and free session structs.
 * Called once from ompi_op_base_close() during MPI_Finalize.
 * -------------------------------------------------------------------------- */
void
ompi_op_gpu_session_pool_finalize(void)
{
    OPAL_THREAD_LOCK(&session_pool_lock);
    ompi_op_gpu_session_t *s = session_pool_head;
    session_pool_head  = NULL;
    session_pool_count = 0;
    OPAL_THREAD_UNLOCK(&session_pool_lock);

    while (NULL != s) {
        ompi_op_gpu_session_t *next = s->pool_next;
        session_destroy(s);
        s = next;
    }
}
