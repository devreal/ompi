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

#include "ompi/op/op_gpu_session.h"
#include "ompi/op/op.h"

ompi_op_gpu_session_t *
ompi_op_gpu_session_begin(struct ompi_op_t *op,
                          struct ompi_datatype_t *dtype,
                          int dev_id)
{
    /* Phase 1 stub: no GPU op components yet.  Always return NULL so that
     * all callers use the host ompi_op_reduce path. */
    (void) op;
    (void) dtype;
    (void) dev_id;
    return NULL;
}

void
ompi_op_gpu_session_reduce(ompi_op_gpu_session_t *session,
                           const void *src, void *dst, size_t count)
{
    /* Must not be called when session is NULL */
    (void) session;
    (void) src;
    (void) dst;
    (void) count;
}

void
ompi_op_gpu_session_end(ompi_op_gpu_session_t *session)
{
    /* NULL-safe no-op in Phase 1 */
    (void) session;
}
