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
#include "opal/mca/allocator/allocator.h"

BEGIN_C_DECLS

struct ompi_op_t;
struct ompi_datatype_t;

/**
 * Per-collective GPU reduction session.  Created by ompi_op_gpu_session_begin()
 * before a collective algorithm's reduction loop starts, and destroyed by
 * ompi_op_gpu_session_end() after the loop completes.  When no GPU op
 * component is available or the (op, dtype) combination has no GPU kernel,
 * begin() returns NULL and all callers fall back to ompi_op_reduce().
 */
typedef struct ompi_op_gpu_session_t {
    int                          dev_id;
    mca_allocator_base_module_t *allocator;  /* GPU scratch allocator for this session */
    void                        *backend;    /* opaque: cuda or rocm session state */
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
 * Shut down the persistent kernel, synchronize the GPU stream, and free all
 * session resources.  NULL-safe.
 */
OMPI_DECLSPEC void ompi_op_gpu_session_end(ompi_op_gpu_session_t *session);

END_C_DECLS

#endif /* OMPI_OP_GPU_SESSION_H */
