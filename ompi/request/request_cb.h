/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2020      High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OMPI_REQUEST_CB_H
#define OMPI_REQUEST_CB_H

#include <assert.h>
#include "ompi_config.h"
#include "opal/class/opal_free_list.h"
#include "opal/class/opal_fifo.h"
#include "mpi.h"


BEGIN_C_DECLS

/**
 * Request continuation pointer states
 */
#define REQUEST_CONT_NONE (NULL)
#define REQUEST_CONT_COMPLETED (ompi_request_cont_t*)(-1ULL)

/**
 * Forward declaration
 */
typedef struct ompi_request_t ompi_request_t;

/**
 * Request callback class
 */
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(ompi_request_cont_t);

struct ompi_request_cont_t {
    opal_free_list_item_t super;      /**< Base type */
    ompi_request_t *cont_req;         /**< The continuation request this continuation is registered with */
    MPI_Continue_cb_t *cont_cb;       /**< The callback function to invoke */
    void *cont_data;                  /**< Continuation state provided by the user */
    opal_atomic_int32_t num_active;   /**< The number of active operation requests on this callback */
};

/* Convenience typedef */
typedef struct ompi_request_cont_t ompi_request_cont_t;

extern opal_fifo_t *request_cont_fifo;

/**
 * Initialize the user-callback infrastructure.
 */
int ompi_request_cont_init(void);

/**
 * Finalize the user-callback infrastructure.
 */
int ompi_request_cont_finalize(void);


/**
 * Enqueue a completed continuation for later execution.
 */
void
ompi_request_cont_enqueue_complete(ompi_request_cont_t *cont);

/**
 * Notify the continuation that the request is complete, potentially
 * enqueueing the callback for later invocation.
 */
static inline
void ompi_request_cont_complete_req(ompi_request_cont_t *cb)
{
    int32_t num_active = opal_atomic_sub_fetch_32(&cb->num_active, 1);
    assert(num_active >= 0);
    if (0 == num_active) {
        // we were the last to deregister so enqueue for later processing
        ompi_request_cont_enqueue_complete(cb);
    }
}

/**
 * Register a continuation for a set of operations represented by \c requests.
 * If all operations have completed already the continuation will not be registered
 * and \c all_complete will be set to 1.
 */
int ompi_request_cont_register(
  ompi_request_t             *cont_req,
  int                         count,
  ompi_request_t             *requests[],
  MPI_Continue_cb_t          *cont_cb,
  void                       *cont_data,
  bool                       *all_complete,
  ompi_status_public_t        statuses[]);


/**
 * Allocate a new (presistent & transient) continuation request.
 */
int ompi_request_cont_allocate_cont_req(ompi_request_t **cont_req);


/**
 * Progress outstanding ready continuations.
 */
int ompi_request_cont_progress_callback();


/**
 * Progress completed requests whose user-callback are pending.
 */
static inline
int ompi_request_cont_progress_ready()
{
    /* fast-path */
    if (opal_fifo_is_empty(request_cont_fifo)) return OMPI_SUCCESS;

    return ompi_request_cont_progress_callback();
}

END_C_DECLS

#endif // OMPI_REQUEST_CB_H
