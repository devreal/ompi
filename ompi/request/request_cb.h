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
#include "mpi.h"


BEGIN_C_DECLS

#define REQUEST_CB_NONE (NULL)
#define REQUEST_CB_COMPLETED (request_user_callback_t*)(-1ULL)

/**
 * Request callback class
 */
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(request_user_callback_t);

struct request_user_callback_t;

struct request_user_callback_t {
    opal_free_list_item_t super;      /**< Base type */
    MPIX_Request_complete_fn_t *fn;   /**< Optional completion callback provided by the user */
    void *fn_data;                    /**< Optional completion callback data provided by the user */
    opal_atomic_int32_t num_active;   /**< The number of active requests on this callback */
};


typedef struct request_user_callback_t request_user_callback_t;

/* freelist of callback objects */
OMPI_DECLSPEC extern opal_free_list_t request_callback_freelist;

/* the number of unfinished callbacks */
OMPI_DECLSPEC extern opal_atomic_int32_t num_active_callbacks;

/**
 * Initialize the user-callback infrastructure.
 */
int ompi_request_user_callback_init(void);

/**
 * Finalize the user-callback infrastructure.
 */
int ompi_request_user_callback_finalize(void);


/**
 * Enqueue the callback for later processing.
 */
void ompi_request_user_callback_enqueue(request_user_callback_t *cb);

/**
 * Notify the user callback that the request is complete, potentially
 * enqueueing the callback for later processing.
 */
static inline
void ompi_request_user_callback_complete(request_user_callback_t *cb)
{
    int32_t num_active = opal_atomic_sub_fetch_32(&cb->num_active, 1);
    assert(num_active >= 0);
    if (0 == num_active) {
        // we were the last to deregister so enqueue for later processing
        ompi_request_user_callback_enqueue(cb);
    }
}


/**
 * Create and initialize a callback object.
 */
static inline
request_user_callback_t *ompi_request_user_callback_create(
  int                         count,
  MPIX_Request_complete_fn_t  fn,
  void                       *fn_data)
{
    request_user_callback_t *cb;
    cb = (request_user_callback_t *)opal_free_list_get(&request_callback_freelist);
    cb->fn = fn;
    cb->fn_data = fn_data;
    cb->num_active = count;

    OPAL_THREAD_FETCH_ADD32(&num_active_callbacks, 1);

    return cb;
}


/**
 * Process a callback. Returns the callback object to the freelist.
 */
static inline
void ompi_request_user_callback_process(request_user_callback_t *cb)
{
    MPIX_Request_complete_fn_t *fn = cb->fn;
    void *fn_data = cb->fn_data;
    cb->fn      = NULL;
    cb->fn_data = NULL;
    fn(fn_data);

    opal_free_list_return(&request_callback_freelist, &cb->super);

    OPAL_THREAD_FETCH_ADD32(&num_active_callbacks, -1);

    assert(num_active_callbacks >= 0);
}


/**
 * Wait for all active callbacks to complete.
 */
int ompi_request_user_callback_wait(void);

/**
 * Return the number of active request callbacks.
 */
static inline
int ompi_request_user_callback_num_active(void)
{
  return num_active_callbacks;
}


/**
 * Return the number of active request callbacks by the current thread (thread-specific).
 */
uint64_t ompi_request_user_callback_num_processed(void);


END_C_DECLS

#endif // OMPI_REQUEST_CB_H
