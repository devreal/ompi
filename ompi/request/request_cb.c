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


#include "ompi_config.h"
#include "request.h"
#include "request_cb.h"
#include "opal/class/opal_fifo.h"
#include "opal/sys/atomic.h"
#include "opal/threads/thread_usage.h"


static opal_free_list_t request_callback_freelist;

/**
 * FIFO of completed requests that need the user-defined completion callback
 * invoked. Use atomic push to avoid race conditions if multiple threads
 * complete requests.
 */
static opal_fifo_t request_cont_fifo;

static opal_mutex_t request_cont_lock;

/*
 * Allow multiple threads to progress callbacks concurrently
 * but protect from recursive progressing
 */
static opal_thread_local int in_progress = 0;

/**
 * Flag indicating whether the progress callback has been registered.
 */
static opal_atomic_int32_t progress_callback_registered;

opal_atomic_int32_t num_active_callbacks = 0; /**< the number of unfinished callbacks */



static void ompi_request_cont_construct(ompi_request_cont_t* cont)
{
    cont->cont_req = NULL;
    cont->cont_data = NULL;
    cont->num_active = 0;
}

static void ompi_request_cont_destruct(ompi_request_cont_t* cont)
{
    assert(cont->cont_req == NULL);
    assert(cont->cont_data == NULL);
    assert(cont->num_active == 0);
}


OBJ_CLASS_INSTANCE(
    ompi_request_cont_t,
    opal_free_list_item_t,
    ompi_request_cont_construct,
    ompi_request_cont_destruct);

/**
 * Process a callback. Returns the callback object to the freelist.
 */
static inline
void ompi_request_cont_invoke(ompi_request_cont_t *cont)
{
    ompi_request_t *cont_req = cont->cont_req;
    assert(NULL != cont_req);
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    MPIX_Continue_cb_t *fn = cont_req->cont_cb;
    void *cont_data = cont->cont_data;
    cont->cont_data = NULL;
    cont->cont_req  = NULL;
    fn(cont_data);

    opal_free_list_return(&request_callback_freelist, &cont->super);

    int32_t num_active = OPAL_THREAD_FETCH_ADD32(&cont_req->cont_num_active, -1);
    assert(num_active >= 0);
    if (0 == (num_active-1)) {
        /* signal that all continuations were found complete */
        ompi_request_complete(cont_req, true);
    }
    OBJ_RELEASE(cont_req);
}

static
int ompi_request_cont_progress_callback()
{
    int completed = 0;
    if (opal_fifo_is_empty(&request_cont_fifo) || in_progress) return 0;

    in_progress = 1;

    do {
        ompi_request_cont_t *cb;
        OPAL_THREAD_LOCK(&request_cont_lock);
        cb = (ompi_request_cont_t*)opal_fifo_pop_st(&request_cont_fifo);
        OPAL_THREAD_UNLOCK(&request_cont_lock);
        if (NULL == cb) break;
        ompi_request_cont_invoke(cb);
        completed++;
    } while (1);

    in_progress = 0;

    return completed;
}

int ompi_request_cont_init(void)
{
    progress_callback_registered = 0;

    OBJ_CONSTRUCT(&request_cont_lock, opal_mutex_t);
    OBJ_CONSTRUCT(&request_cont_fifo, opal_fifo_t);

    OBJ_CONSTRUCT(&request_callback_freelist, opal_free_list_t);
    opal_free_list_init(&request_callback_freelist,
                        sizeof(ompi_request_cont_t),
                        opal_cache_line_size,
                        OBJ_CLASS(ompi_request_cont_t),
                        0, opal_cache_line_size,
                        0, -1 , 8, NULL, 0, NULL, NULL, NULL);

    return OMPI_SUCCESS;
}

int ompi_request_cont_finalize(void)
{
    if (progress_callback_registered) {
        opal_progress_unregister(&ompi_request_cont_progress_callback);
    }
    OBJ_DESTRUCT(&request_cont_fifo);
    OBJ_DESTRUCT(&request_cont_lock);
    OBJ_DESTRUCT(&request_callback_freelist);

    return OMPI_SUCCESS;
}

/**
 * Enqueue the continuation for later invocation.
 */
static void
ompi_request_cont_enqueue_complete(ompi_request_cont_t *cont)
{
    OPAL_THREAD_LOCK(&request_cont_lock);
    opal_fifo_push_st(&request_cont_fifo, &cont->super.super);
    if (OPAL_UNLIKELY(!progress_callback_registered)) {
        opal_progress_register_post(&ompi_request_cont_progress_callback);
        progress_callback_registered = true;
    }
    OPAL_THREAD_UNLOCK(&request_cont_lock);
}

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
 * Create and initialize a continuation object.
 */
static inline
ompi_request_cont_t *ompi_request_cont_create(
  int                         count,
  ompi_request_t             *cont_req,
  void                       *cont_data)
{
    ompi_request_cont_t *cont;
    cont = (ompi_request_cont_t *)opal_free_list_get(&request_callback_freelist);
    cont->cont_req  = cont_req;
    cont->cont_data = cont_data;
    cont->num_active = count;

    OBJ_RETAIN(cont_req);
    int32_t num_active = OPAL_THREAD_FETCH_ADD32(&cont_req->cont_num_active, 1);
    if (num_active == 0) {
        /* (re)activate the continuation request upon first registration */
        cont_req->cont_obj     = REQUEST_CONT_NONE;
        cont_req->req_complete = REQUEST_PENDING;
    }

    return cont;
}


int ompi_request_cont_register(
  ompi_request_t             *cont_req,
  int                         count,
  ompi_request_t             *requests[],
  void                       *cont_data,
  ompi_status_public_t        statuses[])
{
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    ompi_request_cont_t *cb = ompi_request_cont_create(count, cont_req, cont_data);

    // set the status field in each request here to avoid memory barriers
    if (MPI_STATUSES_IGNORE != statuses) {
        for (int i = 0; i < count; ++i) {
            if (MPI_REQUEST_NULL != requests[i]) {
                requests[i]->cont_status = &statuses[i];
            }
        }
    }

    opal_atomic_wmb();

    int32_t num_registered = 0;
    for (int i = 0; i < count; ++i) {
        if (MPI_REQUEST_NULL != requests[i]) {
            ompi_request_t *request = requests[i];
            void *cont_compare = REQUEST_CONT_NONE;
            if (OPAL_ATOMIC_COMPARE_EXCHANGE_STRONG_PTR(&request->cont_obj,
                                                        &cont_compare, cb)) {
                ++num_registered;
            } else if (cont_compare == REQUEST_CONT_COMPLETED) {
                /* set the status, if necessary */
                if (NULL != request->cont_status) {
                    *request->cont_status = request->req_status;
                }
                /* inactivate / free the request */
                if (request->req_transient) {
                    /* nothing to do here */
                } else if (request->req_persistent) {
                    request->req_state = OMPI_REQUEST_INACTIVE;
                } else {
                    /* the request is complete, release the request object */
                    ompi_request_free(&request);
                }
            }
            /* take ownership of any non-persistent request */
            if (!request->req_persistent)
            {
                requests[i] = MPI_REQUEST_NULL;
            }
        }
    }

    int32_t last_num_active = opal_atomic_sub_fetch_32(&cb->num_active, count - num_registered);

    if (0 == last_num_active) {
        // all requests were complete and we were the last to decrement
        // so invoke directly and clean up
        ompi_request_cont_invoke(cb);
    }

    return OMPI_SUCCESS;
}

/**
 * Continuation request management
 */

static int ompi_request_cont_req_free(ompi_request_t** cont_req)
{
    OMPI_REQUEST_FINI(*cont_req);
    (*cont_req)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*cont_req);
    *cont_req = &ompi_request_null.request;
    return OMPI_SUCCESS;
}

int ompi_request_cont_allocate_cont_req(MPIX_Continue_cb_t *fn, ompi_request_t **cont_req)
{
    ompi_request_t *res = OBJ_NEW(ompi_request_t);

    if (OPAL_LIKELY(NULL != cont_req)) {
        res->req_type = OMPI_REQUEST_CONT;
        res->req_complete = REQUEST_COMPLETED;
        res->req_state = OMPI_REQUEST_ACTIVE;
        res->req_persistent = true;
        res->req_transient  = true;
        res->req_free = ompi_request_cont_req_free;
        res->req_status = ompi_status_empty; /* always returns MPI_SUCCESS */
        res->cont_cb = fn;

        *cont_req = res;

        return MPI_SUCCESS;
    }

    return OMPI_ERR_OUT_OF_RESOURCE;
}


int ompi_request_cont_progress_ready()
{
    ompi_request_cont_progress_callback();
    return OMPI_SUCCESS;
}
