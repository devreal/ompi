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
#include "opal/mca/threads/thread_usage.h"


static opal_free_list_t request_callback_freelist;

/**
 * FIFO of completed requests that need the user-defined completion callback
 * invoked. Use atomic push to avoid race conditions if multiple threads
 * complete requests.
 */
static opal_fifo_t request_cont_fifo;

static opal_mutex_t request_cont_lock;

/**
 * Flag indicating whether the progress callback has been registered.
 */
static opal_atomic_int32_t progress_callback_registered;


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

static inline
void ompi_request_cont_destroy(ompi_request_cont_t *cont, ompi_request_t *cont_req)
{
    //ompi_request_t *cont_req = cont->cont_req;
    //printf("Returning cont %p with cont_req %p\n", cont, cont_req);

    opal_atomic_lock(&cont_req->cont_lock);
    //int32_t num_active = opal_atomic_sub_fetch_32(&cont_req->cont_num_active, 1);
    int num_active = --cont_req->cont_num_active;
    assert(num_active >= 0);
    if (0 == num_active) {
        //printf("Completing cont_req %p\n", cont_req);
        assert(!REQUEST_COMPLETE(cont_req));
        opal_atomic_wmb();
        /* signal that all continuations were found complete */
        ompi_request_complete(cont_req, true);
    }
    opal_atomic_unlock(&cont_req->cont_lock);
    OBJ_RELEASE(cont_req);

    cont->cont_data = NULL;
    cont->cont_req  = NULL;
    opal_free_list_return(&request_callback_freelist, &cont->super);
}

/**
 * Process a callback. Returns the callback object to the freelist.
 */
static inline
void ompi_request_cont_invoke(ompi_request_cont_t *cont)
{
    ompi_request_t *cont_req = cont->cont_req;
    assert(NULL != cont_req);
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    MPI_Continue_cb_t *fn = cont_req->cont_cb;
    void *cont_data = cont->cont_data;
    cont->cont_data = NULL;
    cont->cont_req  = NULL;
    fn(cont_data);
    ompi_request_cont_destroy(cont, cont_req);
}

static
int ompi_request_cont_progress_callback()
{
    /*
     * Allow multiple threads to progress callbacks concurrently
     * but protect from recursive progressing
     */
    static opal_thread_local int in_progress = 0;

    if (in_progress || opal_fifo_is_empty(&request_cont_fifo)) return 0;

    int completed = 0;
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
    //printf("ompi_request_cont_finalize \n");
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
    //printf("Request completed for cont %p num_active %d\n", cb, num_active);
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

    /* signal that the continuation request has a new continuation */
    OBJ_RETAIN(cont_req);
    opal_atomic_lock(&cont_req->cont_lock);
    int32_t num_active = cont_req->cont_num_active++;
    if (num_active == 0) {
        /* (re)activate the continuation request upon first registration */
        assert(REQUEST_COMPLETE(cont_req));
        cont_req->cont_obj     = REQUEST_CONT_NONE;
        cont_req->req_complete = REQUEST_PENDING;
    }
    opal_atomic_unlock(&cont_req->cont_lock);

    return cont;
}

static inline void handle_complete_request(ompi_request_t *request)
{
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

int ompi_request_cont_register(
  ompi_request_t             *cont_req,
  const int                   count,
  ompi_request_t             *requests[],
  void                       *cont_data,
  bool                       *all_complete,
  ompi_status_public_t        statuses[])
{
    assert(OMPI_REQUEST_CONT == cont_req->req_type);

    /* Set status objects if required */
    if (MPI_STATUSES_IGNORE != statuses) {
        for (int i = 0; i < count; ++i) {
            ompi_request_t *request = requests[i];
            if (MPI_REQUEST_NULL != request) {
                request->cont_status = &statuses[i];
            }
        }
    }
    *all_complete = false;

    ompi_request_cont_t *cont = ompi_request_cont_create(count, cont_req, cont_data);

    opal_atomic_wmb();

    int32_t num_registered = 0;
    for (int i = 0; i < count; ++i) {
        if (MPI_REQUEST_NULL != requests[i]) {
            ompi_request_t *request = requests[i];
            void *cont_compare = REQUEST_CONT_NONE;
            if (OPAL_ATOMIC_COMPARE_EXCHANGE_STRONG_PTR(&request->cont_obj,
                                                        &cont_compare, cont)) {
                ++num_registered;
            } else {
                assert(REQUEST_CONT_COMPLETED == cont_compare);
                /* set the status, if necessary */
                if (NULL != request->cont_status) {
                    *request->cont_status = request->req_status;
                }
                handle_complete_request(request);
            }
            /* take ownership of any non-persistent request */
            if (!request->req_persistent)
            {
                requests[i] = MPI_REQUEST_NULL;
            }
        }
    }

    int num_complete = count - num_registered;
    int32_t last_num_active = opal_atomic_sub_fetch_32(&cont->num_active,
                                                       num_complete);
    if (0 == last_num_active && 0 < num_complete) {
        /**
         * set flag and return the continuation to the free-list
         * NOTE: only set the flag if *no* continuations have been registered.
         */
        if (0 == num_registered) {
            *all_complete = true;
        }
        ompi_request_cont_destroy(cont, cont_req);
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

int ompi_request_cont_allocate_cont_req(MPI_Continue_cb_t *fn, ompi_request_t **cont_req)
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
        opal_atomic_lock_init(&res->cont_lock, 0);

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
