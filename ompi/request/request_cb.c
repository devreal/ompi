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


opal_free_list_t request_callback_freelist;

/**
 * FIFO of completed requests that need the user-defined completion callback
 * invoked. Use atomic push to avoid race conditions if multiple threads
 * complete requests.
 */
static opal_fifo_t request_callback_fifo;

static opal_mutex_t request_callback_lock;

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


/* the number of processed request callbacks of the current thread, monotonically increasing */
static opal_thread_local uint64_t request_callback_num_processed = 0;

/**
 * Progress completed requests whose user-callback are pending.
 */
int ompi_request_progress_user_completion();

static void ompi_request_user_callback_construct(request_user_callback_t* cb)
{
    cb->fn = NULL;
    cb->fn_data = NULL;
    cb->num_active = 0;
}

static void ompi_request_user_callback_destruct(request_user_callback_t* cb)
{
    assert(cb->fn == NULL);
    assert(cb->fn_data == NULL);
    assert(cb->num_active == 0);
}


OBJ_CLASS_INSTANCE(
    request_user_callback_t,
    opal_free_list_item_t,
    ompi_request_user_callback_construct,
    ompi_request_user_callback_destruct);


int ompi_request_user_callback_init(void)
{
    progress_callback_registered = 0;

    OBJ_CONSTRUCT(&request_callback_lock, opal_mutex_t);
    OBJ_CONSTRUCT(&request_callback_fifo, opal_fifo_t);

    OBJ_CONSTRUCT(&request_callback_freelist, opal_free_list_t);
    opal_free_list_init(&request_callback_freelist,
                        sizeof(request_user_callback_t),
                        opal_cache_line_size,
                        OBJ_CLASS(request_user_callback_t),
                        0, opal_cache_line_size,
                        0, -1 , 8, NULL, 0, NULL, NULL, NULL);

    return OMPI_SUCCESS;
}

int ompi_request_user_callback_finalize(void)
{
    if (progress_callback_registered) {
        opal_progress_unregister(&ompi_request_progress_user_completion);
    }
    OBJ_DESTRUCT(&request_callback_fifo);
    OBJ_DESTRUCT(&request_callback_lock);
    OBJ_DESTRUCT(&request_callback_freelist);

    return OMPI_SUCCESS;
}

int ompi_request_progress_user_completion()
{
    int completed = 0;
    if (opal_fifo_is_empty(&request_callback_fifo) || in_progress) return 0;

    in_progress = 1;

    do {
        request_user_callback_t *cb;
        OPAL_THREAD_LOCK(&request_callback_lock);
        cb = (request_user_callback_t*)opal_fifo_pop_st(&request_callback_fifo);
        OPAL_THREAD_UNLOCK(&request_callback_lock);
        if (NULL == cb) break;
        ompi_request_user_callback_process(cb);
        completed++;
    } while (1);

    in_progress = 0;

    request_callback_num_processed += completed;

    return completed;
}

void
ompi_request_user_callback_enqueue(request_user_callback_t *cb)
{
    OPAL_THREAD_LOCK(&request_callback_lock);
    opal_fifo_push_st(&request_callback_fifo, &cb->super.super);
    if (OPAL_UNLIKELY(!progress_callback_registered)) {
        opal_progress_register_post(&ompi_request_progress_user_completion);
        progress_callback_registered = true;
    }
    OPAL_THREAD_UNLOCK(&request_callback_lock);
}


int ompi_request_user_callback_wait(void)
{
    ompi_request_progress_user_completion();
    while (num_active_callbacks > 0) {
        opal_progress();
    }

    return MPI_SUCCESS;
}


uint64_t ompi_request_user_callback_num_processed(void)
{
  return request_callback_num_processed;
}
