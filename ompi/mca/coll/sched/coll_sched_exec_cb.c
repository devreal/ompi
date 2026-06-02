/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Callback-based (continuation) executor for the coll/sched schedule IR.
 *
 * Unlike the BSP PML executor (coll_sched_exec_pml.c) which blocks inside a
 * per-step ompi_request_wait_all loop, this executor attaches a completion
 * callback to every PML send/recv request.  The last callback to fire in a
 * step executes the step's local ops and immediately posts the next step's
 * network ops — all without returning to the polling loop.  Step transitions
 * therefore happen inside ompi_request_complete() (potentially from a progress
 * thread), eliminating the per-step polling lag.
 *
 * The blocking execute() wrapper still returns only after the full collective
 * completes; it waits on an internal ompi_coll_base_nbc_request_t that the
 * final callback marks done.
 */

#include "ompi_config.h"
#include <stdlib.h>
#include <string.h>
#include "coll_sched.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/request/request.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/include/opal_stdatomic.h"

/* ── Per-invocation execution context ──────────────────────────────────────
 *
 * Heap-allocated by cb_execute(); freed by cb_execute() after the blocking
 * wait returns.  Callbacks hold a non-owning pointer to it; they must never
 * free it.
 */
typedef struct {
    /* Immutable call parameters */
    const ompi_coll_sched_t    *sched;
    struct ompi_communicator_t **comms;
    const void                 *sbuf;
    void                       *rbuf;
    size_t                      count;
    struct ompi_datatype_t     *dtype;
    struct ompi_op_t           *op;
    int                         base_tag;
    int                         n;       /* ompi_comm_size(comms[0]) */
    ptrdiff_t                   extent;

    /* Embedded copy of the comms[] array.  Non-blocking invocations store the
     * communicators here so ctx->comms remains valid after iexecute() returns
     * (the caller's stack-allocated comms[] would otherwise be freed). */
    struct ompi_communicator_t *comms_buf[4];

    /* Temp buffers (same layout as PML executor) */
    char *temp_raw[OMPI_COLL_SCHED_MAX_TEMP_BUFS];
    char *temp_ptrs[OMPI_COLL_SCHED_MAX_TEMP_BUFS];

    /* Current step — only written by the winning callback (pending → 0) */
    int                         step;
    int                         rc;

    /* Countdown of outstanding network ops in the current step.
     * Multiple callbacks may decrement concurrently; the one that
     * reaches 0 advances to the next step. */
    opal_atomic_int32_t         pending;

    /* Completion signal; marked done by the final advance() call */
    ompi_request_t             *comp_req;
} ompi_coll_sched_exec_cb_ctx_t;

/* ── Buffer reference resolver ──────────────────────────────────────────────
 * (identical logic to coll_sched_exec_pml.c)
 */
static char *
resolve_bufref(const ompi_coll_sched_bufref_t *ref,
               const void *sbuf, void *rbuf,
               char **temp_ptrs,
               size_t count, ptrdiff_t extent,
               int n,
               size_t *out_count)
{
    size_t unit         = (count + (size_t) ref->num_slots - 1) / (size_t) ref->num_slots;
    size_t offset_elems = (size_t) ref->slot * unit;

    const char *base;
    switch (ref->buf_id) {
    case OMPI_COLL_SCHED_BUF_SEND: base = (const char *) sbuf; break;
    case OMPI_COLL_SCHED_BUF_RECV: base = (const char *) rbuf; break;
    default:                        base = temp_ptrs[ref->buf_id - 2]; break;
    }

    if (offset_elems >= count) {
        *out_count = 0;
        return (char *) base;
    }
    size_t remaining = count - offset_elems;
    *out_count = unit < remaining ? unit : remaining;
    return (char *) base + offset_elems * extent;
}

/* ── Forward declarations ───────────────────────────────────────────────── */
static void start_step(ompi_coll_sched_exec_cb_ctx_t *ctx);
static void advance(ompi_coll_sched_exec_cb_ctx_t *ctx);

/* ── Local op execution ─────────────────────────────────────────────────── */

static void
execute_local_ops(ompi_coll_sched_exec_cb_ctx_t *ctx)
{
    const ompi_coll_sched_step_t *step = &ctx->sched->steps[ctx->step];

    for (int o = 0; o < step->num_ops; o++) {
        const ompi_coll_sched_op_t *entry = &step->ops[o];

        if (entry->type == OMPI_COLL_SCHED_OP_REDUCE) {
            size_t src_n, dst_n;
            char *src = resolve_bufref(&entry->reduce.src,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &src_n);
            char *dst = resolve_bufref(&entry->reduce.dst,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &dst_n);
            size_t op_count = src_n < dst_n ? src_n : dst_n;
            if (op_count > 0) {
                ompi_op_reduce(ctx->op, src, dst, op_count, ctx->dtype);
            }

        } else if (entry->type == OMPI_COLL_SCHED_OP_REDUCE3) {
            size_t s1_n, s2_n, d_n;
            char *src1 = resolve_bufref(&entry->reduce3.src1,
                                         ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                         ctx->count, ctx->extent, ctx->n, &s1_n);
            char *src2 = resolve_bufref(&entry->reduce3.src2,
                                         ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                         ctx->count, ctx->extent, ctx->n, &s2_n);
            char *dst  = resolve_bufref(&entry->reduce3.dst,
                                         ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                         ctx->count, ctx->extent, ctx->n, &d_n);
            size_t op_count = s1_n < s2_n ? s1_n : s2_n;
            if (op_count > 0) {
                ompi_3buff_op_reduce(ctx->op, src1, src2, dst, op_count, ctx->dtype);
            }

        } else if (entry->type == OMPI_COLL_SCHED_OP_COPY) {
            size_t src_n, dst_n;
            char *src = resolve_bufref(&entry->copy.src,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &src_n);
            char *dst = resolve_bufref(&entry->copy.dst,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &dst_n);
            size_t op_count = src_n < dst_n ? src_n : dst_n;
            if (op_count > 0) {
                ompi_datatype_copy_content_same_ddt(ctx->dtype, op_count, dst, src);
            }
        }
        /* SEND and RECV ops are handled by start_step's network loop */
    }
}

/* ── Network op callback ────────────────────────────────────────────────── */

static int
net_cb(ompi_request_t *req)
{
    ompi_coll_sched_exec_cb_ctx_t *ctx =
        (ompi_coll_sched_exec_cb_ctx_t *) req->req_complete_cb_data;

    /* Release the PML request immediately; we have no further use for it */
    req->req_free(&req);

    /* Decrement pending count; only the last completion advances the state */
    if (opal_atomic_sub_fetch_32(&ctx->pending, 1) != 0) {
        return 1; /* not the last in this step */
    }

    /* We are the last: execute local ops then start the next step */
    execute_local_ops(ctx);
    advance(ctx);

    return 1; /* 1 = callback took ownership of the PML request */
}

/* ── Step starter ───────────────────────────────────────────────────────── */

static void
start_step(ompi_coll_sched_exec_cb_ctx_t *ctx)
{
    const ompi_coll_sched_step_t *step = &ctx->sched->steps[ctx->step];

    /* Count net ops so we can initialise pending before posting any request.
     * (Posting before setting pending would allow callbacks to fire and see
     *  pending==0 before we finish posting.) */
    int32_t net_count = 0;
    for (int o = 0; o < step->num_ops; o++) {
        ompi_coll_sched_optype_t t = step->ops[o].type;
        if (t == OMPI_COLL_SCHED_OP_SEND || t == OMPI_COLL_SCHED_OP_RECV) {
            net_count++;
        }
    }

    if (net_count == 0) {
        /* Local-only step: run ops inline and advance without a callback */
        execute_local_ops(ctx);
        advance(ctx);
        return;
    }

    /* Initialise pending BEFORE posting any request (see above) */
    ctx->pending = net_count;

    /* Post all network ops with callbacks */
    for (int o = 0; o < step->num_ops; o++) {
        const ompi_coll_sched_op_t *entry = &step->ops[o];

        if (entry->type == OMPI_COLL_SCHED_OP_SEND) {
            size_t op_count;
            char *ptr = resolve_bufref(&entry->send.buf,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &op_count);
            ompi_request_t *req;
            int rc = MCA_PML_CALL(isend(ptr, op_count, ctx->dtype,
                                        entry->send.peer,
                                        ctx->base_tag,
                                        MCA_PML_BASE_SEND_STANDARD,
                                        ctx->comms[entry->comm_slot],
                                        &req));
            if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
                ctx->rc = rc;
            }
            ompi_request_set_callback(req, net_cb, ctx);

        } else if (entry->type == OMPI_COLL_SCHED_OP_RECV) {
            size_t op_count;
            char *ptr = resolve_bufref(&entry->recv.buf,
                                        ctx->sbuf, ctx->rbuf, ctx->temp_ptrs,
                                        ctx->count, ctx->extent, ctx->n, &op_count);
            ompi_request_t *req;
            int rc = MCA_PML_CALL(irecv(ptr, op_count, ctx->dtype,
                                         entry->recv.peer,
                                         ctx->base_tag,
                                         ctx->comms[entry->comm_slot],
                                         &req));
            if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
                ctx->rc = rc;
            }
            ompi_request_set_callback(req, net_cb, ctx);
        }
    }
}

/* ── State machine advance ──────────────────────────────────────────────── */

static void
advance(ompi_coll_sched_exec_cb_ctx_t *ctx)
{
    ctx->step++;
    if (ctx->step < ctx->sched->num_steps) {
        start_step(ctx);
    } else {
        /* All steps complete: wake the blocking execute() */
        ompi_request_complete(ctx->comp_req, 1);
    }
}

/* ── Completion request free callbacks ──────────────────────────────────── */

/* Blocking case: ctx and temp bufs are owned by cb_execute, not the request. */
static int
cb_request_free(ompi_request_t **req)
{
    OMPI_REQUEST_FINI(*req);
    (*req)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*req);
    *req = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

/* Non-blocking case: ctx (with temp bufs) is owned by the request.
 * The dispatch may also attach extra allocations via
 * ompi_coll_base_append_array_to_release() (e.g. work_buf_raw for reduce
 * non-root ranks); we free those here too since we bypass the standard
 * free_objs_callback. */
static int
cb_nbc_request_free(ompi_request_t **rptr)
{
    ompi_coll_base_nbc_request_t *cr = (ompi_coll_base_nbc_request_t *) *rptr;
    ompi_coll_sched_exec_cb_ctx_t *ctx =
        (ompi_coll_sched_exec_cb_ctx_t *) cr->req_complete_cb_data;

    if (NULL != ctx) {
        for (int t = 0; t < ctx->sched->num_temp_bufs; t++) {
            free(ctx->temp_raw[t]);
        }
        free(ctx);
        cr->req_complete_cb_data = NULL;
    }

    /* Free any arrays attached by the dispatch (e.g. reduce work_buf_raw). */
    for (int i = 0; i < OMPI_REQ_NB_RELEASE_ARRAYS; i++) {
        if (NULL == cr->data.release_arrays[i]) {
            break;
        }
        free(cr->data.release_arrays[i]);
        cr->data.release_arrays[i] = NULL;
    }

    OMPI_REQUEST_FINI(*rptr);
    (*rptr)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*rptr);
    *rptr = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

/* ── Executor entry points ──────────────────────────────────────────────── */

static int
cb_execute(ompi_coll_sched_exec_t *exec,
           const ompi_coll_sched_t *sched,
           struct ompi_communicator_t **comms,
           const void *sbuf, void *rbuf,
           size_t count,
           struct ompi_datatype_t *dtype,
           struct ompi_op_t *op,
           int base_tag)
{
    /* ── Allocate execution context ─────────────────────────────────────── */

    ompi_coll_sched_exec_cb_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (NULL == ctx) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    ctx->sched    = sched;
    ctx->comms    = comms;
    ctx->sbuf     = sbuf;
    ctx->rbuf     = rbuf;
    ctx->count    = count;
    ctx->dtype    = dtype;
    ctx->op       = op;
    ctx->base_tag = base_tag;
    ctx->n        = ompi_comm_size(comms[0]);
    ctx->step     = 0;
    ctx->rc       = OMPI_SUCCESS;

    ptrdiff_t lb;
    ompi_datatype_get_extent(dtype, &lb, &ctx->extent);

    /* ── Allocate temp buffers ──────────────────────────────────────────── */

    for (int t = 0; t < sched->num_temp_bufs; t++) {
        size_t temp_count = sched->temp_full_size[t]
                            ? count
                            : (count + (size_t) ctx->n - 1) / (size_t) ctx->n;
        ptrdiff_t gap;
        ptrdiff_t span = opal_datatype_span(&dtype->super, temp_count, &gap);
        ctx->temp_raw[t] = (char *) malloc(span);
        if (NULL == ctx->temp_raw[t]) {
            for (int j = 0; j < t; j++) {
                free(ctx->temp_raw[j]);
            }
            free(ctx);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        ctx->temp_ptrs[t] = ctx->temp_raw[t] - gap;
    }

    /* ── Create completion request ──────────────────────────────────────── */

    ompi_coll_base_nbc_request_t *cr = OBJ_NEW(ompi_coll_base_nbc_request_t);
    if (NULL == cr) {
        for (int t = 0; t < sched->num_temp_bufs; t++) {
            free(ctx->temp_raw[t]);
        }
        free(ctx);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    OMPI_REQUEST_INIT(&cr->super, false);
    cr->super.req_state = OMPI_REQUEST_ACTIVE;
    cr->super.req_type  = OMPI_REQUEST_COLL;
    cr->super.req_free  = cb_request_free;
    ctx->comp_req = &cr->super;

    /* ── Handle empty schedule ──────────────────────────────────────────── */

    if (sched->num_steps == 0) {
        ompi_request_complete(ctx->comp_req, 1);
    } else {
        /* ── Start step 0: posts ops and chains callbacks ─────────────── */
        start_step(ctx);
    }

    /* ── Block until the final callback marks completion ────────────────── */

    ompi_request_t *req_ptr = ctx->comp_req;
    ompi_request_wait(&req_ptr, MPI_STATUS_IGNORE);
    /* req_ptr may now be MPI_REQUEST_NULL (freed by ompi_request_wait) */

    /* ── Cleanup ────────────────────────────────────────────────────────── */

    int rc = ctx->rc;
    for (int t = 0; t < sched->num_temp_bufs; t++) {
        free(ctx->temp_raw[t]);
    }
    free(ctx);
    return rc;
}

/* ── Non-blocking executor entry point ──────────────────────────────────── */

/* Starts the schedule and returns a user-visible request without blocking.
 * Resources (ctx, temp bufs) are freed in cb_nbc_request_free when the user
 * eventually releases the request (after MPI_Wait / MPI_Test).
 * The dispatch may additionally attach extra allocations to the request via
 * ompi_coll_base_append_array_to_release(); cb_nbc_request_free handles them. */
static int
cb_iexecute(ompi_coll_sched_exec_t *exec,
            const ompi_coll_sched_t *sched,
            struct ompi_communicator_t **comms,
            const void *sbuf, void *rbuf,
            size_t count,
            struct ompi_datatype_t *dtype,
            struct ompi_op_t *op,
            int base_tag,
            ompi_request_t **request)
{
    ompi_coll_sched_exec_cb_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (NULL == ctx) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* Copy comms into ctx->comms_buf so the pointer remains valid after this
     * function returns (the caller's stack-allocated comms[] is gone by then). */
    ctx->n = ompi_comm_size(comms[0]);
    for (int i = 0; i < sched->num_comm_slots && i < 4; i++) {
        ctx->comms_buf[i] = comms[i];
    }
    ctx->comms    = ctx->comms_buf;
    ctx->sched    = sched;
    ctx->sbuf     = sbuf;
    ctx->rbuf     = rbuf;
    ctx->count    = count;
    ctx->dtype    = dtype;
    ctx->op       = op;
    ctx->base_tag = base_tag;
    ctx->step     = 0;
    ctx->rc       = OMPI_SUCCESS;

    ptrdiff_t lb;
    ompi_datatype_get_extent(dtype, &lb, &ctx->extent);

    for (int t = 0; t < sched->num_temp_bufs; t++) {
        size_t temp_count = sched->temp_full_size[t]
                            ? count
                            : (count + (size_t) ctx->n - 1) / (size_t) ctx->n;
        ptrdiff_t gap;
        ptrdiff_t span = opal_datatype_span(&dtype->super, temp_count, &gap);
        ctx->temp_raw[t] = (char *) malloc(span);
        if (NULL == ctx->temp_raw[t]) {
            for (int j = 0; j < t; j++) {
                free(ctx->temp_raw[j]);
            }
            free(ctx);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        ctx->temp_ptrs[t] = ctx->temp_raw[t] - gap;
    }

    ompi_coll_base_nbc_request_t *cr = OBJ_NEW(ompi_coll_base_nbc_request_t);
    if (NULL == cr) {
        for (int t = 0; t < sched->num_temp_bufs; t++) {
            free(ctx->temp_raw[t]);
        }
        free(ctx);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    OMPI_REQUEST_INIT(&cr->super, false);
    cr->super.req_state = OMPI_REQUEST_ACTIVE;
    cr->super.req_type  = OMPI_REQUEST_COLL;
    cr->super.req_free  = cb_nbc_request_free;
    cr->req_complete_cb_data = ctx;   /* ctx freed by cb_nbc_request_free */
    ctx->comp_req = &cr->super;

    *request = &cr->super;

    if (sched->num_steps == 0) {
        ompi_request_complete(ctx->comp_req, 1);
    } else {
        start_step(ctx);
    }

    return OMPI_SUCCESS;
}

static bool
cb_can_execute(ompi_coll_sched_exec_t *exec,
               const ompi_coll_sched_t *sched,
               struct ompi_communicator_t **comms,
               struct ompi_datatype_t *dtype,
               struct ompi_op_t *op)
{
    /* Can execute any schedule */
    return true;
}

static void
cb_free(ompi_coll_sched_exec_t *exec)
{
    free(exec);
}

/* ── Constructor ────────────────────────────────────────────────────────── */

typedef struct {
    ompi_coll_sched_exec_t base;
    /* No per-communicator state needed */
} ompi_coll_sched_exec_cb_t;

ompi_coll_sched_exec_t *
ompi_coll_sched_exec_cb_create(void)
{
    ompi_coll_sched_exec_cb_t *e = calloc(1, sizeof(*e));
    if (NULL == e) {
        return NULL;
    }
    e->base.can_execute = cb_can_execute;
    e->base.execute     = cb_execute;
    e->base.iexecute    = cb_iexecute;
    e->base.free        = cb_free;
    return &e->base;
}
