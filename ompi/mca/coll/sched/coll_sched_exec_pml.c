/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include <stdlib.h>
#include <string.h>
#include "coll_sched.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_datatype.h"

typedef struct {
    ompi_coll_sched_exec_t base;
    /* PML executor carries no per-communicator state */
} ompi_coll_sched_exec_pml_t;

/* ── Buffer reference resolution ─────────────────────────────────────────
 *
 * Returns a pointer to the start of the region and fills *out_count with
 * the number of elements in it.
 *
 * For TEMP bufs num_slots is always 1 (whole temp buffer).
 * For SEND/RECV bufs the region is slot [ref->slot] of [ref->num_slots]
 * equal partitions of count elements.
 */
static char *
resolve_bufref(const ompi_coll_sched_bufref_t *ref,
               const void *sbuf, void *rbuf,
               char **temp_ptrs,
               size_t count, ptrdiff_t extent,
               int n,
               size_t *out_count)
{
    size_t unit = (count + (size_t) ref->num_slots - 1) / (size_t) ref->num_slots;
    size_t offset_elems = (size_t) ref->slot * unit;

    const char *base;
    switch (ref->buf_id) {
    case OMPI_COLL_SCHED_BUF_SEND:
        base = (const char *) sbuf;
        break;
    case OMPI_COLL_SCHED_BUF_RECV:
        base = (const char *) rbuf;
        break;
    default: /* TEMP */
        base = temp_ptrs[ref->buf_id - 2];
        break;
    }

    /* Slot is entirely out of range: return base with count=0.
     * The caller will still post a 0-count send/recv (required for matching
     * when peer posts the opposite side) but MPI won't touch the pointer. */
    if (offset_elems >= count) {
        *out_count = 0;
        return (char *) base;
    }

    size_t remaining = count - offset_elems;
    *out_count = unit < remaining ? unit : remaining;
    return (char *) base + offset_elems * extent;
}

/* ── Execute ─────────────────────────────────────────────────────────────*/

static int
pml_execute(ompi_coll_sched_exec_t *exec,
            const ompi_coll_sched_t *sched,
            struct ompi_communicator_t **comms,
            const void *sbuf, void *rbuf,
            size_t count,
            struct ompi_datatype_t *dtype,
            struct ompi_op_t *op,
            int base_tag)
{
    int n  = ompi_comm_size(comms[0]);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);

    /* ── Allocate temp buffers ──────────────────────────────────────────── */

    char *temp_raw[OMPI_COLL_SCHED_MAX_TEMP_BUFS]  = {NULL};
    char *temp_ptrs[OMPI_COLL_SCHED_MAX_TEMP_BUFS] = {NULL};

    for (int t = 0; t < sched->num_temp_bufs; t++) {
        size_t temp_count = sched->temp_full_size[t]
                            ? count
                            : (count + (size_t) n - 1) / (size_t) n;
        ptrdiff_t gap;
        ptrdiff_t span = opal_datatype_span(&dtype->super, temp_count, &gap);
        temp_raw[t] = (char *) malloc(span);
        if (NULL == temp_raw[t]) {
            for (int j = 0; j < t; j++) {
                free(temp_raw[j]);
            }
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        temp_ptrs[t] = temp_raw[t] - gap;
    }

    /* ── Find maximum number of concurrent network ops in any step ───── */

    int max_net = 0;
    for (int s = 0; s < sched->num_steps; s++) {
        int cnt = 0;
        for (int o = 0; o < sched->steps[s].num_ops; o++) {
            ompi_coll_sched_optype_t t = sched->steps[s].ops[o].type;
            if (t == OMPI_COLL_SCHED_OP_SEND || t == OMPI_COLL_SCHED_OP_RECV) {
                cnt++;
            }
        }
        if (cnt > max_net) {
            max_net = cnt;
        }
    }

    ompi_request_t **reqs = NULL;
    if (max_net > 0) {
        reqs = (ompi_request_t **) malloc(max_net * sizeof(*reqs));
        if (NULL == reqs) {
            for (int t = 0; t < sched->num_temp_bufs; t++) {
                free(temp_raw[t]);
            }
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
    }

    /* ── Execute steps ───────────────────────────────────────────────── */

    int rc = OMPI_SUCCESS;

    for (int s = 0; s < sched->num_steps && OMPI_SUCCESS == rc; s++) {
        const ompi_coll_sched_step_t *step = &sched->steps[s];
        int num_reqs = 0;

        /* Issue all network ops (non-blocking) */
        for (int o = 0; o < step->num_ops && OMPI_SUCCESS == rc; o++) {
            const ompi_coll_sched_op_t *entry = &step->ops[o];

            if (entry->type == OMPI_COLL_SCHED_OP_SEND) {
                size_t op_count;
                char *ptr = resolve_bufref(&entry->send.buf,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &op_count);
                /* Always post, even for count=0: the peer posts a matching
                 * recv that must be satisfied (e.g. ring allreduce with
                 * count < comm_size has empty slots). */
                rc = MCA_PML_CALL(isend(ptr, op_count, dtype,
                                        entry->send.peer,
                                        base_tag,
                                        MCA_PML_BASE_SEND_STANDARD,
                                        comms[entry->comm_slot],
                                        &reqs[num_reqs++]));

            } else if (entry->type == OMPI_COLL_SCHED_OP_RECV) {
                size_t op_count;
                char *ptr = resolve_bufref(&entry->recv.buf,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &op_count);
                rc = MCA_PML_CALL(irecv(ptr, op_count, dtype,
                                        entry->recv.peer,
                                        base_tag,
                                        comms[entry->comm_slot],
                                        &reqs[num_reqs++]));
            }
        }

        /* Wait for all outstanding network ops before local ops */
        if (num_reqs > 0 && OMPI_SUCCESS == rc) {
            rc = ompi_request_wait_all(num_reqs, reqs, MPI_STATUSES_IGNORE);
        }

        /* Execute local ops (reduce / copy) */
        for (int o = 0; o < step->num_ops && OMPI_SUCCESS == rc; o++) {
            const ompi_coll_sched_op_t *entry = &step->ops[o];

            if (entry->type == OMPI_COLL_SCHED_OP_REDUCE) {
                size_t src_n, dst_n;
                char *src = resolve_bufref(&entry->reduce.src,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &src_n);
                char *dst = resolve_bufref(&entry->reduce.dst,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &dst_n);
                size_t op_count = src_n < dst_n ? src_n : dst_n;
                if (op_count > 0) {
                    ompi_op_reduce(op, src, dst, op_count, dtype);
                }

            } else if (entry->type == OMPI_COLL_SCHED_OP_REDUCE3) {
                size_t s1_n, s2_n, d_n;
                char *src1 = resolve_bufref(&entry->reduce3.src1,
                                             sbuf, rbuf, temp_ptrs,
                                             count, extent, n, &s1_n);
                char *src2 = resolve_bufref(&entry->reduce3.src2,
                                             sbuf, rbuf, temp_ptrs,
                                             count, extent, n, &s2_n);
                char *dst  = resolve_bufref(&entry->reduce3.dst,
                                             sbuf, rbuf, temp_ptrs,
                                             count, extent, n, &d_n);
                size_t op_count = s1_n < s2_n ? s1_n : s2_n;
                if (op_count > 0) {
                    ompi_3buff_op_reduce(op, src1, src2, dst, op_count, dtype);
                }

            } else if (entry->type == OMPI_COLL_SCHED_OP_COPY) {
                size_t src_n, dst_n;
                char *src = resolve_bufref(&entry->copy.src,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &src_n);
                char *dst = resolve_bufref(&entry->copy.dst,
                                            sbuf, rbuf, temp_ptrs,
                                            count, extent, n, &dst_n);
                size_t op_count = src_n < dst_n ? src_n : dst_n;
                if (op_count > 0) {
                    ompi_datatype_copy_content_same_ddt(dtype, op_count, dst, src);
                }
            }
        }
    }

    free(reqs);
    for (int t = 0; t < sched->num_temp_bufs; t++) {
        free(temp_raw[t]);
    }
    return rc;
}

static bool
pml_can_execute(ompi_coll_sched_exec_t *exec,
                const ompi_coll_sched_t *sched,
                struct ompi_communicator_t **comms,
                struct ompi_datatype_t *dtype,
                struct ompi_op_t *op)
{
    /* PML can execute any schedule */
    return true;
}

static void
pml_free(ompi_coll_sched_exec_t *exec)
{
    free(exec);
}

ompi_coll_sched_exec_t *
ompi_coll_sched_exec_pml_create(void)
{
    ompi_coll_sched_exec_pml_t *e = calloc(1, sizeof(*e));
    if (NULL == e) {
        return NULL;
    }
    e->base.can_execute = pml_can_execute;
    e->base.execute     = pml_execute;
    e->base.iexecute    = NULL;
    e->base.free        = pml_free;
    return &e->base;
}
