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

/* ── Schedule alloc / free ─────────────────────────────────────────────── */

ompi_coll_sched_t *
ompi_coll_sched_alloc(int num_steps)
{
    ompi_coll_sched_t *sched = calloc(1, sizeof(*sched));
    if (NULL == sched) {
        return NULL;
    }

    if (num_steps > 0) {
        sched->steps = calloc(num_steps, sizeof(ompi_coll_sched_step_t));
        if (NULL == sched->steps) {
            free(sched);
            return NULL;
        }
    }

    sched->num_steps     = num_steps;
    sched->num_temp_bufs = 0;
    sched->num_comm_slots = 1;
    return sched;
}

void
ompi_coll_sched_free(ompi_coll_sched_t *sched)
{
    if (NULL == sched) {
        return;
    }
    for (int s = 0; s < sched->num_steps; s++) {
        free(sched->steps[s].ops);
    }
    free(sched->steps);
    free(sched);
}

/* ── Step initialiser ──────────────────────────────────────────────────── */

int
ompi_coll_sched_step_init(ompi_coll_sched_t *sched, int step_idx,
                           int num_ops, bool barrier)
{
    ompi_coll_sched_step_t *step = &sched->steps[step_idx];
    step->ops = calloc(num_ops, sizeof(ompi_coll_sched_op_t));
    if (NULL == step->ops) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    step->num_ops = num_ops;
    step->barrier = barrier;
    return OMPI_SUCCESS;
}

/* ── Temp buffer registration ──────────────────────────────────────────── */

int
ompi_coll_sched_add_temp_buf(ompi_coll_sched_t *sched, bool full_size)
{
    int idx = sched->num_temp_bufs;
    if (idx >= OMPI_COLL_SCHED_MAX_TEMP_BUFS) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    sched->temp_full_size[idx] = full_size;
    sched->num_temp_bufs++;
    return OMPI_COLL_SCHED_BUF_TEMP(idx);
}

/* ── Op builders ───────────────────────────────────────────────────────── */

void
ompi_coll_sched_op_send(ompi_coll_sched_t *s, int step, int op_idx,
                         int comm_slot, int peer, ompi_coll_sched_bufref_t buf)
{
    ompi_coll_sched_op_t *op = &s->steps[step].ops[op_idx];
    op->type             = OMPI_COLL_SCHED_OP_SEND;
    op->comm_slot        = comm_slot;
    op->send.peer        = peer;
    op->send.buf         = buf;
}

void
ompi_coll_sched_op_recv(ompi_coll_sched_t *s, int step, int op_idx,
                         int comm_slot, int peer, ompi_coll_sched_bufref_t buf)
{
    ompi_coll_sched_op_t *op = &s->steps[step].ops[op_idx];
    op->type             = OMPI_COLL_SCHED_OP_RECV;
    op->comm_slot        = comm_slot;
    op->recv.peer        = peer;
    op->recv.buf         = buf;
}

void
ompi_coll_sched_op_reduce(ompi_coll_sched_t *s, int step, int op_idx,
                           ompi_coll_sched_bufref_t src, ompi_coll_sched_bufref_t dst)
{
    ompi_coll_sched_op_t *op = &s->steps[step].ops[op_idx];
    op->type        = OMPI_COLL_SCHED_OP_REDUCE;
    op->comm_slot   = 0;
    op->reduce.src  = src;
    op->reduce.dst  = dst;
}

void
ompi_coll_sched_op_reduce3(ompi_coll_sched_t *s, int step, int op_idx,
                            ompi_coll_sched_bufref_t src1,
                            ompi_coll_sched_bufref_t src2,
                            ompi_coll_sched_bufref_t dst)
{
    ompi_coll_sched_op_t *op = &s->steps[step].ops[op_idx];
    op->type           = OMPI_COLL_SCHED_OP_REDUCE3;
    op->comm_slot      = 0;
    op->reduce3.src1   = src1;
    op->reduce3.src2   = src2;
    op->reduce3.dst    = dst;
}

void
ompi_coll_sched_op_copy(ompi_coll_sched_t *s, int step, int op_idx,
                         ompi_coll_sched_bufref_t src, ompi_coll_sched_bufref_t dst)
{
    ompi_coll_sched_op_t *op = &s->steps[step].ops[op_idx];
    op->type        = OMPI_COLL_SCHED_OP_COPY;
    op->comm_slot   = 0;
    op->copy.src    = src;
    op->copy.dst    = dst;
}

/* ── Schedule cache ────────────────────────────────────────────────────── */

ompi_coll_sched_t *
ompi_coll_sched_cache_get(mca_coll_sched_module_t *m,
                           int colltype, int variant, int root, int param)
{
    for (int i = 0; i < m->cache_count; i++) {
        ompi_coll_sched_cache_entry_t *e = &m->cache[i];
        if (e->colltype == colltype && e->variant == variant &&
            e->root == root && e->param == param) {
            return e->sched;
        }
    }
    return NULL;
}

int
ompi_coll_sched_cache_put(mca_coll_sched_module_t *m,
                           int colltype, int variant, int root, int param,
                           ompi_coll_sched_t *sched)
{
    if (m->cache_count >= OMPI_COLL_SCHED_CACHE_SIZE) {
        /* Cache full: evict the oldest entry (index 0) */
        ompi_coll_sched_free(m->cache[0].sched);
        memmove(&m->cache[0], &m->cache[1],
                (OMPI_COLL_SCHED_CACHE_SIZE - 1) * sizeof(m->cache[0]));
        m->cache_count--;
    }
    ompi_coll_sched_cache_entry_t *e = &m->cache[m->cache_count++];
    e->colltype = colltype;
    e->variant  = variant;
    e->root     = root;
    e->param    = param;
    e->sched    = sched;
    return OMPI_SUCCESS;
}

/* ── Executor selection ────────────────────────────────────────────────── */

ompi_coll_sched_exec_t *
ompi_coll_sched_select_exec(mca_coll_sched_module_t *m,
                             const ompi_coll_sched_t *sched,
                             struct ompi_communicator_t **comms,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op)
{
    for (int i = 0; i < m->num_executors; i++) {
        ompi_coll_sched_exec_t *e = m->executors[i];
        if (e && e->can_execute(e, sched, comms, dtype, op)) {
            return e;
        }
    }
    return NULL;
}
