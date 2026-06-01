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
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_datatype.h"

#define REDUCE_VARIANT_BINOMIAL 0

int
mca_coll_sched_reduce_intra(const void *sbuf, void *rbuf,
                             size_t count,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op,
                             int root,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    /* ── Get or build schedule ──────────────────────────────────────────── */

    ompi_coll_sched_t *sched =
        ompi_coll_sched_cache_get(m, REDUCE, REDUCE_VARIANT_BINOMIAL, root, 0);
    if (NULL == sched) {
        sched = ompi_coll_sched_build_reduce_binomial(rank, n, root);
        if (NULL == sched) {
            return m->c_coll.coll_reduce(sbuf, rbuf, count, dtype, op, root,
                                          comm, m->c_coll.coll_reduce_module);
        }
        ompi_coll_sched_cache_put(m, REDUCE, REDUCE_VARIANT_BINOMIAL, root, 0, sched);
    }

    /* ── Select executor ────────────────────────────────────────────────── */

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_exec(m, sched, comms, dtype, op);
    if (NULL == exec) {
        return m->c_coll.coll_reduce(sbuf, rbuf, count, dtype, op, root,
                                      comm, m->c_coll.coll_reduce_module);
    }

    /* ── Prepare the working buffer ─────────────────────────────────────
     *
     * The schedule uses BUF_RECV as the accumulation buffer and BUF_SEND as
     * the read-only source of this rank's own contribution.  The first recv
     * step uses REDUCE3 to initialise BUF_RECV = op(received, BUF_SEND),
     * so no explicit pre-copy is needed.
     *
     * Root: BUF_RECV = rbuf (the final result destination).
     * Non-root: BUF_RECV = scratch buffer allocated here; result is sent to
     *   parent by the schedule's final send step.
     * MPI_IN_PLACE (root only): rbuf already holds contribution; treat sbuf
     *   as rbuf so BUF_SEND == BUF_RECV — REDUCE3 behaves like REDUCE2.
     */
    void *work_buf     = rbuf;
    char *work_buf_raw = NULL;

    if (rank != root) {
        ptrdiff_t gap;
        ptrdiff_t span = opal_datatype_span(&dtype->super, count, &gap);
        work_buf_raw = (char *) malloc(span);
        if (NULL == work_buf_raw) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        work_buf = work_buf_raw - gap;
    }

    const void *effective_sbuf = (sbuf == MPI_IN_PLACE) ? rbuf : sbuf;

    int rc = exec->execute(exec, sched, comms,
                           effective_sbuf, work_buf, count, dtype, op,
                           MCA_COLL_BASE_TAG_REDUCE);

    free(work_buf_raw);
    return rc;
}
