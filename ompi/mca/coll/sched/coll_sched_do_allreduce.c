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
#include <string.h>
#include "coll_sched.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_datatype.h"

/* Algorithm variant IDs (used as cache keys) */
#define ALLREDUCE_VARIANT_RING   0
#define ALLREDUCE_VARIANT_RDBL   1

/*
 * Select allreduce algorithm, retrieve or build schedule, execute.
 *
 * The dispatch:
 *  1. Handles MPI_IN_PLACE.
 *  2. Copies sbuf → rbuf so the schedule works in-place on rbuf.
 *  3. Looks up / builds a cached schedule.
 *  4. Picks the best available executor.
 *  5. Falls back to the saved lower-priority function if needed.
 */
int
mca_coll_sched_allreduce_intra(const void *sbuf, void *rbuf,
                                size_t count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    /* ── Algorithm selection ────────────────────────────────────────────── */

    /* Prefer recursive doubling for power-of-two comms (better latency).
     * Fall back to ring for all other cases (bandwidth-friendly). */
    int variant;
    if ((n & (n - 1)) == 0) {
        variant = ALLREDUCE_VARIANT_RDBL;
    } else {
        variant = ALLREDUCE_VARIANT_RING;
    }

    /* ── Get or build schedule ──────────────────────────────────────────── */

    ompi_coll_sched_t *sched = ompi_coll_sched_cache_get(m, ALLREDUCE, variant, -1, 0);
    if (NULL == sched) {
        sched = (variant == ALLREDUCE_VARIANT_RDBL)
                ? ompi_coll_sched_build_allreduce_recursivedoubling(rank, n)
                : ompi_coll_sched_build_allreduce_ring(rank, n);

        if (NULL == sched) {
            /* Fall back to lower-priority component */
            return m->c_coll.coll_allreduce(sbuf, rbuf, count, dtype, op, comm,
                                             m->c_coll.coll_allreduce_module);
        }
        ompi_coll_sched_cache_put(m, ALLREDUCE, variant, -1, 0, sched);
    }

    /* ── Select executor ────────────────────────────────────────────────── */

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_exec(m, sched, comms, dtype, op);
    if (NULL == exec) {
        return m->c_coll.coll_allreduce(sbuf, rbuf, count, dtype, op, comm,
                                         m->c_coll.coll_allreduce_module);
    }

    /* Pass sbuf and rbuf as separate buffers so the schedule's REDUCE3 ops can
     * fold the "seed with own contribution" step into the first reduction,
     * avoiding an explicit full-message copy.
     * MPI_IN_PLACE: rbuf already holds our contribution; pass it as both
     * sbuf and rbuf — REDUCE3(temp, rbuf_slot, rbuf_slot) is equivalent to
     * the 2-buffer reduce in that case. */
    const void *effective_sbuf = (sbuf == MPI_IN_PLACE) ? rbuf : sbuf;

    return exec->execute(exec, sched, comms, effective_sbuf, rbuf, count, dtype, op,
                         MCA_COLL_BASE_TAG_ALLREDUCE);
}
