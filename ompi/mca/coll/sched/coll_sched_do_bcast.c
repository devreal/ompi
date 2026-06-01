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
#include "coll_sched.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"

#define BCAST_VARIANT_BINOMIAL 0

int
mca_coll_sched_bcast_intra(void *buffer, size_t count,
                            struct ompi_datatype_t *dtype,
                            int root,
                            struct ompi_communicator_t *comm,
                            mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    /* ── Get or build schedule ──────────────────────────────────────────── */

    ompi_coll_sched_t *sched =
        ompi_coll_sched_cache_get(m, BCAST, BCAST_VARIANT_BINOMIAL, root, 0);
    if (NULL == sched) {
        sched = ompi_coll_sched_build_bcast_binomial(rank, n, root);
        if (NULL == sched) {
            return m->c_coll.coll_bcast(buffer, count, dtype, root,
                                         comm, m->c_coll.coll_bcast_module);
        }
        ompi_coll_sched_cache_put(m, BCAST, BCAST_VARIANT_BINOMIAL, root, 0, sched);
    }

    /* ── Select executor ────────────────────────────────────────────────── */

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_exec(m, sched, comms, dtype, NULL);
    if (NULL == exec) {
        return m->c_coll.coll_bcast(buffer, count, dtype, root,
                                     comm, m->c_coll.coll_bcast_module);
    }

    /* Bcast: buffer is both sbuf (at root) and rbuf.
     * Pass buffer for both so the schedule's RECV refs write into it. */
    return exec->execute(exec, sched, comms,
                         buffer, buffer, count, dtype, NULL,
                         MCA_COLL_BASE_TAG_BCAST);
}
