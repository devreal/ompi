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
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/coll/base/coll_tags.h"

#define BCAST_VARIANT_BINOMIAL 0
#define BCAST_VARIANT_CHAIN    1

int
mca_coll_sched_ibcast_intra(void *buffer, size_t count,
                             struct ompi_datatype_t *datatype,
                             int root,
                             struct ompi_communicator_t *comm,
                             ompi_request_t **request,
                             mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    int variant = (n == 2) ? BCAST_VARIANT_CHAIN : BCAST_VARIANT_BINOMIAL;

    ompi_coll_sched_t *sched =
        ompi_coll_sched_cache_get(m, BCAST, variant, root, 0);
    if (NULL == sched) {
        sched = (variant == BCAST_VARIANT_CHAIN)
                ? ompi_coll_sched_build_bcast_chain(rank, n, root)
                : ompi_coll_sched_build_bcast_binomial(rank, n, root);
        if (NULL == sched) {
            return m->c_coll.coll_ibcast(buffer, count, datatype, root,
                                          comm, request, m->c_coll.coll_ibcast_module);
        }
        ompi_coll_sched_cache_put(m, BCAST, variant, root, 0, sched);
    }

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_iexec(m, sched, comms, datatype, NULL);
    if (NULL == exec) {
        return m->c_coll.coll_ibcast(buffer, count, datatype, root,
                                      comm, request, m->c_coll.coll_ibcast_module);
    }

    int tag = ompi_coll_base_nbc_reserve_tags(comm, 1);

    return exec->iexecute(exec, sched, comms, buffer, buffer, count, datatype, NULL,
                          tag, request);
}
