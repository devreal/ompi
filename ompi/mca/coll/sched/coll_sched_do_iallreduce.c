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
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"

#define ALLREDUCE_VARIANT_RING          0
#define ALLREDUCE_VARIANT_RDBL          1
#define ALLREDUCE_VARIANT_NONOVERLAP    2

int
mca_coll_sched_iallreduce_intra(const void *sendbuf, void *recvbuf,
                                 size_t count,
                                 struct ompi_datatype_t *datatype,
                                 struct ompi_op_t *op,
                                 struct ompi_communicator_t *comm,
                                 ompi_request_t **request,
                                 mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    int variant = ((n & (n - 1)) == 0) ? ALLREDUCE_VARIANT_RDBL : ALLREDUCE_VARIANT_NONOVERLAP;

    ompi_coll_sched_t *sched = ompi_coll_sched_cache_get(m, ALLREDUCE, variant, -1, 0);
    if (NULL == sched) {
        if (variant == ALLREDUCE_VARIANT_RDBL) {
            sched = ompi_coll_sched_build_allreduce_recursivedoubling(rank, n);
        } else {
            sched = ompi_coll_sched_build_allreduce_nonoverlapping(rank, n);
        }
        if (NULL == sched) {
            return m->c_coll.coll_iallreduce(sendbuf, recvbuf, count, datatype, op, comm,
                                              request, m->c_coll.coll_iallreduce_module);
        }
        ompi_coll_sched_cache_put(m, ALLREDUCE, variant, -1, 0, sched);
    }

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_iexec(m, sched, comms, datatype, op);
    if (NULL == exec) {
        return m->c_coll.coll_iallreduce(sendbuf, recvbuf, count, datatype, op, comm,
                                          request, m->c_coll.coll_iallreduce_module);
    }

    /* Each concurrent Iallreduce must use a unique tag to avoid message matching
     * across concurrent invocations on the same communicator. */
    int tag = ompi_coll_base_nbc_reserve_tags(comm, 1);

    const void *effective_sbuf = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;

    return exec->iexecute(exec, sched, comms, effective_sbuf, recvbuf, count, datatype, op,
                          tag, request);
}
