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
#include "coll_sched.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_datatype.h"

#define REDUCE_VARIANT_BINOMIAL 0

int
mca_coll_sched_ireduce_intra(const void *sendbuf, void *recvbuf,
                              size_t count,
                              struct ompi_datatype_t *datatype,
                              struct ompi_op_t *op,
                              int root,
                              struct ompi_communicator_t *comm,
                              ompi_request_t **request,
                              mca_coll_base_module_t *module)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;
    int rank = ompi_comm_rank(comm);
    int n    = ompi_comm_size(comm);

    ompi_coll_sched_t *sched =
        ompi_coll_sched_cache_get(m, REDUCE, REDUCE_VARIANT_BINOMIAL, root, 0);
    if (NULL == sched) {
        sched = ompi_coll_sched_build_reduce_binomial(rank, n, root);
        if (NULL == sched) {
            return m->c_coll.coll_ireduce(sendbuf, recvbuf, count, datatype, op, root,
                                           comm, request, m->c_coll.coll_ireduce_module);
        }
        ompi_coll_sched_cache_put(m, REDUCE, REDUCE_VARIANT_BINOMIAL, root, 0, sched);
    }

    struct ompi_communicator_t *comms[1] = {comm};
    ompi_coll_sched_exec_t *exec = ompi_coll_sched_select_iexec(m, sched, comms, datatype, op);
    if (NULL == exec) {
        return m->c_coll.coll_ireduce(sendbuf, recvbuf, count, datatype, op, root,
                                       comm, request, m->c_coll.coll_ireduce_module);
    }

    /* Non-root ranks need a scratch buffer to accumulate into (BUF_RECV).
     * It must live until the operation completes; attach it to the request
     * via ompi_coll_base_append_array_to_release so cb_nbc_request_free frees it. */
    void *work_buf     = recvbuf;
    char *work_buf_raw = NULL;

    if (rank != root) {
        ptrdiff_t gap;
        ptrdiff_t span = opal_datatype_span(&datatype->super, count, &gap);
        work_buf_raw = (char *) malloc(span);
        if (NULL == work_buf_raw) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        work_buf = work_buf_raw - gap;
    }

    const void *effective_sbuf = (sendbuf == MPI_IN_PLACE) ? recvbuf : sendbuf;
    int tag = ompi_coll_base_nbc_reserve_tags(comm, 1);

    int rc = exec->iexecute(exec, sched, comms, effective_sbuf, work_buf, count,
                             datatype, op, tag, request);
    if (OMPI_SUCCESS != rc) {
        free(work_buf_raw);
        return rc;
    }

    if (NULL != work_buf_raw) {
        ompi_coll_base_append_array_to_release(*request, work_buf_raw);
    }

    return OMPI_SUCCESS;
}
