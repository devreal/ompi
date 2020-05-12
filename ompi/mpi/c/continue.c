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
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/request/request.h"
#include "ompi/memchecker.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Continue = PMPI_Continue
#endif
#define MPI_Continue PMPI_Continue
#endif

static const char FUNC_NAME[] = "MPI_Continue";

int MPI_Continue(
    MPI_Request       *request,
    int               *flag,
    MPI_Continue_cb_t *cont_cb,
    void              *cb_data,
    MPI_Status        *status,
    MPI_Request        cont_req)
{
    int rc;
    bool all_complete = false;

    MEMCHECKER(
        memchecker_request(request);
    );

    if (MPI_PARAM_CHECK) {
        rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (NULL == request) {
            rc = MPI_ERR_REQUEST;
        }
        if (NULL == flag) {
            rc = MPI_ERR_ARG;
        }
        if (MPI_REQUEST_NULL == cont_req || OMPI_REQUEST_CONT != cont_req->req_type) {
            rc = MPI_ERR_REQUEST;
        }
        OMPI_ERRHANDLER_CHECK(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    rc = ompi_request_cont_register(cont_req, 1, request, cont_cb, cb_data, &all_complete,
                                    MPI_STATUS_IGNORE == status ? MPI_STATUSES_IGNORE : status);

    *flag = (all_complete) ? 1 : 0;

    OMPI_ERRHANDLER_RETURN(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
}

