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
#pragma weak MPI_Continueall = PMPI_Continueall
#endif
#define MPI_Continueall PMPI_Continueall
#endif

static const char FUNC_NAME[] = "MPI_Continueall";

int MPI_Continueall(
    int                count,
    MPI_Request        requests[],
    int               *flag,
    MPI_Continue_cb_t *cont_cb,
    void              *cont_data,
    MPI_Status         statuses[],
    MPI_Request        cont_req)
{
    int rc;
    bool all_complete = false;

    MEMCHECKER(
        for (int j = 0; j < count; j++){
            memchecker_request(&requests[j]);
        }
    );


    if (MPI_PARAM_CHECK) {
        rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (MPI_REQUEST_NULL == cont_req || OMPI_REQUEST_CONT != cont_req->req_type) {
            rc = MPI_ERR_REQUEST;
        }
        if (NULL == flag) {
            rc = MPI_ERR_ARG;
        }
        if( (NULL == requests) && (0 != count) ) {
            rc = MPI_ERR_REQUEST;
        } else {
            for (int i = 0; i < count; i++) {
                if (NULL == requests[i]) {
                    rc = MPI_ERR_REQUEST;
                    break;
                }
            }
        }
        OMPI_ERRHANDLER_CHECK(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    rc = ompi_request_cont_register(cont_req, count, requests, cont_cb,
                                    cont_data, &all_complete, statuses);

    *flag = (all_complete) ? 1 : 0;

    OMPI_ERRHANDLER_RETURN(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
}

