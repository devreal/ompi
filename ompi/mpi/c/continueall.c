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
#pragma weak MPIX_Continueall = PMPIX_Continueall
#endif
#define MPIX_Continueall PMPIX_Continueall
#endif

static const char FUNC_NAME[] = "MPIX_Continueall";

int MPIX_Continueall(
    int count,
    MPI_Request requests[],
    MPIX_Request_complete_fn_t cb,
    void *cb_data,
    MPI_Status statuses[])
{
    int rc;

    MEMCHECKER(
        for (int j = 0; j < count; j++){
            memchecker_request(&requests[j]);
        }
    );


    if (MPI_PARAM_CHECK) {
        rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (NULL == requests) {
            rc = MPI_ERR_REQUEST;
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

    rc = ompi_request_register_user_completion_cb(count, requests, cb, cb_data, statuses);

    for (int i = 0; i < count; ++i) {
        requests[i] = MPI_REQUEST_NULL;
    }

    OMPI_ERRHANDLER_RETURN(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
}

