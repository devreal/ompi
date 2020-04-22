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
#pragma weak MPIX_Continue = PMPIX_Continue
#endif
#define MPIX_Continue PMPIX_Continue
#endif

static const char FUNC_NAME[] = "MPIX_Continue";

int MPIX_Continue(
    MPI_Request *request,
    MPIX_Request_complete_fn_t cb,
    void *cb_data,
    MPI_Status *status)
{
    int rc;

    MEMCHECKER(
        memchecker_request(request);
    );

    if (MPI_PARAM_CHECK) {
        rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (NULL == request) {
            rc = MPI_ERR_REQUEST;
        }
        OMPI_ERRHANDLER_CHECK(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    rc = ompi_request_register_user_completion_cb(1, request, cb, cb_data,
                                                  MPI_STATUS_IGNORE == status
                                                  ? MPI_STATUSES_IGNORE : status);

    *request = MPI_REQUEST_NULL;

    OMPI_ERRHANDLER_RETURN(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
}

