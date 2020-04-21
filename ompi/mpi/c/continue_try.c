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
#include "ompi/request/request_cb.h"
#include "ompi/memchecker.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Continue_try = PMPIX_Continue_try
#endif
#define MPIX_Continue_try PMPIX_Continue_try
#endif

static const char FUNC_NAME[] = "MPIX_Continue_try";

int MPIX_Continue_try(int *ndone, int *nremain)
{
    int rc = MPI_SUCCESS;

    if (MPI_PARAM_CHECK) {
        rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (NULL == ndone || NULL == nremain) {
            rc = MPI_ERR_ARG;
        }
        OMPI_ERRHANDLER_CHECK(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
    }


    OPAL_CR_ENTER_LIBRARY();

    uint64_t num_processed_entry = ompi_request_user_callback_num_processed();

    /* calling opal_progress once is sufficient here */
    opal_progress();

    uint64_t num_processed_exit = ompi_request_user_callback_num_processed();

    // TODO: handle overflow gracefully
    *ndone = num_processed_exit - num_processed_entry;

    *nremain = ompi_request_user_callback_num_active();

    OMPI_ERRHANDLER_RETURN(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
}

