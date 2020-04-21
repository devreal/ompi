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
#pragma weak MPIX_Continue_waitsome = PMPIX_Continue_waitsome
#endif
#define MPIX_Continue_waitsome PMPIX_Continue_waitsome
#endif

static const char FUNC_NAME[] = "MPIX_Continue_waitsome";

int MPIX_Continue_waitsome(int *ndone, int *nremain)
{

    if (MPI_PARAM_CHECK) {
        int rc = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (NULL == ndone || NULL == nremain) {
            rc = MPI_ERR_ARG;
        }
        OMPI_ERRHANDLER_CHECK(rc, MPI_COMM_WORLD, rc, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    int num_remain = ompi_request_user_callback_num_active();

    if (0 < num_remain) {

        const uint64_t num_processed_entry = ompi_request_user_callback_num_processed();
        uint64_t num_processed_exit  = num_processed_entry;

        /* loop until a) no active callbacks are available; or b) we have processed at least one */
        while (0 < (num_remain = ompi_request_user_callback_num_active()) &&
               num_processed_entry == num_processed_exit) {
            /* calling opal_progress is sufficient */
            opal_progress();
            num_processed_exit = ompi_request_user_callback_num_processed();
        }

        /* output variables */
        *ndone   = num_processed_exit - num_processed_entry;
        *nremain = num_remain;

    } else {
        /* nothing to be done */
        *ndone   = 0;
        *nremain = 0;
    }

    return MPI_SUCCESS;
}

