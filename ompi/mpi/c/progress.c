/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2019 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2009 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
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
#include "ompi/request/request.h"

#include "opal/runtime/opal_progress.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Progress = PMPIX_Progress
#endif
#define MPIX_Progress PMPIX_Progress
#endif

static const char FUNC_NAME[] = "MPIX_Progress";

int MPIX_Progress(void)
{
    int rc;

    // This makes it a pending request progress, see if this helps
    //rc = ompi_request_progress_user_completion();
    //if (rc == 0) {
      opal_progress();
      //ompi_request_progress_user_completion();
    //}
    return MPI_SUCCESS;
}

