/*
 * Copyright (c) 2026      Joseph Antony.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Notified RMA (MPI-5.1 section 12.6) is optional: an osc component that
 * does not implement part of it leaves the corresponding entries of the
 * module struct NULL.  Every such entry point must report that as
 * MPI_ERR_UNSUPPORTED_OPERATION rather than calling through a NULL
 * function pointer.
 *
 * osc/rdma is forced because it is a general-purpose component that
 * implements the blocking put and get with notification and the counter
 * management calls, but none of the accumulate or request-based notified
 * operations, so it exercises the guard on those entry points.  If it cannot
 * be selected in this build the test reports that and passes trivially.
 *
 * Because osc/rdma does implement counter management, the window must also
 * report usable notification bounds rather than the zeros cached for a
 * component with no support at all.
 *
 * Note: the library is compiled with -DNDEBUG, so assert() is a no-op
 * here -- all verification must go through test_verify().
 */

#include "ompi_config.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "support.h"

#include "mpi.h"

#define WIN_COUNT 8

int main(int argc, char *argv[])
{
    /* Must be set before MPI_Init: component selection happens there. */
    setenv("OMPI_MCA_osc", "rdma", 1);

    test_init("ompi win_notify_unsupported");

    int rc = MPI_Init(&argc, &argv);
    test_verify("MPI_Init succeeds", MPI_SUCCESS == rc);

    int *base = NULL;
    MPI_Win win = MPI_WIN_NULL;
    rc = MPI_Win_allocate(WIN_COUNT * sizeof(int), sizeof(int), MPI_INFO_NULL,
                          MPI_COMM_SELF, &base, &win);
    if (MPI_SUCCESS != rc) {
        test_comment("osc/rdma not selectable in this build; skipping");
        int r = test_finalize();
        MPI_Finalize();
        return r;
    }
    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
    memset(base, 0, WIN_COUNT * sizeof(int));

    int src = 1;
    int result = 0;
    MPI_Request req = MPI_REQUEST_NULL;

    MPI_Win_lock_all(0, win);

    /* The blocking accumulate operations. */
    rc = MPI_Accumulate_notify(&src, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM,
                               0, win);
    test_verify("Accumulate_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);
    rc = MPI_Get_accumulate_notify(&src, 1, MPI_INT, &result, 1, MPI_INT,
                                   0, 0, 1, MPI_INT, MPI_SUM, 0, win);
    test_verify("Get_accumulate_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);

    /* The four request-based communication operations. */
    rc = MPI_Rput_notify(&src, 1, MPI_INT, 0, 0, 1, MPI_INT, 0, win, &req);
    test_verify("Rput_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);
    rc = MPI_Rget_notify(&result, 1, MPI_INT, 0, 0, 1, MPI_INT, 0, win, &req);
    test_verify("Rget_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);
    rc = MPI_Raccumulate_notify(&src, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM,
                                0, win, &req);
    test_verify("Raccumulate_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);
    rc = MPI_Rget_accumulate_notify(&src, 1, MPI_INT, &result, 1, MPI_INT,
                                    0, 0, 1, MPI_INT, MPI_SUM, 0, win, &req);
    test_verify("Rget_accumulate_notify reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);

    /* The guard sits ahead of the MPI_PROC_NULL no-op, so an unsupported
     * operation is reported identically no matter what the target is. */
    rc = MPI_Accumulate_notify(&src, 1, MPI_INT, MPI_PROC_NULL, 0, 1, MPI_INT,
                               MPI_SUM, 0, win);
    test_verify("Accumulate_notify to MPI_PROC_NULL reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);
    rc = MPI_Rput_notify(&src, 1, MPI_INT, MPI_PROC_NULL, 0, 1, MPI_INT, 0,
                         win, &req);
    test_verify("Rput_notify to MPI_PROC_NULL reports unsupported",
                MPI_ERR_UNSUPPORTED_OPERATION == rc);

    /* Nothing above should have moved any data. */
    test_verify("no unsupported operation touched the window", 0 == base[0]);

    MPI_Win_unlock_all(win);

    /* MPI-5.1 section 12.2.6: the notification bounds are cached on every
     * window.  osc/rdma can attach counters, so it must not report the zeros
     * that mean "no notification support". */
    int *num_sb = NULL, *num_ub = NULL;
    MPI_Count *value_ub = NULL;
    int flag = 0;

    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_SB, &num_sb, &flag);
    test_verify("NUM_SB is present and positive",
                MPI_SUCCESS == rc && flag && NULL != num_sb && 0 < *num_sb);
    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_UB, &num_ub, &flag);
    test_verify("NUM_UB is present and positive",
                MPI_SUCCESS == rc && flag && NULL != num_ub && 0 < *num_ub);
    test_verify("NUM_SB does not exceed NUM_UB",
                NULL != num_sb && NULL != num_ub && *num_sb <= *num_ub);
    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_VALUE_UB, &value_ub, &flag);
    test_verify("VALUE_UB is present and positive",
                MPI_SUCCESS == rc && flag && NULL != value_ub && 0 < *value_ub);

    /* Every count up to NUM_UB can be attached, and nothing beyond it. */
    if (NULL != num_ub && 0 < *num_ub && INT_MAX > *num_ub) {
        rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, *num_ub);
        test_verify("Win_set_num_notify accepts NUM_UB counters", MPI_SUCCESS == rc);
        rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, *num_ub + 1);
        test_verify("Win_set_num_notify rejects more than NUM_UB counters",
                    MPI_ERR_ARG == rc);
    }

    MPI_Win_free(&win);

    int r = test_finalize();
    MPI_Finalize();
    return r;
}
