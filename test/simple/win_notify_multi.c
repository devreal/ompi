/*
 * Copyright (c) 2026      Joseph Antony.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Multi-process test for notified RMA communication (MPI-5.1 section 12.6)
 * on a shared memory window.
 *
 * ompi/test/general/win_notify.c covers the same interfaces on
 * MPI_COMM_SELF, which leaves every comm_size > 1 path in osc/sm
 * untested: the notification counters carved out of the shared segment at
 * window creation, the allgather that sizes them, the collective
 * reallocation MPI_Win_set_num_notify performs when a rank asks for more
 * counters than were reserved, and the recomputation of each rank's
 * counter base that follows it.  This program exercises those, so it has
 * to be launched as a real job:
 *
 *     mpirun --np 4 ./win_notify_multi
 *
 * All ranks must land on one node: the window is created with
 * MPI_Win_allocate_shared, and osc/sm is currently the only component
 * implementing the notified operations.
 *
 * Every rank checks its own results and the failures are reduced at the
 * end, so a failure on any rank fails the run.
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

/* Minimum window size in ints.  test_collective_growth() has every rank
 * write to the slot named by its own rank, so a window must hold at least
 * one int per rank; win_elems carries that once the job size is known. */
#define WIN_COUNT 8

/* Comfortably more than the osc_sm_num_notify_counters default of 16, so
 * that asking for this many forces the counters to be reallocated. */
#define NUM_NOTIFY_GROWN 100

/* A notified operation on a shared memory window completes at the target
 * without any help from the target process, so a bounded spin is enough to
 * see it; the bound only keeps a broken build from hanging the test. */
#define MAX_SPINS 100000000L

static int rank, nprocs, failures, win_elems;

static void check(const char *what, int ok)
{
    if (!ok) {
        failures++;
        fprintf(stderr, "FAIL [rank %d] %s\n", rank, what);
    }
}

/* Spin until notification "idx" on this rank's window reaches "expect". */
static int await_notify(MPI_Win win, int idx, MPI_Count expect)
{
    MPI_Count value = -1;
    long spins;

    for (spins = 0 ; spins < MAX_SPINS ; ++spins) {
        if (MPI_SUCCESS != MPI_Win_get_notify_value(win, idx, &value)) {
            return 0;
        }
        if (value >= expect) {
            break;
        }
    }

    return expect == value;
}

/* Read "key" out of the window's info, or leave "buf" empty if absent. */
static void get_win_info_value(MPI_Win win, const char *key, char *buf, int buflen)
{
    MPI_Info used = MPI_INFO_NULL;
    int flag = 0;

    buf[0] = '\0';
    MPI_Win_get_info(win, &used);
    MPI_Info_get_string(used, key, &buflen, buf, &flag);
    if (!flag) {
        buf[0] = '\0';
    }
    MPI_Info_free(&used);
}

static MPI_Win make_window(MPI_Info info, int **base)
{
    MPI_Win win = MPI_WIN_NULL;
    int rc;

    rc = MPI_Win_allocate_shared(win_elems * sizeof(int), sizeof(int), info,
                                 MPI_COMM_WORLD, base, &win);
    check("Win_allocate_shared succeeds", MPI_SUCCESS == rc);
    if (MPI_SUCCESS != rc) {
        return MPI_WIN_NULL;
    }

    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
    memset(*base, 0, win_elems * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);

    return win;
}

/* ------------------------------------------------------------------ */

/* The counters a shared window is created with live in the segment osc/sm
 * lays out for every rank at once.  Pass one notification around the ring
 * to show that a notified put reaches the counter the target reads. */
static void test_ring_notify(void)
{
    int *base = NULL;
    MPI_Win win;
    int right = (rank + 1) % nprocs;
    int left = (rank + nprocs - 1) % nprocs;
    MPI_Count value = -1;
    int rc;

    win = make_window(MPI_INFO_NULL, &base);
    if (MPI_WIN_NULL == win) {
        return;
    }

    rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, 4);
    check("Win_set_num_notify succeeds within the reservation",
          MPI_SUCCESS == rc);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(0, win);

    rc = MPI_Put_notify(&rank, 1, MPI_INT, right, 0, 1, MPI_INT, 1, win);
    check("Put_notify to the right neighbour succeeds", MPI_SUCCESS == rc);
    MPI_Win_flush(right, win);

    check("the notification from the left neighbour arrives",
          await_notify(win, 1, 1));

    /* The notification is what makes the data safe to read. */
    check("Put_notify moved the neighbour's data", left == base[0]);

    /* Nothing may land on a counter no operation named. */
    rc = MPI_Win_get_notify_value(win, 0, &value);
    check("an unused counter stays at zero", MPI_SUCCESS == rc && 0 == value);

    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);
}

/* ------------------------------------------------------------------ */

/* Asking for more counters than were reserved moves every rank's counters
 * into a freshly created shared segment.  Each rank then has to find every
 * other rank's counters at their new addresses, so notify one counter per
 * source rank and check that each landed where its source intended. */
static void test_collective_growth(void)
{
    int *base = NULL;
    MPI_Win win;
    MPI_Count value = -1;
    int num = -1;
    int i, rc;

    win = make_window(MPI_INFO_NULL, &base);
    if (MPI_WIN_NULL == win) {
        return;
    }

    rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, NUM_NOTIFY_GROWN);
    check("Win_set_num_notify grows the counters", MPI_SUCCESS == rc);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0 ; i < nprocs ; ++i) {
        rc = MPI_Win_get_num_notify(win, i, &num);
        check("every rank reports the grown count",
              MPI_SUCCESS == rc && NUM_NOTIFY_GROWN == num);
    }

    MPI_Win_lock_all(0, win);

    /* Rank i notifies counter i on every other rank, so a rank that
     * miscomputed any peer's counter base shows up as a lost or misplaced
     * notification rather than as a silently passing test. */
    for (i = 0 ; i < nprocs ; ++i) {
        if (i == rank) {
            continue;
        }
        rc = MPI_Put_notify(&rank, 1, MPI_INT, i, rank, 1, MPI_INT, rank, win);
        check("Put_notify after growth succeeds", MPI_SUCCESS == rc);
    }
    MPI_Win_flush_all(win);

    for (i = 0 ; i < nprocs ; ++i) {
        if (i == rank) {
            continue;
        }
        check("the notification from each peer arrives",
              await_notify(win, i, 1));
        check("each peer wrote to its own slot", i == base[i]);
    }

    /* This rank's own counter names no source, so nothing may touch it. */
    rc = MPI_Win_get_notify_value(win, rank, &value);
    check("no peer notified this rank's own counter",
          MPI_SUCCESS == rc && 0 == value);

    /* The topmost grown counter has to be reachable, not just allocated. */
    rc = MPI_Put_notify(&rank, 1, MPI_INT, rank, 0, 1, MPI_INT,
                        NUM_NOTIFY_GROWN - 1, win);
    check("Put_notify reaches the last grown counter", MPI_SUCCESS == rc);
    MPI_Win_flush(rank, win);
    check("the last grown counter advances",
          await_notify(win, NUM_NOTIFY_GROWN - 1, 1));

    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);
}

/* ------------------------------------------------------------------ */

/* Ranks need not attach the same number of counters.  Each rank's count is
 * its own, and it is the target's count that bounds a notification index. */
static void test_asymmetric_counts(void)
{
    int *base = NULL;
    MPI_Win win;
    int mine = 2 + rank;
    int right = (rank + 1) % nprocs;
    int num = -1;
    int i, rc;

    win = make_window(MPI_INFO_NULL, &base);
    if (MPI_WIN_NULL == win) {
        return;
    }

    rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, mine);
    check("Win_set_num_notify accepts a per-rank count", MPI_SUCCESS == rc);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0 ; i < nprocs ; ++i) {
        rc = MPI_Win_get_num_notify(win, i, &num);
        check("each rank's own count is reported",
              MPI_SUCCESS == rc && (2 + i) == num);
    }

    MPI_Win_lock_all(0, win);

    /* The target's last valid index, which differs from ours. */
    rc = MPI_Put_notify(&rank, 1, MPI_INT, right, 0, 1, MPI_INT,
                        (2 + right) - 1, win);
    check("Put_notify accepts the target's last index", MPI_SUCCESS == rc);

    /* One past it belongs to no counter on that target. */
    rc = MPI_Put_notify(&rank, 1, MPI_INT, right, 0, 1, MPI_INT, 2 + right, win);
    check("Put_notify rejects an index past the target's count",
          MPI_ERR_RMA_NOTIFICATION == rc);

    MPI_Win_flush_all(win);
    check("the in-range notification arrived", await_notify(win, mine - 1, 1));

    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);
}

/* ------------------------------------------------------------------ */

/* The window reports mpi_assert_max_num_notify through its info
 * subscriber, and the notification bounds through its attributes.  Both
 * must agree on every rank of a shared window. */
static void test_info_and_attributes(void)
{
    int *base = NULL;
    MPI_Win win;
    MPI_Info info = MPI_INFO_NULL;
    char value[32];
    int *num_sb = NULL, *num_ub = NULL;
    int flag = 0, num = -1, rc;

    /* Without an assertion the reservation is only a suggestion: nothing
     * bounds what may be requested later. */
    win = make_window(MPI_INFO_NULL, &base);
    if (MPI_WIN_NULL == win) {
        return;
    }

    get_win_info_value(win, "mpi_assert_max_num_notify", value, sizeof(value));
    check("no assertion is reported as 0", 0 == strcmp(value, "0"));

    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_SB, &num_sb, &flag);
    check("NUM_SB is present and positive",
          MPI_SUCCESS == rc && flag && *num_sb > 0);
    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_UB, &num_ub, &flag);
    check("NUM_UB is unbounded without an assertion",
          MPI_SUCCESS == rc && flag && INT_MAX == *num_ub);

    MPI_Info_create(&info);
    MPI_Info_set(info, "mpi_assert_max_num_notify", "1000");
    rc = MPI_Win_set_info(win, info);
    check("Win_set_info succeeds", MPI_SUCCESS == rc);
    MPI_Info_free(&info);

    get_win_info_value(win, "mpi_assert_max_num_notify", value, sizeof(value));
    check("Win_set_info cannot invent an assertion", 0 == strcmp(value, "0"));

    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);

    /* With an assertion the window is sized for exactly that many, and the
     * promise is held against every rank. */
    MPI_Info_create(&info);
    MPI_Info_set(info, "mpi_assert_max_num_notify", "8");
    win = make_window(info, &base);
    MPI_Info_free(&info);
    if (MPI_WIN_NULL == win) {
        return;
    }

    get_win_info_value(win, "mpi_assert_max_num_notify", value, sizeof(value));
    check("the assertion is reported back", 0 == strcmp(value, "8"));

    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_SB, &num_sb, &flag);
    check("NUM_SB collapses onto the assertion",
          MPI_SUCCESS == rc && flag && 8 == *num_sb);
    rc = MPI_Win_get_attr(win, MPI_WIN_NOTIFICATION_NUM_UB, &num_ub, &flag);
    check("NUM_UB collapses onto the assertion",
          MPI_SUCCESS == rc && flag && 8 == *num_ub);

    rc = MPI_Win_get_num_notify(win, (rank + 1) % nprocs, &num);
    check("a peer attached the asserted number of counters",
          MPI_SUCCESS == rc && 8 == num);

    rc = MPI_Win_set_num_notify(win, MPI_INFO_NULL, 9);
    check("Win_set_num_notify refuses to exceed the assertion",
          MPI_ERR_ARG == rc);

    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    int *base = NULL;
    MPI_Win win = MPI_WIN_NULL;
    int total, num, rc;

    /* Must be set before MPI_Init: component selection happens there. */
    setenv("OMPI_MCA_osc", "sm", 1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* An error creating a window is raised on the communicator, not on the
     * window that does not exist yet, so the skips below are only reachable
     * if the communicator returns rather than aborts. */
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    win_elems = (nprocs > WIN_COUNT) ? nprocs : WIN_COUNT;

    if (nprocs < 2) {
        if (0 == rank) {
            fprintf(stderr, "win_notify_multi needs at least 2 ranks "
                            "(mpirun --np 4 ./win_notify_multi)\n");
        }
        MPI_Finalize();
        return 77;
    }

    /* Nothing below means anything on a component without notified RMA. */
    rc = MPI_Win_allocate_shared(win_elems * sizeof(int), sizeof(int),
                                 MPI_INFO_NULL, MPI_COMM_WORLD, &base, &win);
    if (MPI_SUCCESS != rc) {
        if (0 == rank) {
            fprintf(stderr, "could not create a shared window: all ranks "
                            "must be on one node\n");
        }
        MPI_Finalize();
        return 77;
    }
    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
    rc = MPI_Win_get_num_notify(win, 0, &num);
    MPI_Win_free(&win);
    if (MPI_ERR_UNSUPPORTED_OPERATION == rc) {
        if (0 == rank) {
            fprintf(stderr, "the selected osc component does not implement "
                            "notified RMA; skipping\n");
        }
        MPI_Finalize();
        return 77;
    }

    test_ring_notify();
    test_collective_growth();
    test_asymmetric_counts();
    test_info_and_attributes();

    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (0 == rank) {
        if (0 == total) {
            printf("win_notify_multi: PASSED on %d ranks\n", nprocs);
        } else {
            printf("win_notify_multi: FAILED on %d ranks (%d failure(s))\n",
                   nprocs, total);
        }
    }

    MPI_Finalize();
    return total ? 1 : 0;
}
