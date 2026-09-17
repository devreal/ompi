/*
 * Copyright (c) 2026      Joseph Antony.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * Functional test for MPI Notified Communication (MPI draft section 12.6).
 *
 * Exercises MPI_Win_set_num_notify / MPI_Win_get_num_notify,
 * MPI_Put_notify, MPI_Get_notify, MPI_Win_get_notify_value and
 * MPI_Win_reset_notify_value against whichever osc component is
 * selected.  Run with at least 2 ranks; force a component with e.g.
 *
 *     mpirun -np 2 --mca osc ucx ./notified_rma
 *     mpirun -np 2 --mca osc sm  ./notified_rma
 *
 * The interesting property under test is the ordering guarantee: a
 * notification counter must only become visible at the target *after*
 * the payload of the notified operation has landed there.  The payload
 * is deliberately large so that it is carried by real RDMA rather than
 * an inlined/eager path, which is where a missing fence shows up.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#define NELEM       16384   /* 64 KB payload: big enough to force real RDMA */
#define NCOUNTERS   4
#define NITERS      64
#define TIMEOUT_SEC 30.0

static int rank, size;
static int failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "[%d] FAIL %s:%d: ", rank, __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            fflush(stderr);                                                 \
            failures++;                                                     \
        }                                                                   \
    } while (0)

#define CHECK_MPI(call, expected)                                           \
    do {                                                                    \
        int rc_ = (call);                                                   \
        int cls_ = MPI_SUCCESS;                                             \
        if (MPI_SUCCESS != rc_) {                                           \
            MPI_Error_class(rc_, &cls_);                                    \
        }                                                                   \
        CHECK(cls_ == (expected), "%s returned class %d, expected %d",      \
              #call, cls_, (expected));                                     \
    } while (0)

/* Poll a local notification counter until it reaches "target", or give
 * up after TIMEOUT_SEC.  A hang here means notifications are not being
 * delivered (or the counter is not being progressed). */
static int poll_counter(MPI_Win win, int idx, MPI_Count target, const char *what)
{
    double deadline = MPI_Wtime() + TIMEOUT_SEC;
    MPI_Count value = 0;

    while (value < target) {
        int rc = MPI_Win_get_notify_value(win, idx, &value);
        if (MPI_SUCCESS != rc) {
            fprintf(stderr, "[%d] FAIL %s: get_notify_value rc=%d\n", rank, what, rc);
            failures++;
            return -1;
        }
        if (MPI_Wtime() > deadline) {
            fprintf(stderr, "[%d] FAIL %s: timed out waiting for counter %d "
                    "to reach %lld (stuck at %lld)\n",
                    rank, what, idx, (long long) target, (long long) value);
            fflush(stderr);
            failures++;
            return -1;
        }
    }
    return 0;
}

static int expected_get_elem(int owner, int i)
{
    return owner * 1000003 + i;
}

static int expected_put_elem(int origin, int iter, int i)
{
    return (origin + 1) * 7919 + iter * 31 + i;
}

int main(int argc, char *argv[])
{
    MPI_Win win;
    int *winbuf = NULL;
    int *localbuf = NULL;
    int left, right;
    int i, iter, r;
    int num;
    MPI_Count value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (0 == rank) {
            fprintf(stderr, "This test requires at least 2 ranks\n");
        }
        MPI_Finalize();
        return 77; /* automake "skipped" */
    }

    left = (rank - 1 + size) % size;
    right = (rank + 1) % size;

    localbuf = malloc(NELEM * sizeof(int));
    if (NULL == localbuf) {
        fprintf(stderr, "[%d] malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Win_allocate(NELEM * sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &winbuf, &win);
    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);

    /* Seed each window with a rank-specific pattern for the Get test. */
    for (i = 0; i < NELEM; i++) {
        winbuf[i] = expected_get_elem(rank, i);
    }

    /* ---------------------------------------------------------------
     * Test 1: counter attachment.  Collective, must be called outside
     * any access epoch, and must never decrease the count.
     * --------------------------------------------------------------- */
    CHECK_MPI(MPI_Win_set_num_notify(win, MPI_INFO_NULL, NCOUNTERS), MPI_SUCCESS);

    for (r = 0; r < size; r++) {
        num = -1;
        CHECK_MPI(MPI_Win_get_num_notify(win, r, &num), MPI_SUCCESS);
        CHECK(NCOUNTERS == num, "get_num_notify(rank %d) = %d, expected %d",
              r, num, NCOUNTERS);
    }

    /* Counters must read zero immediately after attachment. */
    for (i = 0; i < NCOUNTERS; i++) {
        value = -1;
        CHECK_MPI(MPI_Win_get_notify_value(win, i, &value), MPI_SUCCESS);
        CHECK(0 == value, "counter %d = %lld after set_num_notify, expected 0",
              i, (long long) value);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Notified operations are only valid in a passive target epoch. */
    CHECK_MPI(MPI_Win_lock_all(0, win), MPI_SUCCESS);

    /* ---------------------------------------------------------------
     * Test 2: MPI_Get_notify.  Each rank fetches its right neighbour's
     * window; the counter is incremented at the *target* (12.6), so
     * every rank is notified exactly once on counter 1 by its left
     * neighbour.
     * --------------------------------------------------------------- */
    memset(localbuf, 0, NELEM * sizeof(int));
    CHECK_MPI(MPI_Get_notify(localbuf, NELEM, MPI_INT, right, 0, NELEM, MPI_INT,
                             1, win),
              MPI_SUCCESS);
    CHECK_MPI(MPI_Win_flush(right, win), MPI_SUCCESS);

    for (i = 0; i < NELEM; i++) {
        if (localbuf[i] != expected_get_elem(right, i)) {
            CHECK(0, "Get_notify data mismatch at %d: got %d, expected %d",
                  i, localbuf[i], expected_get_elem(right, i));
            break;
        }
    }

    if (0 == poll_counter(win, 1, 1, "Get_notify")) {
        value = -1;
        MPI_Win_get_notify_value(win, 1, &value);
        CHECK(1 == value, "counter 1 = %lld after one Get_notify, expected 1",
              (long long) value);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------------------------------------------------------
     * Test 3: MPI_Put_notify ordering.  Each iteration writes a fresh
     * pattern to the right neighbour and notifies counter 0; the target
     * polls the counter and then validates the payload.  If the counter
     * can become visible before the data lands, this reports a mismatch.
     * The barrier keeps iteration i+1 from overwriting the buffer the
     * target is still checking for iteration i.
     * --------------------------------------------------------------- */
    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < NELEM; i++) {
            localbuf[i] = expected_put_elem(rank, iter, i);
        }

        CHECK_MPI(MPI_Put_notify(localbuf, NELEM, MPI_INT, right, 0, NELEM,
                                 MPI_INT, 0, win),
                  MPI_SUCCESS);

        if (0 != poll_counter(win, 0, iter + 1, "Put_notify")) {
            break;
        }

        /* Data written by "left" must be fully visible now. */
        for (i = 0; i < NELEM; i++) {
            if (winbuf[i] != expected_put_elem(left, iter, i)) {
                CHECK(0, "Put_notify ordering violation, iter %d elem %d: "
                         "got %d, expected %d (counter fired before data landed)",
                      iter, i, winbuf[i], expected_put_elem(left, iter, i));
                iter = NITERS; /* stop after the first bad iteration */
                break;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------------------------------------------------------
     * Test 4: several notifications in flight against distinct
     * counters, with no per-operation synchronisation.
     * --------------------------------------------------------------- */
    for (i = 1; i < NCOUNTERS; i++) {
        CHECK_MPI(MPI_Put_notify(localbuf, 1, MPI_INT, right, 0, 1, MPI_INT,
                                 i, win),
                  MPI_SUCCESS);
    }
    for (i = 1; i < NCOUNTERS; i++) {
        /* counter 1 already carries the Get_notify from test 2 */
        poll_counter(win, i, (1 == i) ? 2 : 1, "burst Put_notify");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------------------------------------------------------
     * Test 5: reset is an atomic fetch-and-zero.
     * --------------------------------------------------------------- */
    value = -1;
    CHECK_MPI(MPI_Win_reset_notify_value(win, 0, &value), MPI_SUCCESS);
    CHECK(value == NITERS, "reset_notify_value(0) returned %lld, expected %d",
          (long long) value, NITERS);

    value = -1;
    CHECK_MPI(MPI_Win_get_notify_value(win, 0, &value), MPI_SUCCESS);
    CHECK(0 == value, "counter 0 = %lld after reset, expected 0",
          (long long) value);

    CHECK_MPI(MPI_Win_unlock_all(win), MPI_SUCCESS);

    /* ---------------------------------------------------------------
     * Test 6: error paths.  An out-of-range notification index must
     * raise MPI_ERR_RMA_NOTIFICATION rather than corrupting memory.
     * --------------------------------------------------------------- */
    CHECK_MPI(MPI_Win_get_notify_value(win, NCOUNTERS, &value),
              MPI_ERR_RMA_NOTIFICATION);
    CHECK_MPI(MPI_Win_get_notify_value(win, -1, &value),
              MPI_ERR_RMA_NOTIFICATION);
    CHECK_MPI(MPI_Win_reset_notify_value(win, NCOUNTERS, &value),
              MPI_ERR_RMA_NOTIFICATION);
    CHECK_MPI(MPI_Win_get_num_notify(win, size, &num), MPI_ERR_RANK);

    /* An out-of-range index on the data path must be caught too. */
    CHECK_MPI(MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win), MPI_SUCCESS);
    CHECK_MPI(MPI_Put_notify(localbuf, 1, MPI_INT, right, 0, 1, MPI_INT,
                             NCOUNTERS, win),
              MPI_ERR_RMA_NOTIFICATION);
    CHECK_MPI(MPI_Win_unlock(right, win), MPI_SUCCESS);

    /* Requesting more counters than the implementation supports must
     * fail cleanly.  All ranks pass the same value, so this stays
     * collectively consistent. */
    CHECK_MPI(MPI_Win_set_num_notify(win, MPI_INFO_NULL, 1 << 20), MPI_ERR_ARG);
    CHECK_MPI(MPI_Win_set_num_notify(win, MPI_INFO_NULL, -1), MPI_ERR_ARG);

    /* The count never decreases (12.6.1). */
    CHECK_MPI(MPI_Win_set_num_notify(win, MPI_INFO_NULL, 1), MPI_SUCCESS);
    CHECK_MPI(MPI_Win_get_num_notify(win, rank, &num), MPI_SUCCESS);
    CHECK(num >= NCOUNTERS, "get_num_notify = %d after shrinking request, "
          "expected >= %d", num, NCOUNTERS);

    MPI_Win_free(&win);
    free(localbuf);

    /* ---------------------------------------------------------------
     * Test 7: notified communication is not supported on dynamic
     * windows in this implementation; it must say so rather than crash.
     * --------------------------------------------------------------- */
    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
    CHECK_MPI(MPI_Win_set_num_notify(win, MPI_INFO_NULL, 1), MPI_ERR_RMA_FLAVOR);
    MPI_Win_free(&win);

    /* --------------------------------------------------------------- */
    {
        int total = 0;
        MPI_Reduce(&failures, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (0 == rank) {
            if (0 == total) {
                printf("notified_rma: PASSED (%d ranks)\n", size);
            } else {
                printf("notified_rma: FAILED (%d errors across %d ranks)\n",
                       total, size);
            }
            fflush(stdout);
        }
        MPI_Finalize();
        return (0 == total) ? 0 : 1;
    }
}
