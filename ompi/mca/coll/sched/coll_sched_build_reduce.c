/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include <stdlib.h>
#include "coll_sched.h"

/*
 * Binomial-tree reduce.
 *
 * The dispatch function ensures rbuf holds the rank's own contribution
 * before calling the executor (for non-root ranks this means rbuf points
 * to a caller-allocated scratch buffer, not the MPI rbuf argument).
 *
 * Virtual rank: vrank = (rank - root + n) % n; vrank 0 is the tree root.
 *
 * For each mask = 1, 2, 4, … while mask < n:
 *   if (vrank & mask): send rbuf to parent ((vrank-mask+n)%n mapped back to
 *                      real rank), then stop participating.
 *   else if (vrank + mask < n): receive from child ((vrank+mask)%n mapped),
 *                               reduce into rbuf.
 *
 * Non-root intermediate nodes: recv-steps first, then one send step.
 * Root (vrank=0): recv-steps only.
 * Pure leaves (lowest set bit = mask on first check): one send step only.
 *
 * Temp buf 0 (full-size): receive scratch for each incoming reduction.
 */
ompi_coll_sched_t *
ompi_coll_sched_build_reduce_binomial(int rank, int n, int root)
{
    if (n == 1) {
        /* Nothing to communicate; dispatch copies sbuf→rbuf. */
        ompi_coll_sched_t *sched = ompi_coll_sched_alloc(0);
        if (sched) {
            sched->num_comm_slots = 1;
        }
        return sched;
    }

    int vrank = (rank - root + n) % n;

    /* Count how many recv steps and whether we send */
    int num_recv = 0;
    int send_mask = -1; /* mask at which this rank sends; -1 = never */

    for (int mask = 1; mask < n; mask <<= 1) {
        if (vrank & mask) {
            send_mask = mask;
            break; /* send once, then stop */
        }
        if (vrank + mask < n) {
            num_recv++;
        }
    }

    int num_steps = num_recv + (send_mask >= 0 ? 1 : 0);

    /* Leaf that sends immediately without prior receives: no recv steps, 1 send */
    /* Root: recv steps only */

    ompi_coll_sched_t *sched = ompi_coll_sched_alloc(num_steps);
    if (NULL == sched) {
        return NULL;
    }

    /* Temp buf 0: full-size receive scratch (needed only if we receive) */
    if (num_recv > 0) {
        if (OMPI_ERR_OUT_OF_RESOURCE == ompi_coll_sched_add_temp_buf(sched, true)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
    }

    int step = 0;

    /* Receive steps (bottom-up from children) */
    for (int mask = 1; mask < n && step < num_recv; mask <<= 1) {
        if (vrank & mask) {
            break; /* this rank sends at this mask; no more receives */
        }
        if (vrank + mask >= n) {
            continue; /* no child at this level */
        }

        int child_vrank = vrank + mask;
        int child_real  = (child_vrank + root) % n;

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 2, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_recv(sched, step, 0, 0, child_real,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)));
        if (step == 0) {
            /* First recv: REDUCE3 to initialise rbuf from own sbuf + received.
             * Avoids the pre-copy of sbuf → rbuf in the dispatch function. */
            ompi_coll_sched_op_reduce3(sched, step, 1,
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)),
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_SEND),
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        } else {
            ompi_coll_sched_op_reduce(sched, step, 1,
                                       ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)),
                                       ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        }
        step++;
    }

    /* Send step (if this rank is not the root) */
    if (send_mask >= 0) {
        int parent_vrank = vrank - send_mask;
        int parent_real  = (parent_vrank + root) % n;

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 1, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        /* Leaf ranks (num_recv==0) never wrote into BUF_RECV; send original
         * data directly from BUF_SEND.  Internal non-root ranks send their
         * accumulated partial result from BUF_RECV. */
        int send_buf_id = (num_recv == 0) ? OMPI_COLL_SCHED_BUF_SEND
                                          : OMPI_COLL_SCHED_BUF_RECV;
        ompi_coll_sched_op_send(sched, step, 0, 0, parent_real,
                                 ompi_coll_sched_bufref_whole(send_buf_id));
    }

    return sched;
}
