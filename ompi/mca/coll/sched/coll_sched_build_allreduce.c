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
 * Ring allreduce (reduce-scatter + allgather).
 *
 * Phase 1 – reduce-scatter (n-1 steps):
 *   step k: rank r sends chunk (r-k+n)%n, receives into temp[0],
 *            then reduces temp[0] into chunk (r-k+n-1)%n
 *
 * Phase 2 – allgather (n-1 steps):
 *   step k: rank r sends chunk (r-k+1+n)%n, receives directly into
 *            chunk (r-k+n)%n (no reduction)
 *
 * After reduce-scatter rank r owns the fully-reduced result for chunk
 * (r+1)%n.  After allgather every rank holds all chunks.
 *
 * Buffer layout: rbuf is divided into n equal chunks (num_slots=n).
 * Temp buf 0: 1 unit slot (= ceil(count/n) elements) – receive scratch.
 */
ompi_coll_sched_t *
ompi_coll_sched_build_allreduce_ring(int rank, int n)
{
    /* 3 ops per reduce-scatter step (send+recv+reduce), 2 per allgather step */
    int total_steps = 2 * (n - 1);
    ompi_coll_sched_t *sched = ompi_coll_sched_alloc(total_steps);
    if (NULL == sched) {
        return NULL;
    }

    /* Temp buf 0: one chunk, for receiving into during reduce-scatter */
    if (OMPI_ERR_OUT_OF_RESOURCE == ompi_coll_sched_add_temp_buf(sched, false)) {
        ompi_coll_sched_free(sched);
        return NULL;
    }

    int right = (rank + 1) % n;
    int left  = (rank - 1 + n) % n;

    /* ── reduce-scatter ─────────────────────────────────────────────────── */
    for (int k = 0; k < n - 1; k++) {
        int send_slot   = (rank - k + n) % n;
        int reduce_slot = (rank - k + n - 1) % n;

        /* 3 ops: send, recv, reduce3 */
        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, k, 3, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        /* Step 0: send original data from sbuf; steps 1+: send the slot that
         * was accumulated into rbuf by the previous step's REDUCE3. */
        int send_buf_id = (k == 0) ? OMPI_COLL_SCHED_BUF_SEND : OMPI_COLL_SCHED_BUF_RECV;
        ompi_coll_sched_op_send(sched, k, 0, 0, right,
                                 ompi_coll_sched_bufref_slot(send_buf_id, send_slot, n));
        ompi_coll_sched_op_recv(sched, k, 1, 0, left,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)));
        /* REDUCE3: dst = op(received_temp, sbuf[reduce_slot])
         * Writes directly into rbuf without requiring a pre-copy of sbuf. */
        ompi_coll_sched_op_reduce3(sched, k, 2,
                                    ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)),
                                    ompi_coll_sched_bufref_slot(OMPI_COLL_SCHED_BUF_SEND,
                                                                 reduce_slot, n),
                                    ompi_coll_sched_bufref_slot(OMPI_COLL_SCHED_BUF_RECV,
                                                                 reduce_slot, n));
    }

    /* ── allgather ──────────────────────────────────────────────────────── */
    for (int k = 0; k < n - 1; k++) {
        int step      = (n - 1) + k;
        int send_slot = (rank - k + 1 + n) % n;
        int recv_slot = (rank - k + n) % n;

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 2, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_send(sched, step, 0, 0, right,
                                 ompi_coll_sched_bufref_slot(OMPI_COLL_SCHED_BUF_RECV,
                                                              send_slot, n));
        ompi_coll_sched_op_recv(sched, step, 1, 0, left,
                                 ompi_coll_sched_bufref_slot(OMPI_COLL_SCHED_BUF_RECV,
                                                              recv_slot, n));
    }

    return sched;
}

/*
 * Recursive-doubling allreduce.
 *
 * Requires n to be a power of two.  In each of log2(n) steps, each
 * rank exchanges its ENTIRE buffer with the peer at distance 2^step and
 * reduces the received data into its own buffer.
 *
 * Temp buf 0: full-size (count elements) – receive scratch.
 */
ompi_coll_sched_t *
ompi_coll_sched_build_allreduce_recursivedoubling(int rank, int n)
{
    /* Verify power-of-two */
    if (n < 2 || (n & (n - 1)) != 0) {
        return NULL;
    }

    int num_steps = 0;
    for (int m = n; m > 1; m >>= 1) {
        num_steps++;
    }

    ompi_coll_sched_t *sched = ompi_coll_sched_alloc(num_steps);
    if (NULL == sched) {
        return NULL;
    }

    /* Temp buf 0: full size, to receive the peer's entire buffer */
    if (OMPI_ERR_OUT_OF_RESOURCE == ompi_coll_sched_add_temp_buf(sched, true)) {
        ompi_coll_sched_free(sched);
        return NULL;
    }

    for (int step = 0; step < num_steps; step++) {
        int peer = rank ^ (1 << step);

        /* 3 ops: send, recv into temp, reduce temp into rbuf */
        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 3, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        /* Step 0: send original sbuf; steps 1+: send accumulated rbuf. */
        int send_buf_id = (step == 0) ? OMPI_COLL_SCHED_BUF_SEND : OMPI_COLL_SCHED_BUF_RECV;
        ompi_coll_sched_op_send(sched, step, 0, 0, peer,
                                 ompi_coll_sched_bufref_whole(send_buf_id));
        ompi_coll_sched_op_recv(sched, step, 1, 0, peer,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)));
        if (step == 0) {
            /* REDUCE3: rbuf = op(received_temp, sbuf) — initialises rbuf */
            ompi_coll_sched_op_reduce3(sched, step, 2,
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)),
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_SEND),
                                        ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        } else {
            /* REDUCE2: rbuf = op(received_temp, rbuf) */
            ompi_coll_sched_op_reduce(sched, step, 2,
                                       ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)),
                                       ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        }
    }

    return sched;
}
