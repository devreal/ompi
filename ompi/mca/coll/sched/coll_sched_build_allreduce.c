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

/*
 * Non-overlapping allreduce: binomial reduce to rank 0, then binomial bcast
 * from rank 0.  Matches ompi_coll_base_allreduce_intra_nonoverlapping.
 *
 * The virtual root is always rank 0, so vrank == rank throughout.
 *
 * Phase 1 – Reduce to rank 0 (same logic as build_reduce_binomial with root=0):
 *   For mask = 1, 2, 4, … < n:
 *     if (rank & mask): send to parent (rank - mask), stop.
 *     else if (rank + mask < n): recv from child (rank + mask) into TEMP(0),
 *                                reduce into BUF_RECV.
 *   First recv step uses REDUCE3(TEMP(0), BUF_SEND → BUF_RECV).
 *   Subsequent recv steps use REDUCE2(TEMP(0) → BUF_RECV).
 *   Leaf ranks (no recvs) send from BUF_SEND; internal non-root send from BUF_RECV.
 *
 * Phase 2 – Bcast from rank 0 (same logic as build_bcast_binomial with root=0):
 *   recv_mask = highest power-of-two ≤ rank (0 for root).
 *   send_start = (rank == 0) ? 1 : (recv_mask << 1).
 *   Non-root: one recv step from parent (rank ^ recv_mask).
 *   For each child mask = send_start, 2*send_start, …: one send step, BUF_RECV.
 *
 * Temp buffer TEMP(0) full-size: only allocated if rank has any reduce recv steps.
 */
ompi_coll_sched_t *
ompi_coll_sched_build_allreduce_nonoverlapping(int rank, int n)
{
    if (n == 1) {
        ompi_coll_sched_t *sched = ompi_coll_sched_alloc(0);
        if (sched) {
            sched->num_comm_slots = 1;
        }
        return sched;
    }

    /* ── Phase 1: Reduce to rank 0 ─────────────────────────────────────── */

    int reduce_recv = 0;   /* number of recv steps in reduce phase */
    int reduce_send = 0;   /* 1 if rank sends in reduce, 0 otherwise */

    for (int mask = 1; mask < n; mask <<= 1) {
        if (rank & mask) {
            reduce_send = 1;
            break;
        }
        if (rank + mask < n) {
            reduce_recv++;
        }
    }

    /* ── Phase 2: Bcast from rank 0 ────────────────────────────────────── */

    int bcast_recv_mask = 0;
    if (rank > 0) {
        bcast_recv_mask = 1;
        while ((bcast_recv_mask << 1) <= rank) {
            bcast_recv_mask <<= 1;
        }
    }
    int bcast_send_start = (rank == 0) ? 1 : (bcast_recv_mask << 1);

    int bcast_recv = (rank != 0) ? 1 : 0;
    int bcast_send = 0;
    for (int mask = bcast_send_start; rank + mask < n; mask <<= 1) {
        bcast_send++;
        if (mask <= 0 || mask >= n) break; /* guard overflow */
    }

    /* Total steps */
    int num_steps = reduce_recv + reduce_send + bcast_recv + bcast_send;

    ompi_coll_sched_t *sched = ompi_coll_sched_alloc(num_steps);
    if (NULL == sched) {
        return NULL;
    }
    sched->num_comm_slots = 1;

    /* Temp buf 0: full-size, needed only if rank receives during reduce */
    if (reduce_recv > 0) {
        if (OMPI_ERR_OUT_OF_RESOURCE == ompi_coll_sched_add_temp_buf(sched, true)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
    }

    int step = 0;

    /* ── Reduce recv steps ──────────────────────────────────────────────── */
    for (int mask = 1; mask < n && step < reduce_recv; mask <<= 1) {
        if (rank & mask) {
            break; /* this rank sends at this mask; no more recvs */
        }
        if (rank + mask >= n) {
            continue; /* no child at this level */
        }

        int child_real = rank + mask; /* vrank == rank since root=0 */

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 2, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_recv(sched, step, 0, 0, child_real,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_TEMP(0)));
        if (step == 0) {
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

    /* ── Reduce send step ───────────────────────────────────────────────── */
    if (reduce_send) {
        /* Find the send mask: lowest set bit of rank */
        int send_mask = 1;
        while (!(rank & send_mask)) {
            send_mask <<= 1;
        }
        int parent_real = rank - send_mask; /* vrank == rank since root=0 */

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 1, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        int send_buf_id = (reduce_recv == 0) ? OMPI_COLL_SCHED_BUF_SEND
                                             : OMPI_COLL_SCHED_BUF_RECV;
        ompi_coll_sched_op_send(sched, step, 0, 0, parent_real,
                                 ompi_coll_sched_bufref_whole(send_buf_id));
        step++;
    }

    /* ── Bcast recv step ────────────────────────────────────────────────── */
    if (bcast_recv) {
        int parent_real = rank ^ bcast_recv_mask; /* = rank - bcast_recv_mask */

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 1, true)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_recv(sched, step, 0, 0, parent_real,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        step++;
    }

    /* ── Bcast send steps ───────────────────────────────────────────────── */
    for (int mask = bcast_send_start; rank + mask < n; mask <<= 1) {
        int child_real = rank + mask; /* = rank ^ mask since rank & mask == 0 */

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 1, false)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_send(sched, step, 0, 0, child_real,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        step++;

        if (mask <= 0 || mask >= n) break; /* guard overflow */
    }

    return sched;
}
