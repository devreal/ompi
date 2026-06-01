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
 * Binomial-tree broadcast – matches the ompi bmtree (coll_base_topo_build_bmtree).
 *
 * Virtual rank: vrank = (rank - root + n) % n.
 *
 * Tree structure for a node with vrank V:
 *   recv_mask  = highest power-of-two that is ≤ V  (highest set bit of V)
 *   parent     = V ^ recv_mask  (= V - recv_mask, since recv_mask is a set bit)
 *   send_start = next power-of-two above V  (= recv_mask << 1 for V>0, = 1 for V=0)
 *   children   = V ^ mask  for mask = send_start, 2*send_start, … while child < n
 *
 * Example n=8:
 *   V=0 (root)  : children [1,2,4]
 *   V=1         : parent=0, children [3,5]
 *   V=2         : parent=0, children [3,6]   -- WAIT, let me re-check
 *
 * Actually for n=8:
 *   V=0: send_start=1, children: 0^1=1, 0^2=2, 0^4=4  → [1,2,4]
 *   V=1: recv_mask=1, send_start=2, children: 1^2=3, 1^4=5  → [3,5]
 *   V=2: recv_mask=2, send_start=4, children: 2^4=6  → [6]
 *   V=3: recv_mask=2, send_start=4, children: 3^4=7  → [7]
 *   V=4: recv_mask=4, send_start=8 >= n=8  → []
 *   V=5: recv_mask=4, send_start=8 >= n=8  → []
 *   V=6: recv_mask=4, send_start=8 >= n=8  → []
 *   V=7: recv_mask=4, send_start=8 >= n=8  → []
 *
 * Note: recv_mask for V=2 is 2 (highest set bit of 2=0b10), parent=2^2=0 ✓
 *       recv_mask for V=3 is 2 (highest set bit of 3=0b11), parent=3^2=1 ✓
 *       The LOWEST set bit determines the parent, not the highest.
 *
 * Wait, let me re-check V=3 parent:
 *   opal_next_poweroftwo(3) = 4, 4>>1 = 2 → parent = 3^2 = 1 ✓
 *   Also: highest set bit of 3 = 2 → recv_mask = 2, parent = 3^2 = 1 ✓
 *
 * Both "highest set bit" and "opal_next_poweroftwo(V)>>1" give the same recv_mask. ✓
 *
 * No temp buffers needed: all communication is directly to/from rbuf (BUF_RECV).
 */
ompi_coll_sched_t *
ompi_coll_sched_build_bcast_binomial(int rank, int n, int root)
{
    if (n == 1) {
        ompi_coll_sched_t *sched = ompi_coll_sched_alloc(0);
        if (sched) {
            sched->num_comm_slots = 1;
        }
        return sched;
    }

    int vrank = (rank - root + n) % n;

    /*
     * Compute recv_mask: highest power-of-two ≤ vrank.
     * For vrank=0 (root) there is no receive.
     */
    int recv_mask = 0;
    if (vrank > 0) {
        recv_mask = 1;
        while ((recv_mask << 1) <= vrank) {
            recv_mask <<= 1;
        }
    }

    /*
     * send_start: first mask at which this rank sends to a child.
     *   root  → send_start = 1
     *   other → send_start = recv_mask << 1  (next bit above recv_mask)
     */
    int send_start = (vrank == 0) ? 1 : (recv_mask << 1);

    /* Count send steps: child = vrank ^ mask for mask = send_start, *2, *4, … */
    int num_sends = 0;
    for (int mask = send_start; vrank + mask < n; mask <<= 1) {
        num_sends++;
        if (mask <= 0 || mask >= n) break; /* guard against infinite loop / overflow */
    }

    int num_steps = (vrank != 0 ? 1 : 0) + num_sends;

    ompi_coll_sched_t *sched = ompi_coll_sched_alloc(num_steps);
    if (NULL == sched) {
        return NULL;
    }

    int step = 0;

    /* Receive step (non-root only): wait for data before forwarding */
    if (vrank != 0) {
        int parent_vrank = vrank ^ recv_mask; /* = vrank - recv_mask */
        int parent_real  = (parent_vrank + root) % n;

        if (OMPI_SUCCESS != ompi_coll_sched_step_init(sched, step, 1, true)) {
            ompi_coll_sched_free(sched);
            return NULL;
        }
        ompi_coll_sched_op_recv(sched, step, 0, 0, parent_real,
                                 ompi_coll_sched_bufref_whole(OMPI_COLL_SCHED_BUF_RECV));
        step++;
    }

    /* Send steps: propagate to children in ascending mask order */
    for (int mask = send_start; vrank + mask < n; mask <<= 1) {
        int child_vrank = vrank ^ mask; /* = vrank + mask since vrank & mask == 0 */
        int child_real  = (child_vrank + root) % n;

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
