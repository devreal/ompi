/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University.
 *                         All rights reserved.
 * Copyright (c) 2004-2005 The Trustees of the University of Tennessee.
 *                         All rights reserved.
 * Copyright (c) 2004-2020 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2016 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2012-2013 Sandia National Laboratories.  All rights reserved.
 * Copyright (c) 2017      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Google, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal/mca/mpool/base/base.h"

#include "osc_rdma.h"
#include "osc_rdma_lock.h"

#include "mpi.h"

int ompi_osc_module_add_peer (ompi_osc_rdma_module_t *module, ompi_osc_rdma_peer_t *peer)
{
    int ret = OMPI_SUCCESS;

    if (NULL == module->peer_array) {
        ret = opal_hash_table_set_value_uint32 (&module->peer_hash, peer->rank, (void *) peer);
    } else {
        module->peer_array[peer->rank] = peer;
    }

    return ret;
}

int ompi_osc_rdma_free(ompi_win_t *win)
{
    int ret = OMPI_SUCCESS;
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    ompi_osc_rdma_peer_t *peer;
    uint32_t key;
    void *node;

    if (NULL == module) {
        return OMPI_SUCCESS;
    }

    while (module->pending_ops) {
        ompi_osc_rdma_progress (module);
    }

    if (NULL != module->comm) {
        opal_output_verbose(1, ompi_osc_base_framework.framework_output,
                            "rdma component destroying window with id %s",
                            ompi_comm_print_cid(module->comm));

        /* finish with a barrier */
        if (ompi_group_size(win->w_group) > 1) {
            (void) module->comm->c_coll->coll_barrier (module->comm,
                                                      module->comm->c_coll->coll_barrier_module);
        }

        /* remove from component information */
        OPAL_THREAD_LOCK(&mca_osc_rdma_component.lock);
        opal_hash_table_remove_value_uint32(&mca_osc_rdma_component.modules,
                                            ompi_comm_get_local_cid(module->comm));
        OPAL_THREAD_UNLOCK(&mca_osc_rdma_component.lock);
    }

    win->w_osc_module = NULL;

    if (module->state) {
        int region_count = module->state->region_count & 0xffffffffL;
        if (NULL != module->dynamic_handles) {
            for (int i = 0 ; i < region_count ; ++i) {
                ompi_osc_rdma_handle_t *region_handle = module->dynamic_handles[i];
                ompi_osc_rdma_deregister (module, region_handle->btl_handle);
                OBJ_RELEASE(region_handle);
            }

            free (module->dynamic_handles);
        }
    }

    OBJ_DESTRUCT(&module->outstanding_locks);
    OBJ_DESTRUCT(&module->lock);
    OBJ_DESTRUCT(&module->peer_lock);
    OBJ_DESTRUCT(&module->all_sync);

    ompi_osc_rdma_deregister (module, module->state_handle);
    ompi_osc_rdma_deregister (module, module->base_handle);

    OPAL_LIST_DESTRUCT(&module->pending_posts);

    if (NULL != module->rdma_frag) {
        ompi_osc_rdma_deregister (module, module->rdma_frag->handle);
    }

    /* remove all cached peers */
    if (NULL == module->peer_array) {
        ret = opal_hash_table_get_first_key_uint32 (&module->peer_hash, &key, (void **) &peer, &node);
        while (OPAL_SUCCESS == ret) {
            OBJ_RELEASE(peer);
            ret = opal_hash_table_get_next_key_uint32 (&module->peer_hash, &key, (void **) &peer,
                                                       node, &node);
        }

        OBJ_DESTRUCT(&module->peer_hash);
    } else if (NULL != module->comm) {
        for (int i = 0 ; i < ompi_comm_size (module->comm) ; ++i) {
            if (NULL != module->peer_array[i]) {
                OBJ_RELEASE(module->peer_array[i]);
            }
        }
    }

    if (module->local_leaders && MPI_COMM_NULL != module->local_leaders) {
        ompi_comm_free (&module->local_leaders);
    }

    if (module->shared_comm && MPI_COMM_NULL != module->shared_comm) {
        ompi_comm_free (&module->shared_comm);
    }

    if (module->comm && MPI_COMM_NULL != module->comm) {
        ompi_comm_free (&module->comm);
    }

    if (module->segment_base) {
        opal_shmem_segment_detach (&module->seg_ds);
        module->segment_base = NULL;
    }

    free (module->peer_array);
    free (module->outstanding_lock_array);
    free (module->notify_counts);
    mca_mpool_base_default_module->mpool_free(mca_mpool_base_default_module,
                                              module->free_after);
    if (!module->use_accelerated_btl) {
        for (int i = 0 ; i < module->alternate_btl_count ; ++i) {
            OBJ_RELEASE(module->alternate_am_rdmas[i]);
        }
        free(module->alternate_am_rdmas);
    }
    free (module);

    return OMPI_SUCCESS;
}

/* ******************* notified communication ******************* */

int ompi_osc_rdma_win_get_notify_value (struct ompi_win_t *win, int notify, OMPI_MPI_COUNT_TYPE *value)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    int my_rank = ompi_comm_rank (module->comm);

    OMPI_OSC_RDMA_CHECK_NOTIFY_IDX(module, notify, my_rank);

    /* the standard's usage model is polling this function until a counter
     * reaches a threshold. progress the module so counter updates arrive even
     * when the btl requires target-side progress (emulated atomics) */
    ompi_osc_rdma_progress (module);

    *value = (OMPI_MPI_COUNT_TYPE) ((volatile osc_rdma_counter_t *) module->state->notify_counters)[notify];
    /* ensure loads of the window data are not reordered before the counter read */
    opal_atomic_rmb ();

    return OMPI_SUCCESS;
}

int ompi_osc_rdma_win_reset_notify_value (struct ompi_win_t *win, int notify, OMPI_MPI_COUNT_TYPE *value)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    ompi_osc_rdma_peer_t *my_peer = module->my_peer;
    int my_rank = ompi_comm_rank (module->comm);
    ompi_osc_rdma_lock_t old_value;
    int ret;

    OMPI_OSC_RDMA_CHECK_NOTIFY_IDX(module, notify, my_rank);

    /* the counter is incremented by remote origins with btl atomics, so the
     * read-and-zero must be atomic with respect to those. use a CPU atomic
     * only when the peer flags indicate it is safe to mix CPU and btl
     * atomics, otherwise loop over a btl compare-and-swap on our own state */
    if (ompi_osc_rdma_peer_local_state (my_peer)) {
        old_value = module->state->notify_counters[notify];
        while (!ompi_osc_rdma_lock_compare_exchange ((osc_rdma_atomic_counter_t *) (module->state->notify_counters + notify),
                                                     &old_value, 0));
    } else {
        uint64_t address = (uint64_t) (intptr_t) my_peer->state + offsetof (ompi_osc_rdma_state_t, notify_counters) +
            (uint64_t) notify * sizeof (osc_rdma_counter_t);
        ompi_osc_rdma_lock_t result;

        /* the local value is a guess at the current value. if another origin
         * updates the counter concurrently the compare-and-swap fails and
         * returns the new value to retry with */
        old_value = module->state->notify_counters[notify];

        do {
            ret = ompi_osc_rdma_lock_btl_cswap (module, my_peer, address, old_value, 0, &result);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
                return ret;
            }

            if (result == old_value) {
                break;
            }

            old_value = result;
        } while (1);
    }

    *value = (OMPI_MPI_COUNT_TYPE) old_value;

    return OMPI_SUCCESS;
}

int ompi_osc_rdma_win_set_num_notify (struct ompi_win_t *win, struct opal_info_t *info, int num_notifications)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    int my_rank = ompi_comm_rank (module->comm);
    int my_count, ret;

    (void) info; /* "mpi_assert_same_num_notifications" is an optimization hint only */

    if (OPAL_UNLIKELY(num_notifications < 0 || num_notifications > OMPI_OSC_RDMA_NOTIFY_MAX)) {
        return MPI_ERR_ARG;
    }

    /* it is erroneous to call MPI_WIN_SET_NUM_NOTIFY while an access epoch is open */
    if (OPAL_UNLIKELY(ompi_osc_rdma_access_epoch_active (module))) {
        return OMPI_ERR_RMA_SYNC;
    }

    if (NULL == module->notify_counts) {
        module->notify_counts = calloc (ompi_comm_size (module->comm), sizeof (int));
        if (OPAL_UNLIKELY(NULL == module->notify_counts)) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
    }

    /* the number of attached notification counters is never decreased (section 12.6.1) */
    if (num_notifications > module->notify_counts[my_rank]) {
        module->notify_counts[my_rank] = num_notifications;
    }

    /* all notification counters (existing and newly attached) are reset to zero by
     * this call. no access epoch is open so no btl atomics can be in flight on them */
    memset ((void *) module->state->notify_counters, 0, sizeof (module->state->notify_counters));
    opal_atomic_wmb ();

    /* publish every rank's attached count to the whole group so origins can validate
     * notification indices against the target's count. this allgather doubles as the
     * blocking, synchronizing collective required by the standard */
    my_count = module->notify_counts[my_rank];
    ret = module->comm->c_coll->coll_allgather (&my_count, 1, MPI_INT, module->notify_counts, 1, MPI_INT,
                                                module->comm, module->comm->c_coll->coll_allgather_module);

    return ret;
}

int ompi_osc_rdma_win_get_num_notify (struct ompi_win_t *win, int target_rank, int *num_notifications)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);

    if (OPAL_UNLIKELY(target_rank < 0 || target_rank >= ompi_comm_size (module->comm))) {
        return MPI_ERR_RANK;
    }

    *num_notifications = (NULL != module->notify_counts) ? module->notify_counts[target_rank] : 0;

    return OMPI_SUCCESS;
}
