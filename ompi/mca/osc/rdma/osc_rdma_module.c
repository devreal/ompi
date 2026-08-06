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

    ompi_osc_rdma_notify_counters_destroy (module);

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

/**
 * @brief release every btl notification counter attached to this window
 */
void ompi_osc_rdma_notify_counters_destroy (ompi_osc_rdma_module_t *module)
{
    mca_btl_base_module_t *btl = module->use_accelerated_btl ? module->accelerated_btl : NULL;

    for (int i = 0 ; i < OMPI_OSC_RDMA_NOTIFY_MAX ; ++i) {
        if (NULL != module->notify_handles[i]) {
            if (NULL != btl) {
                (void) btl->btl_deregister_notification (btl, module->notify_handles[i]);
            }
            module->notify_handles[i] = NULL;
        }
    }

    free (module->notify_peer_handles);
    module->notify_peer_handles = NULL;
    module->notify_stride = 0;
    module->use_notify_counters = false;
}

/**
 * @brief can this window notify through btl notification counters?
 *
 * Dynamic windows are excluded because a counter is bound to one registration
 * and the attached regions of a dynamic window come and go.
 */
static bool ompi_osc_rdma_notify_counters_available (ompi_osc_rdma_module_t *module)
{
    return module->use_accelerated_btl && MPI_WIN_FLAVOR_DYNAMIC != module->flavor
        && module->use_memory_registration && module->size
        && !!(module->accelerated_btl->btl_flags & MCA_BTL_FLAGS_NOTIFIED_RMA);
}

/**
 * @brief attach {my_count} notification counters to the window and publish their handles
 *
 * Each counter is a separate registration of the same window memory, so an
 * origin picks which counter to advance purely by choosing which registration
 * handle to target -- the adapter then counts the data movement itself and no
 * follow-up atomic (nor a flush to order it behind the data) is required.
 *
 * Every rank in the group runs the same sequence of collectives here, whether
 * or not its own registrations succeeded, so that a failure on one rank makes
 * the whole window fall back rather than deadlocking the rest.
 */
static int ompi_osc_rdma_notify_counters_setup (ompi_osc_rdma_module_t *module, int my_count, int max_count)
{
    mca_btl_base_module_t *btl = module->accelerated_btl;
    size_t handle_size = btl->btl_registration_handle_size;
    ompi_osc_rdma_region_t *region;
    unsigned char *my_handles = NULL;
    int registered_ok = 1;
    int all_ok = 0;
    int ret;

    /* a non-dynamic window always has exactly one region, which describes the
     * memory the window was created over */
    region = (ompi_osc_rdma_region_t *) module->state->regions;

    module->notify_handle_size = handle_size;
    module->notify_stride = max_count;

    module->notify_peer_handles = calloc ((size_t) ompi_comm_size (module->comm) * (size_t) max_count,
                                          handle_size);
    my_handles = calloc ((size_t) max_count, handle_size);

    if (NULL == module->notify_peer_handles || NULL == my_handles) {
        registered_ok = 0;
    }

    for (int i = 0 ; registered_ok && i < my_count ; ++i) {
        mca_btl_base_registration_handle_t *handle = NULL;

        module->notify_handles[i] = btl->btl_register_notification (btl,
                                                                    (void *) (intptr_t) region->base,
                                                                    (size_t) region->len,
                                                                    MCA_BTL_REG_FLAG_ACCESS_ANY, &handle);
        if (OPAL_UNLIKELY(NULL == module->notify_handles[i])) {
            /* leaving some indices counted by the adapter and others not would
             * be worse than not using counters at all */
            registered_ok = 0;
            break;
        }

        memcpy (my_handles + (size_t) i * handle_size, handle, handle_size);
    }

    ret = module->comm->c_coll->coll_allreduce (&registered_ok, &all_ok, 1, MPI_INT, MPI_MIN, module->comm,
                                                module->comm->c_coll->coll_allreduce_module);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        all_ok = 0;
    }

    if (!all_ok) {
        free (my_handles);
        ompi_osc_rdma_notify_counters_destroy (module);
        /* not an error: the caller notifies with atomics instead */
        return OMPI_SUCCESS;
    }

    ret = module->comm->c_coll->coll_allgather (my_handles, (int) ((size_t) max_count * handle_size), MPI_BYTE,
                                                module->notify_peer_handles,
                                                (int) ((size_t) max_count * handle_size), MPI_BYTE,
                                                module->comm, module->comm->c_coll->coll_allgather_module);
    free (my_handles);

    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        ompi_osc_rdma_notify_counters_destroy (module);
        return ret;
    }

    module->use_notify_counters = true;

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_INFO, "notifying through %d btl notification counter(s)", my_count);

    return OMPI_SUCCESS;
}

int ompi_osc_rdma_win_get_notify_value (struct ompi_win_t *win, int notify, OMPI_MPI_COUNT_TYPE *value)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    int my_rank = ompi_comm_rank (module->comm);

    OMPI_OSC_RDMA_CHECK_NOTIFY_IDX(module, notify, my_rank);

    if (module->use_notify_counters) {
        mca_btl_base_module_t *btl = module->accelerated_btl;
        uint64_t hardware;
        int ret;

        ret = btl->btl_notification_read (btl, module->notify_handles[notify], &hardware);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            return ret;
        }

        /* origins on this node reach the window with a direct copy that the
         * adapter never sees, so their notifications land in the state counter
         * instead. both count operations on this index, so the value the
         * standard asks for is their sum. */
        *value = (OMPI_MPI_COUNT_TYPE) (hardware - module->notify_reset_base[notify]
                                        + (uint64_t) ((volatile osc_rdma_counter_t *)
                                                      module->state->notify_counters)[notify]);
        /* ensure loads of the window data are not reordered before the counter read */
        opal_atomic_rmb ();

        return OMPI_SUCCESS;
    }

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

    if (module->use_notify_counters) {
        mca_btl_base_module_t *btl = module->accelerated_btl;
        uint64_t hardware;

        ret = btl->btl_notification_read (btl, module->notify_handles[notify], &hardware);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            return ret;
        }

        /* the adapter's counter is monotonic and cannot be zeroed, so record
         * where this reset happened and report differences from there. any
         * increment that lands between the read above and the store below is
         * simply attributed to the next interval, which is what an atomic
         * fetch-and-zero at the instant of the read would also have done. */
        *value = (OMPI_MPI_COUNT_TYPE) (hardware - module->notify_reset_base[notify]);
        module->notify_reset_base[notify] = hardware;

        /* same-node origins notify through the state counter, which only ever
         * sees CPU atomics in this mode, so it can be zeroed directly */
        old_value = module->state->notify_counters[notify];
        while (!ompi_osc_rdma_lock_compare_exchange ((osc_rdma_atomic_counter_t *) (module->state->notify_counters + notify),
                                                     &old_value, 0));

        *value += (OMPI_MPI_COUNT_TYPE) old_value;

        return OMPI_SUCCESS;
    }

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
    int max_count = 0, supported, all_supported = 0;
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
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        return ret;
    }

    /* A hardware counter cannot be zeroed, so rather than track a reset base
     * for a counter nobody has used yet, drop the existing registrations and
     * attach fresh ones. This call is collective and no access epoch is open,
     * so no origin can be holding a handle we are about to invalidate. */
    ompi_osc_rdma_notify_counters_destroy (module);
    memset (module->notify_reset_base, 0, sizeof (module->notify_reset_base));

    for (int i = 0 ; i < ompi_comm_size (module->comm) ; ++i) {
        if (module->notify_counts[i] > max_count) {
            max_count = module->notify_counts[i];
        }
    }

    if (0 == max_count) {
        return OMPI_SUCCESS;
    }

    /* whether counters can be used has to be agreed group-wide: an origin
     * chooses how to notify, and it must make the same choice its target made */
    supported = ompi_osc_rdma_notify_counters_available (module) ? 1 : 0;
    ret = module->comm->c_coll->coll_allreduce (&supported, &all_supported, 1, MPI_INT, MPI_MIN, module->comm,
                                                module->comm->c_coll->coll_allreduce_module);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        return ret;
    }

    if (all_supported) {
        /* falls back silently (leaving use_notify_counters clear) if the
         * adapter will not hand out counters everywhere */
        ret = ompi_osc_rdma_notify_counters_setup (module, my_count, max_count);
    }

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
