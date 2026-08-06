/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include "btl_ofi.h"

/**
 * Hardware notification counters.
 *
 * A notification is a libfabric counter (fi_cntr) bound to a memory region
 * with FI_REMOTE_WRITE | FI_REMOTE_READ. Once a region has been registered
 * this way the adapter increments the counter every time a remote operation
 * on that region completes here, without any involvement of this process's
 * CPU and without the origin having to issue a second operation to announce
 * that the data has landed.
 *
 * The counter counts operations on one memory region, so a consumer that
 * needs several independent counters over the same buffer obtains them by
 * registering the buffer once per counter and publishing a different rkey to
 * origins for each. These registrations deliberately bypass the rcache: it
 * keys on the address range and would hand back the same registration every
 * time, which is the opposite of what is wanted here.
 *
 * This requires the provider to have been opened with FI_RMA_EVENT, which is
 * requested as an optional capability during component init.
 */

struct mca_btl_base_notification_t {
    struct fid_mr *mr;
    struct fid_cntr *cntr;
    mca_btl_base_registration_handle_t handle;
};

struct mca_btl_base_notification_t *
mca_btl_ofi_register_notification(mca_btl_base_module_t *btl, void *base, size_t size,
                                  uint32_t flags, mca_btl_base_registration_handle_t **handle)
{
    /* the same access flags as an ordinary registration. the counter binding
     * below, not the access mask, is what decides which operations count. */
    static uint64_t access_flags = FI_REMOTE_WRITE | FI_REMOTE_READ | FI_READ | FI_WRITE;
    mca_btl_ofi_module_t *ofi_btl = (mca_btl_ofi_module_t *) btl;
    struct mca_btl_base_notification_t *notification;
    struct fi_cntr_attr cntr_attr = {0};
    struct fi_mr_attr attr = {0};
    struct iovec iov = {0};
    int rc;

    (void) flags;

    notification = calloc(1, sizeof(*notification));
    if (OPAL_UNLIKELY(NULL == notification)) {
        return NULL;
    }

    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    cntr_attr.wait_obj = FI_WAIT_UNSPEC;

    rc = fi_cntr_open(ofi_btl->domain, &cntr_attr, &notification->cntr, NULL);
    if (FI_SUCCESS != rc) {
        BTL_VERBOSE(("%s failed fi_cntr_open with err=%s", ofi_btl->linux_device_name,
                     fi_strerror(-rc)));
        notification->cntr = NULL;
        goto fail;
    }

    iov.iov_base = base;
    iov.iov_len = size;
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    attr.access = access_flags;
    attr.offset = 0;
    attr.context = NULL;
    attr.requested_key = (uint64_t) (uintptr_t) notification;

    rc = fi_mr_regattr(ofi_btl->domain, &attr, 0, &notification->mr);
    if (FI_SUCCESS != rc) {
        BTL_VERBOSE(("%s failed fi_mr_regattr with err=%s", ofi_btl->linux_device_name,
                     fi_strerror(-rc)));
        notification->mr = NULL;
        goto fail;
    }

    /* Bind the counter for both remote writes and remote reads so that a
     * consumer can notify on either. We deliberately do not fall back to
     * counting writes alone: a counter that silently misses remote reads
     * would leave a get-with-notification permanently unnoticed at the
     * target, which is worse than reporting no support at all and letting
     * the consumer use its own fallback path. */
    rc = fi_mr_bind(notification->mr, &notification->cntr->fid,
                    FI_REMOTE_WRITE | FI_REMOTE_READ);
    if (FI_SUCCESS != rc) {
        BTL_VERBOSE(("%s failed to bind notification counter with err=%s",
                     ofi_btl->linux_device_name, fi_strerror(-rc)));
        goto fail;
    }

    if (ofi_btl->use_fi_mr_bind) {
        rc = fi_mr_bind(notification->mr, &ofi_btl->ofi_endpoint->fid, 0ULL);
        if (FI_SUCCESS != rc) {
            BTL_VERBOSE(("%s failed to bind notification mr to endpoint with err=%s",
                         ofi_btl->linux_device_name, fi_strerror(-rc)));
            goto fail;
        }
    }

    /* Unlike an ordinary registration this is required unconditionally: an mr
     * that has anything bound to it is created disabled and does not service
     * remote operations until enabled. */
    rc = fi_mr_enable(notification->mr);
    if (FI_SUCCESS != rc) {
        BTL_VERBOSE(("%s failed fi_mr_enable with err=%s", ofi_btl->linux_device_name,
                     fi_strerror(-rc)));
        goto fail;
    }

    notification->handle.rkey = fi_mr_key(notification->mr);
    notification->handle.desc = fi_mr_desc(notification->mr);

    /* In case the provider doesn't support FI_MR_VIRT_ADDR, remote addresses
     * are expressed as a distance from the base registered address. */
    if (ofi_btl->use_virt_addr) {
        notification->handle.base_addr = 0;
    } else {
        notification->handle.base_addr = base;
    }

    *handle = &notification->handle;

    return notification;

fail:
    if (NULL != notification->mr) {
        fi_close(&notification->mr->fid);
    }
    if (NULL != notification->cntr) {
        fi_close(&notification->cntr->fid);
    }
    free(notification);

    return NULL;
}

int mca_btl_ofi_deregister_notification(mca_btl_base_module_t *btl,
                                        struct mca_btl_base_notification_t *notification)
{
    int ret = OPAL_SUCCESS;
    int rc;

    if (OPAL_UNLIKELY(NULL == notification)) {
        return OPAL_SUCCESS;
    }

    /* close the mr first, it references the counter */
    if (NULL != notification->mr) {
        rc = fi_close(&notification->mr->fid);
        if (FI_SUCCESS != rc) {
            BTL_ERROR(("%s: error unpinning notification memory: %s", __func__, fi_strerror(-rc)));
            ret = OPAL_ERROR;
        }
    }

    if (NULL != notification->cntr) {
        rc = fi_close(&notification->cntr->fid);
        if (FI_SUCCESS != rc) {
            BTL_ERROR(("%s: error closing notification counter: %s", __func__, fi_strerror(-rc)));
            ret = OPAL_ERROR;
        }
    }

    free(notification);

    return ret;
}

int mca_btl_ofi_notification_read(mca_btl_base_module_t *btl,
                                  struct mca_btl_base_notification_t *notification,
                                  uint64_t *value)
{
    uint64_t errors;

    errors = fi_cntr_readerr(notification->cntr);
    if (OPAL_UNLIKELY(0 != errors)) {
        BTL_ERROR(("notification counter reported %llu error(s)", (unsigned long long) errors));
        return OPAL_ERROR;
    }

    *value = fi_cntr_read(notification->cntr);

    return OPAL_SUCCESS;
}

int mca_btl_ofi_notification_wait(mca_btl_base_module_t *btl,
                                  struct mca_btl_base_notification_t *notification,
                                  uint64_t threshold, int timeout)
{
    int rc;

    rc = fi_cntr_wait(notification->cntr, threshold, timeout);
    if (FI_SUCCESS == rc) {
        return OPAL_SUCCESS;
    }

    if (-FI_ETIMEDOUT == rc) {
        return OPAL_ERR_TIMEOUT;
    }

    BTL_VERBOSE(("fi_cntr_wait failed with err=%s", fi_strerror(-rc)));

    return OPAL_ERROR;
}
