/*
 * Copyright (C) 2001-2017 Mellanox Technologies Ltd. ALL RIGHTS RESERVED.
 * Copyright (c) 2019-2020 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef COMMON_UCX_WPOOL_H
#define COMMON_UCX_WPOOL_H

#include "opal_config.h"

#include "common_ucx.h"
#include <stdint.h>
#include <string.h>

#include <ucp/api/ucp.h>

#include "opal/class/opal_list.h"
#include "opal/include/opal/constants.h"
#include "opal/mca/mca.h"
#include "opal/mca/threads/tsd.h"
#include "opal/runtime/opal_progress.h"
#include "opal/util/output.h"

BEGIN_C_DECLS

/* fordward declaration */
typedef struct opal_common_ucx_winfo opal_common_ucx_winfo_t;

/* Worker pool is a global object that that is allocated per component or can be
 * shared between multiple compatible components.
 * The lifetime of this object is normally equal to the lifetime of a component[s].
 * It is expected to be initialized in MPI_Init and finalized in MPI_Finalize.
 */
typedef struct {
    /* Ref counting & locking*/
    int refcnt;
    opal_mutex_t mutex;

    /* UCX data */
    ucp_context_h ucp_ctx;
    opal_common_ucx_winfo_t *dflt_winfo;
    ucp_address_t *recv_waddr;
    size_t recv_waddr_len;

    /* Bookkeeping information */
    opal_list_t idle_workers;
    opal_list_t active_workers;
} opal_common_ucx_wpool_t;

/* Worker Pool Context (wpctx) is an object that is comprised of a set of UCP
 * workers that are considered as one logical communication entity.
 * One UCP worker per "active" thread is used.
 * Thread is considered "active" if it performs communication operations on this
 * Wpool context.
 * A lifetime of this object is dynamic and determined by the application
 * (the object is created and destroyed with corresponding functions).
 * Context is bound to a particular Worker Pool object.
 */
typedef struct {
    opal_object_t super;
    opal_mutex_t mutex;

    /* the reference to a Worker pool this context belongs to*/
    opal_common_ucx_wpool_t *wpool;

    /* A list of context records
     * We need to keep a track of allocated context records so
     * that we can free them at the end if thread fails to release context record */
    opal_list_t ctx_records;

    /* Thread-local key to allow each thread to have
     * local information associated with this wpctx */
    opal_tsd_tracked_key_t tls_key;

    /* UCX addressing information */
    char *recv_worker_addrs;
    int *recv_worker_displs;
    size_t comm_size;
} opal_common_ucx_ctx_t;

OBJ_CLASS_DECLARATION(opal_common_ucx_ctx_t);

struct opal_common_ucx_ep_rkey {
    opal_list_item_t super;
    //ucp_ep_h ep;
    opal_common_ucx_winfo_t *winfo;
    ucp_rkey_h rkey;
};

typedef struct opal_common_ucx_ep_rkey opal_common_ucx_ep_rkey_t;
OBJ_CLASS_DECLARATION(opal_common_ucx_ep_rkey_t);

/* Worker Pool memory (wpmem) is an object that represents a remotely accessible
 * distributed memory.
 * It has dynamic lifetime (created and destroyed by corresponding functions).
 * It depends on particular Wpool context.
 * Currently OSC is using one context per MPI Window, though in future it will
 * be possible to have one context for multiple windows.
 */
typedef struct opal_common_ucx_wpmem {
    opal_mutex_t mutex;

    /* reference context to which memory region belongs */
    opal_common_ucx_ctx_t *ctx;

    /* UCX memory handler */
    ucp_mem_h memh;
    char *mem_addrs;
    int *mem_displs; // NULL if this is a memhandle window

    /* connection information for memhandle windows, opal_common_ucx_ep_rkey_t elements */
    opal_list_t thread_rkey_list;
#if 0
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    opal_common_ucx_winfo_t *winfo;
#endif // 0

    /* A list of mem records
     * We need to kepp trakc o fallocated memory records so that we can free them at the end
     * if a thread fails to release the memory record */
    //opal_list_t mem_records;

    /* TLS item that allows each thread to
     * store endpoints and rkey arrays
     * for faster access */
    opal_tsd_tracked_key_t tls_key;
} opal_common_ucx_wpmem_t;

/* The structure that wraps UCP worker and holds the state that is required
 * for its use.
 * The structure is allocated along with UCP worker on demand and is being held
 * in the Worker Pool lists (either active or idle).
 * One wpmem is intended per shared memory segment (i.e. MPI Window).
 */
struct opal_common_ucx_winfo {
    opal_list_item_t super;
    opal_recursive_mutex_t mutex;
    ucp_worker_h worker;
    ucp_ep_h *endpoints;
    size_t comm_size;
    short *inflight_ops;
    short global_inflight_ops;
    bool has_dflt_worker;
    ucs_status_ptr_t inflight_req;
};
OBJ_CLASS_DECLARATION(opal_common_ucx_winfo_t);

typedef void (*opal_common_ucx_user_req_handler_t)(void *request);

/* A fast-path structure that gathers all pointers that are required to
 * perform RMA operation
 * wpmem's mem_tls_key holds the pointer to this structure
 */
typedef struct {
    void *ext_req;
    opal_common_ucx_user_req_handler_t ext_cb;
    opal_common_ucx_winfo_t *winfo;
} opal_common_ucx_request_t;

typedef enum { OPAL_COMMON_UCX_PUT, OPAL_COMMON_UCX_GET } opal_common_ucx_op_t;

typedef enum {
    OPAL_COMMON_UCX_SCOPE_EP,
    OPAL_COMMON_UCX_SCOPE_WORKER
} opal_common_ucx_flush_scope_t;

typedef enum {
    OPAL_COMMON_UCX_FLUSH_NB,
    OPAL_COMMON_UCX_FLUSH_B,
    OPAL_COMMON_UCX_FLUSH_NB_PREFERRED
} opal_common_ucx_flush_type_t;

typedef enum {
    OPAL_COMMON_UCX_MEM_ALLOCATE_MAP,
    OPAL_COMMON_UCX_MEM_MAP
} opal_common_ucx_mem_type_t;

typedef struct {
    opal_list_item_t super;
    opal_common_ucx_ctx_t *gctx;
    opal_common_ucx_winfo_t *winfo;
} _ctx_record_t;
OBJ_CLASS_DECLARATION(_ctx_record_t);

typedef struct {
    opal_list_item_t super;
    opal_common_ucx_wpmem_t *gmem;
    opal_common_ucx_winfo_t *winfo;
    ucp_rkey_h *rkeys;
    ucp_rkey_h  rkey; // used if rkeys is NULL
    _ctx_record_t *ctx_rec;
} _mem_record_t;
OBJ_CLASS_DECLARATION(_mem_record_t);

typedef int (*opal_common_ucx_exchange_func_t)(void *my_info, size_t my_info_len, char **recv_info,
                                               int **disps, void *metadata);

/* Manage Worker Pool (wpool) */
OPAL_DECLSPEC opal_common_ucx_wpool_t *opal_common_ucx_wpool_allocate(void);
OPAL_DECLSPEC void opal_common_ucx_wpool_free(opal_common_ucx_wpool_t *wpool);
OPAL_DECLSPEC int opal_common_ucx_wpool_init(opal_common_ucx_wpool_t *wpool, int proc_world_size,
                                             bool enable_mt);
OPAL_DECLSPEC void opal_common_ucx_wpool_finalize(opal_common_ucx_wpool_t *wpool);
OPAL_DECLSPEC int opal_common_ucx_wpool_progress(opal_common_ucx_wpool_t *wpool);

/* Manage Communication context */
OPAL_DECLSPEC int opal_common_ucx_wpctx_create(opal_common_ucx_wpool_t *wpool, int comm_size,
                                               opal_common_ucx_exchange_func_t exchange_func,
                                               void *exchange_metadata,
                                               opal_common_ucx_ctx_t **ctx_ptr);
OPAL_DECLSPEC void opal_common_ucx_wpctx_release(opal_common_ucx_ctx_t *ctx);

/* request init / completion */
OPAL_DECLSPEC void opal_common_ucx_req_init(void *request);
OPAL_DECLSPEC void opal_common_ucx_req_completion(void *request, ucs_status_t status);

/* Managing thread local storage */
OPAL_DECLSPEC int opal_common_ucx_tlocal_fetch_spath(opal_common_ucx_wpmem_t *mem, int target);

OPAL_DECLSPEC int opal_common_ucx_ctx_tlocal_fetch_spath(opal_common_ucx_wpmem_t *mem, int target);

OPAL_DECLSPEC int opal_common_ucx_ctx_set_dflt_worker(ucp_worker_h dflt_worker);

static inline int
opal_common_ucx_tlocal_fetch(opal_common_ucx_wpmem_t *mem, int target,
                                ucp_ep_h *_ep, ucp_rkey_h *_rkey,
                                opal_common_ucx_winfo_t **_winfo)
{
    int rc = OPAL_SUCCESS;

    if (NULL == mem->mem_displs) {
        _ctx_record_t *ctx_rec = NULL;
        rc = opal_tsd_tracked_key_get(&mem->ctx->tls_key, (void**)&ctx_rec);
        if (OPAL_SUCCESS != rc) {
            return rc;
        }
        bool is_ready = (NULL != ctx_rec) && NULL != ctx_rec->winfo->endpoints[target];
        if (!is_ready) {
            rc = opal_common_ucx_ctx_tlocal_fetch_spath(mem, target);
            if (OPAL_SUCCESS != rc) {
                return rc;
            }
            rc = opal_tsd_tracked_key_get(&mem->ctx->tls_key, (void**)&ctx_rec);
            if (OPAL_SUCCESS != rc) {
                return rc;
            }
        }
        opal_common_ucx_winfo_t *winfo = ctx_rec->winfo;
        ucp_ep_h ep = winfo->endpoints[target];

        /* memhandle windows carry a list of rkeys for each thread */
        opal_mutex_lock(&mem->mutex);
        opal_common_ucx_ep_rkey_t *pair;
        ucp_rkey_h rkey = NULL;
        /* see if the rkey exists already */
        OPAL_LIST_FOREACH(pair, &mem->thread_rkey_list, opal_common_ucx_ep_rkey_t) {
            if (pair->winfo == winfo) {
                rkey = pair->rkey;
                break;
            }
        }
        if (OPAL_UNLIKELY(NULL == rkey)) {
            /* need to unpack the key */
            pair = OBJ_NEW(opal_common_ucx_ep_rkey_t);
            pair->winfo = winfo;
            ucs_status_t status;

            opal_mutex_lock(&winfo->mutex);

            /* TODO: calling UCP functions here breaks the encapsulation but we don't care for now */
            status = ucp_ep_rkey_unpack(ep, mem->mem_addrs, &pair->rkey);
            //opal_mutex_unlock(&mem_rec->winfo->mutex);
            if (status != UCS_OK) {
                MCA_COMMON_UCX_VERBOSE(1, "ucp_ep_rkey_unpack failed: %d", status);
                return OPAL_ERROR;
            }
            opal_mutex_unlock(&winfo->mutex);

            rkey = pair->rkey;
            opal_list_prepend(&mem->thread_rkey_list, &pair->super);
        }
        opal_mutex_unlock(&mem->mutex);

        *_winfo = winfo;
        *_rkey = rkey;
        *_ep = ep;
    } else {
        _mem_record_t *mem_rec = NULL;
        int is_ready;
        /* First check the fast-path */
        rc = opal_tsd_tracked_key_get(&mem->tls_key, (void**)&mem_rec);
        if (OPAL_SUCCESS != rc) {
            return rc;
        }
        is_ready = mem_rec && (mem_rec->winfo->endpoints[target]) &&
                (NULL != mem_rec->rkey || (NULL != mem_rec->rkeys && NULL != mem_rec->rkeys[target]));
        MCA_COMMON_UCX_ASSERT((NULL == mem_rec) || (NULL != mem_rec->winfo));
        if (OPAL_UNLIKELY(!is_ready)) {
            rc = opal_common_ucx_tlocal_fetch_spath(mem, target);
            if (OPAL_SUCCESS != rc) {
                return rc;
            }
            rc = opal_tsd_tracked_key_get(&mem->tls_key, (void**)&mem_rec);
            if (OPAL_SUCCESS != rc) {
                return rc;
            }
        }
        MCA_COMMON_UCX_ASSERT(NULL != mem_rec);
        MCA_COMMON_UCX_ASSERT(NULL != mem_rec->winfo);
        MCA_COMMON_UCX_ASSERT(NULL != mem_rec->winfo->endpoints[target]);
        MCA_COMMON_UCX_ASSERT(NULL != mem_rec->rkeys[target]);
        *_winfo = mem_rec->winfo;
        *_ep = mem_rec->winfo->endpoints[target];
        *_rkey = NULL != mem_rec->rkey ? mem_rec->rkey : mem_rec->rkeys[target];
    }
    return OPAL_SUCCESS;
}

/* Manage & operations on the Memory registrations */
OPAL_DECLSPEC int opal_common_ucx_wpmem_create(opal_common_ucx_ctx_t *ctx, void **mem_base,
                                               size_t mem_size, opal_common_ucx_mem_type_t mem_type,
                                               opal_common_ucx_exchange_func_t exchange_func,
                                               void *exchange_metadata, char **my_mem_addr,
                                               int *my_mem_addr_size,
                                               opal_common_ucx_wpmem_t **mem_ptr);
OPAL_DECLSPEC void opal_common_ucx_wpmem_free(opal_common_ucx_wpmem_t *mem);

OPAL_DECLSPEC int opal_common_ucx_wpmem_flush(opal_common_ucx_wpmem_t *mem,
                                              opal_common_ucx_flush_scope_t scope, int target);
OPAL_DECLSPEC int opal_common_ucx_wpmem_fence(opal_common_ucx_wpmem_t *mem);

OPAL_DECLSPEC int opal_common_ucx_winfo_flush(opal_common_ucx_winfo_t *winfo, int target,
                                              opal_common_ucx_flush_type_t type,
                                              opal_common_ucx_flush_scope_t scope,
                                              ucs_status_ptr_t *req_ptr);

static inline int opal_common_ucx_wait_request_mt(ucs_status_ptr_t request, const char *msg)
{
    ucs_status_t status;
    int ctr = 0, ret = 0;
    opal_common_ucx_winfo_t *winfo;

    /* check for request completed or failed */
    if (OPAL_LIKELY(UCS_OK == request)) {
        return OPAL_SUCCESS;
    } else if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(request))) {
        MCA_COMMON_UCX_VERBOSE(1, "%s failed: %d, %s", msg ? msg : __func__,
                               UCS_PTR_STATUS(request), ucs_status_string(UCS_PTR_STATUS(request)));
        return OPAL_ERROR;
    }

    winfo = ((opal_common_ucx_request_t *) request)->winfo;
    assert(winfo != NULL);

    do {
        ctr = opal_common_ucx.progress_iterations;
        opal_mutex_lock(&winfo->mutex);
        do {
            ret = ucp_worker_progress(winfo->worker);
            status = opal_common_ucx_request_status(request);
            if (status != UCS_INPROGRESS) {
                ucp_request_free(request);
                if (OPAL_UNLIKELY(UCS_OK != status)) {
                    MCA_COMMON_UCX_VERBOSE(1, "%s failed: %d, %s", msg ? msg : __func__,
                                           UCS_PTR_STATUS(request),
                                           ucs_status_string(UCS_PTR_STATUS(request)));
                    opal_mutex_unlock(&winfo->mutex);
                    return OPAL_ERROR;
                }
                break;
            }
            ctr--;
        } while (ctr > 0 && ret > 0 && status == UCS_INPROGRESS);
        opal_mutex_unlock(&winfo->mutex);
        opal_progress();
    } while (status == UCS_INPROGRESS);

    return OPAL_SUCCESS;
}

static inline int _periodical_flush_nb(opal_common_ucx_wpmem_t *mem, opal_common_ucx_winfo_t *winfo,
                                       int target)
{
    int rc = OPAL_SUCCESS;

    winfo->inflight_ops[target]++;
    winfo->global_inflight_ops++;

    if (OPAL_UNLIKELY(winfo->inflight_ops[target] >= MCA_COMMON_UCX_PER_TARGET_OPS_THRESHOLD)
        || OPAL_UNLIKELY(winfo->global_inflight_ops >= MCA_COMMON_UCX_GLOBAL_OPS_THRESHOLD)) {
        opal_common_ucx_flush_scope_t scope;

        if (winfo->inflight_req != UCS_OK) {
            rc = opal_common_ucx_wait_request_mt(winfo->inflight_req, "opal_common_ucx_flush_nb");
            if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
                MCA_COMMON_UCX_VERBOSE(1, "opal_common_ucx_wait_request failed: %d", rc);
                return rc;
            }
            winfo->inflight_req = UCS_OK;
        }

        if (winfo->global_inflight_ops >= MCA_COMMON_UCX_GLOBAL_OPS_THRESHOLD) {
            scope = OPAL_COMMON_UCX_SCOPE_WORKER;
            winfo->global_inflight_ops = 0;
            memset(winfo->inflight_ops, 0, winfo->comm_size * sizeof(short));
        } else {
            scope = OPAL_COMMON_UCX_SCOPE_EP;
            winfo->global_inflight_ops -= winfo->inflight_ops[target];
            winfo->inflight_ops[target] = 0;
        }

        rc = opal_common_ucx_winfo_flush(winfo, target, OPAL_COMMON_UCX_FLUSH_NB_PREFERRED, scope,
                                         &winfo->inflight_req);
        if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
            MCA_COMMON_UCX_VERBOSE(1, "opal_common_ucx_flush failed: %d", rc);
            return rc;
        }
    } else if (OPAL_UNLIKELY(winfo->inflight_req != UCS_OK)) {
        int ret;
        do {
            ret = ucp_worker_progress(winfo->worker);
        } while (ret);
    }
    return rc;
}

static inline int opal_common_ucx_wpmem_putget(opal_common_ucx_wpmem_t *mem,
                                               opal_common_ucx_op_t op, int target, void *buffer,
                                               size_t len, uint64_t rem_addr)
{
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    ucs_status_t status;
    opal_common_ucx_winfo_t *winfo;
    int rc = OPAL_SUCCESS;
    char *called_func = "";

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "tlocal_fetch failed: %d", rc);
        return rc;
    }

    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    switch (op) {
    case OPAL_COMMON_UCX_PUT:
        status = ucp_put_nbi(ep, buffer, len, rem_addr, rkey);
        called_func = "ucp_put_nbi";
        break;
    case OPAL_COMMON_UCX_GET:
        status = ucp_get_nbi(ep, buffer, len, rem_addr, rkey);
        called_func = "ucp_get_nbi";
        break;
    }

    if (OPAL_UNLIKELY(status != UCS_OK && status != UCS_INPROGRESS)) {
        MCA_COMMON_UCX_ERROR("%s failed: %d", called_func, status);
        rc = OPAL_ERROR;
        goto out;
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

out:
    opal_mutex_unlock(&winfo->mutex);

    return rc;
}

static inline int opal_common_ucx_wpmem_cmpswp(opal_common_ucx_wpmem_t *mem, uint64_t compare,
                                               uint64_t value, int target, void *buffer, size_t len,
                                               uint64_t rem_addr)
{
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    opal_common_ucx_winfo_t *winfo = NULL;
    ucs_status_t status;
    int rc = OPAL_SUCCESS;

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("opal_common_ucx_tlocal_fetch failed: %d", rc);
        return rc;
    }

    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    status = opal_common_ucx_atomic_cswap(ep, compare, value, buffer, len, rem_addr, rkey,
                                          winfo->worker);
    if (OPAL_UNLIKELY(status != UCS_OK)) {
        MCA_COMMON_UCX_ERROR("opal_common_ucx_atomic_cswap failed: %d", status);
        rc = OPAL_ERROR;
        goto out;
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

out:
    opal_mutex_unlock(&winfo->mutex);

    return rc;
}

static inline int opal_common_ucx_wpmem_cmpswp_nb(opal_common_ucx_wpmem_t *mem, uint64_t compare,
                                                  uint64_t value, int target, void *buffer,
                                                  size_t len, uint64_t rem_addr,
                                                  opal_common_ucx_user_req_handler_t user_req_cb,
                                                  void *user_req_ptr)
{
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    opal_common_ucx_winfo_t *winfo = NULL;
    opal_common_ucx_request_t *req;
    int rc = OPAL_SUCCESS;

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("opal_common_ucx_tlocal_fetch failed: %d", rc);
        return rc;
    }

    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    req = opal_common_ucx_atomic_cswap_nb(ep, compare, value, buffer, len, rem_addr, rkey,
                                          opal_common_ucx_req_completion, winfo->worker);

    if (UCS_PTR_IS_PTR(req)) {
        req->ext_req = user_req_ptr;
        req->ext_cb = user_req_cb;
        req->winfo = winfo;
    } else {
        if (user_req_cb != NULL) {
            (*user_req_cb)(user_req_ptr);
        }
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

    opal_mutex_unlock(&winfo->mutex);

    return rc;
}

static inline int opal_common_ucx_wpmem_post(opal_common_ucx_wpmem_t *mem,
                                             ucp_atomic_post_op_t opcode, uint64_t value,
                                             int target, size_t len, uint64_t rem_addr)
{
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    opal_common_ucx_winfo_t *winfo = NULL;
    ucs_status_t status;
    int rc = OPAL_SUCCESS;

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("tlocal_fetch failed: %d", rc);
        return rc;
    }

    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    status = ucp_atomic_post(ep, opcode, value, len, rem_addr, rkey);
    if (OPAL_UNLIKELY(status != UCS_OK)) {
        MCA_COMMON_UCX_ERROR("ucp_atomic_post failed: %d", status);
        rc = OPAL_ERROR;
        goto out;
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

out:
    opal_mutex_unlock(&winfo->mutex);
    return rc;
}

static inline int opal_common_ucx_wpmem_fetch(opal_common_ucx_wpmem_t *mem,
                                              ucp_atomic_fetch_op_t opcode, uint64_t value,
                                              int target, void *buffer, size_t len,
                                              uint64_t rem_addr)
{
    ucp_ep_h ep = NULL;
    ucp_rkey_h rkey = NULL;
    opal_common_ucx_winfo_t *winfo = NULL;
    ucs_status_t status;
    int rc = OPAL_SUCCESS;

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("tlocal_fetch failed: %d", rc);
        return rc;
    }

    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    status = opal_common_ucx_atomic_fetch(ep, opcode, value, buffer, len, rem_addr, rkey,
                                          winfo->worker);
    if (OPAL_UNLIKELY(status != UCS_OK)) {
        MCA_COMMON_UCX_ERROR("ucp_atomic_cswap64 failed: %d", status);
        rc = OPAL_ERROR;
        goto out;
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

out:
    opal_mutex_unlock(&winfo->mutex);

    return rc;
}

static inline int opal_common_ucx_wpmem_fetch_nb(opal_common_ucx_wpmem_t *mem,
                                                 ucp_atomic_fetch_op_t opcode, uint64_t value,
                                                 int target, void *buffer, size_t len,
                                                 uint64_t rem_addr,
                                                 opal_common_ucx_user_req_handler_t user_req_cb,
                                                 void *user_req_ptr)
{
    ucp_ep_h ep = NULL;
    ucp_rkey_h rkey = NULL;
    opal_common_ucx_winfo_t *winfo = NULL;
    int rc = OPAL_SUCCESS;
    opal_common_ucx_request_t *req;

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("tlocal_fetch failed: %d", rc);
        return rc;
    }
    /* Perform the operation */
    opal_mutex_lock(&winfo->mutex);
    req = opal_common_ucx_atomic_fetch_nb(ep, opcode, value, buffer, len, rem_addr, rkey,
                                          opal_common_ucx_req_completion, winfo->worker);
    if (UCS_PTR_IS_PTR(req)) {
        req->ext_req = user_req_ptr;
        req->ext_cb = user_req_cb;
        req->winfo = winfo;
    } else {
        if (user_req_cb != NULL) {
            (*user_req_cb)(user_req_ptr);
        }
    }

    rc = _periodical_flush_nb(mem, winfo, target);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_VERBOSE(1, "_incr_and_check_inflight_ops failed: %d", rc);
    }

    opal_mutex_unlock(&winfo->mutex);

    return rc;
}

int _comm_ucx_wpmem_map(opal_common_ucx_wpool_t *wpool,
                        void **base, size_t size, ucp_mem_h *memh_ptr,
                        opal_common_ucx_mem_type_t mem_type);

END_C_DECLS

#endif // COMMON_UCX_WPOOL_H
