/*
 * Copyright (c) 2014      Intel, Inc. All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include "opal/constants.h"
#include "opal/mca/accelerator/base/base.h"
#include "opal/mca/base/base.h"
#include "opal/mca/mca.h"
#include "opal/mca/allocator/allocator.h"
#include "opal/mca/threads/mutex.h"

/*
 * The following file was created by configure.  It contains extern
 * components and the definition of an array of pointers to each
 * module's public mca_base_module_t struct.
 */
#include "opal/mca/accelerator/base/static-components.h"


opal_accelerator_base_module_t opal_accelerator = {0};
opal_accelerator_base_component_t opal_accelerator_base_selected_component = {{0}};

/* Per-device allocator pool — allocated lazily to num_devices on first use. */
static mca_allocator_base_module_t **opal_accel_device_allocators = NULL;
static int opal_accel_num_devices = 0;
static opal_mutex_t opal_accel_alloc_lock = OPAL_MUTEX_STATIC_INIT;

/*
 * Device buffers handed out by this allocator may be read/written directly
 * by a persistent GPU kernel (see ompi_op_gpu_session_t) as well as by the
 * host, so they are backed by whatever opal_accelerator.mem_alloc() returns
 * for the device (e.g. plain cuMemAlloc/hipMalloc memory, which the host
 * CPU cannot dereference). That rules out the generic "basic" allocator:
 * it pools/coalesces free blocks by writing a size header directly into the
 * bytes it hands out, which segfaults the moment that memory is device-only.
 *
 * Collective scratch buffers are typically requested at a handful of
 * distinct sizes, repeatedly, over the lifetime of a run (e.g. the same
 * MPI_Reduce count called in a loop) — device mem_alloc/mem_release round
 * trips on every call are a real latency hit. So this module caches freed
 * blocks for reuse instead of releasing them back to the device immediately.
 * All bookkeeping (which blocks are free, and the size of an outstanding
 * allocation) lives in host-side opal_list_t's keyed by pointer — never in
 * the device buffer itself, which is the bug the "basic" allocator hit.
 * Free blocks are only released back to the device at alc_finalize().
 */
typedef struct {
    opal_list_item_t super;
    void *ptr;
    size_t size;
} opal_accel_device_seg_t;
OBJ_CLASS_INSTANCE(opal_accel_device_seg_t, opal_list_item_t, NULL, NULL);

typedef struct {
    mca_allocator_base_module_t super;
    int dev_id;
    opal_mutex_t lock;
    opal_list_t free_segs;        /* cached blocks available for reuse */
    opal_list_t outstanding_segs; /* blocks currently handed out to callers */
} opal_accel_device_allocator_t;

static void *opal_accel_device_alloc(mca_allocator_base_module_t *base, size_t size, size_t align)
{
    opal_accel_device_allocator_t *m = (opal_accel_device_allocator_t *) base;
    opal_accel_device_seg_t *seg, *best = NULL;
    void *ptr;

    OPAL_THREAD_LOCK(&m->lock);

    /* Best-fit scan of the free list: the smallest cached block that is
     * still large enough, so we don't habitually hand out a much bigger
     * block than requested. The free list is expected to stay small (a
     * handful of distinct scratch-buffer sizes), so a linear scan is fine. */
    OPAL_LIST_FOREACH (seg, &m->free_segs, opal_accel_device_seg_t) {
        if (seg->size >= size && (NULL == best || seg->size < best->size)) {
            best = seg;
        }
    }

    if (NULL != best) {
        /* Reuse the cached block as-is (best->size already >= size). */
        opal_list_remove_item(&m->free_segs, &best->super);
        ptr = best->ptr;
    } else {
        if (OPAL_SUCCESS != opal_accelerator.mem_alloc(m->dev_id, &ptr, size)) {
            OPAL_THREAD_UNLOCK(&m->lock);
            return NULL;
        }
        best = OBJ_NEW(opal_accel_device_seg_t);
        if (NULL == best) {
            opal_accelerator.mem_release(m->dev_id, ptr);
            OPAL_THREAD_UNLOCK(&m->lock);
            return NULL;
        }
        best->ptr  = ptr;
        best->size = size;
    }

    opal_list_append(&m->outstanding_segs, &best->super);
    OPAL_THREAD_UNLOCK(&m->lock);
    return ptr;
}

static void *opal_accel_device_realloc(mca_allocator_base_module_t *base, void *ptr, size_t size)
{
    /* Device memory cannot be resized in place; unused by current callers. */
    return NULL;
}

static void opal_accel_device_free(mca_allocator_base_module_t *base, void *ptr)
{
    opal_accel_device_allocator_t *m = (opal_accel_device_allocator_t *) base;
    opal_accel_device_seg_t *seg;

    if (NULL == ptr) {
        return;
    }

    OPAL_THREAD_LOCK(&m->lock);
    OPAL_LIST_FOREACH (seg, &m->outstanding_segs, opal_accel_device_seg_t) {
        if (seg->ptr == ptr) {
            opal_list_remove_item(&m->outstanding_segs, &seg->super);
            opal_list_append(&m->free_segs, &seg->super);
            OPAL_THREAD_UNLOCK(&m->lock);
            return;
        }
    }
    OPAL_THREAD_UNLOCK(&m->lock);

    /* Unknown pointer (double free, or foreign pointer) -- release it
     * directly to the device rather than silently dropping it. */
    opal_accelerator.mem_release(m->dev_id, ptr);
}

static int opal_accel_device_compact(mca_allocator_base_module_t *base)
{
    return OPAL_SUCCESS;
}

static int opal_accel_device_finalize(mca_allocator_base_module_t *base)
{
    opal_accel_device_allocator_t *m = (opal_accel_device_allocator_t *) base;
    opal_accel_device_seg_t *seg;

    while (NULL != (seg = (opal_accel_device_seg_t *) opal_list_remove_first(&m->free_segs))) {
        opal_accelerator.mem_release(m->dev_id, seg->ptr);
        OBJ_RELEASE(seg);
    }
    /* Any segments still in outstanding_segs were never freed by their
     * caller; leave them for the GPU driver to reclaim on context teardown,
     * same as the previous allocator's documented behavior. */
    while (NULL != (seg = (opal_accel_device_seg_t *) opal_list_remove_first(&m->outstanding_segs))) {
        OBJ_RELEASE(seg);
    }
    OBJ_DESTRUCT(&m->free_segs);
    OBJ_DESTRUCT(&m->outstanding_segs);
    OBJ_DESTRUCT(&m->lock);
    free(m);
    return OPAL_SUCCESS;
}

mca_allocator_base_module_t *
opal_accelerator_base_get_device_allocator(int dev_id)
{
    if (dev_id < 0 || NULL == opal_accelerator.mem_alloc) {
        return NULL;
    }

    /* Fast path: array already sized and slot already filled. */
    if (NULL != opal_accel_device_allocators
        && dev_id < opal_accel_num_devices
        && NULL != opal_accel_device_allocators[dev_id]) {
        return opal_accel_device_allocators[dev_id];
    }

    OPAL_THREAD_LOCK(&opal_accel_alloc_lock);

    /* Lazily allocate the per-device array on first call. */
    if (NULL == opal_accel_device_allocators) {
        int num_devices = 0;
        if (OPAL_SUCCESS != opal_accelerator.num_devices(&num_devices) || num_devices <= 0) {
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        opal_accel_device_allocators = calloc(num_devices,
                                              sizeof(*opal_accel_device_allocators));
        if (NULL == opal_accel_device_allocators) {
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        opal_accel_num_devices = num_devices;
    }

    if (dev_id >= opal_accel_num_devices) {
        OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
        return NULL;
    }

    if (NULL == opal_accel_device_allocators[dev_id]) {
        opal_accel_device_allocator_t *m =
            (opal_accel_device_allocator_t *) malloc(sizeof(*m));
        if (NULL == m) {
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        m->dev_id             = dev_id;
        OBJ_CONSTRUCT(&m->lock, opal_mutex_t);
        OBJ_CONSTRUCT(&m->free_segs, opal_list_t);
        OBJ_CONSTRUCT(&m->outstanding_segs, opal_list_t);
        m->super.alc_alloc    = opal_accel_device_alloc;
        m->super.alc_realloc  = opal_accel_device_realloc;
        m->super.alc_free     = opal_accel_device_free;
        m->super.alc_compact  = opal_accel_device_compact;
        m->super.alc_finalize = opal_accel_device_finalize;
        m->super.alc_context  = NULL;
        opal_accel_device_allocators[dev_id] = &m->super;
    }

    OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
    return opal_accel_device_allocators[dev_id];
}

static int opal_accelerator_base_frame_register(mca_base_register_flag_t flags)
{
    return OPAL_SUCCESS;
}

static int opal_accelerator_base_frame_close(void)
{
    if (NULL != opal_accel_device_allocators) {
        for (int i = 0; i < opal_accel_num_devices; i++) {
            if (NULL != opal_accel_device_allocators[i]) {
                /* alc_finalize frees the opal_accel_device_allocator_t itself;
                 * the individual device buffers it handed out are released by
                 * their owners via COLL_SESSION_FREE/alc_free before this
                 * point, and the GPU driver reclaims device memory on
                 * context teardown regardless. */
                opal_accel_device_allocators[i]->alc_finalize(opal_accel_device_allocators[i]);
                opal_accel_device_allocators[i] = NULL;
            }
        }
        free(opal_accel_device_allocators);
        opal_accel_device_allocators = NULL;
        opal_accel_num_devices = 0;
    }
    return mca_base_framework_components_close(&opal_accelerator_base_framework, NULL);
}

static int opal_accelerator_base_frame_open(mca_base_open_flag_t flags)
{
    return mca_base_framework_components_open(&opal_accelerator_base_framework, flags);
}

OBJ_CLASS_INSTANCE(
    opal_accelerator_stream_t,
    opal_object_t,
    NULL,
    NULL);

OBJ_CLASS_INSTANCE(
    opal_accelerator_event_t,
    opal_object_t,
    NULL,
    NULL);

OBJ_CLASS_INSTANCE(
    opal_accelerator_ipc_handle_t,
    opal_object_t,
    NULL,
    NULL);

OBJ_CLASS_INSTANCE(
    opal_accelerator_ipc_event_handle_t,
    opal_object_t,
    NULL,
    NULL);


MCA_BASE_FRAMEWORK_DECLARE(opal, accelerator, "OPAL Accelerator Framework",
                           opal_accelerator_base_frame_register, opal_accelerator_base_frame_open,
                           opal_accelerator_base_frame_close, mca_accelerator_base_static_components,
                           0);
