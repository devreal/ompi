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
#include "opal/mca/allocator/basic/allocator_basic.h"
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
 * Tracks a single GPU segment returned by opal_accelerator.mem_alloc so it
 * can be released on cleanup.  The basic allocator never calls seg_free during
 * normal operation (only compact/finalize would, and compact is a no-op), so
 * we keep our own list instead of relying on it.
 */
struct opal_accel_alloc_seg_t {
    opal_list_item_t super;
    void *ptr;
};
typedef struct opal_accel_alloc_seg_t opal_accel_alloc_seg_t;
OBJ_CLASS_INSTANCE(opal_accel_alloc_seg_t, opal_list_item_t, NULL, NULL);

typedef struct {
    int dev_id;
    opal_list_t segs; /* every GPU segment allocated via seg_alloc */
} opal_accel_alloc_ctx_t;

/*
 * seg_alloc is called (under the basic allocator's internal lock) whenever the
 * free list has no block large enough.  Record each new GPU segment so it can
 * be released on cleanup.
 */
static void *opal_accel_seg_alloc(void *ctx, size_t *size)
{
    opal_accel_alloc_ctx_t *ac = (opal_accel_alloc_ctx_t *) ctx;
    opal_accel_alloc_seg_t *seg;
    void *ptr = NULL;

    if (OPAL_SUCCESS != opal_accelerator.mem_alloc(ac->dev_id, &ptr, *size)) {
        return NULL;
    }

    seg = OBJ_NEW(opal_accel_alloc_seg_t);
    if (OPAL_LIKELY(NULL != seg)) {
        seg->ptr = ptr;
        opal_list_append(&ac->segs, &seg->super);
    }
    return ptr;
}

/* seg_free is wired into the allocator API but never invoked during normal
 * operation (basic allocator compact is a no-op).  Cleanup is handled
 * explicitly in opal_accelerator_base_frame_close via the segs list. */
static void opal_accel_seg_free(void *ctx, void *seg)
{
    (void) ctx;
    (void) seg;
}

mca_allocator_base_module_t *
opal_accelerator_base_get_device_allocator(int dev_id)
{
    mca_allocator_base_module_t *alloc;
    opal_accel_alloc_ctx_t *ctx;

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
        ctx = (opal_accel_alloc_ctx_t *) malloc(sizeof(*ctx));
        if (NULL == ctx) {
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        ctx->dev_id = dev_id;
        OBJ_CONSTRUCT(&ctx->segs, opal_list_t);
        /*
         * Use the basic (first-fit + coalescing) allocator rather than the
         * bucket allocator.  When a large block is freed it can be split to
         * serve a smaller future request, and adjacent free blocks are merged
         * back together, giving good reuse across the varying scratch-buffer
         * sizes produced by collective algorithms.  GPU segments are retained
         * in the free list for the lifetime of the process; the GPU driver
         * reclaims device memory on context teardown.
         */
        alloc = mca_allocator_basic_component_init(true,
                                                   opal_accel_seg_alloc,
                                                   opal_accel_seg_free,
                                                   ctx);
        if (NULL == alloc) {
            free(ctx);
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        opal_accel_device_allocators[dev_id] = alloc;
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
                opal_accel_alloc_ctx_t *ctx =
                    (opal_accel_alloc_ctx_t *) opal_accel_device_allocators[i]->alc_context;
                opal_accel_alloc_seg_t *seg;

                /* Release all GPU segments tracked in seg_alloc before the
                 * basic allocator frees its internal structures. */
                while (NULL != (seg = (opal_accel_alloc_seg_t *)
                                      opal_list_remove_first(&ctx->segs))) {
                    opal_accelerator.mem_release(ctx->dev_id, seg->ptr);
                    OBJ_RELEASE(seg);
                }
                OBJ_DESTRUCT(&ctx->segs);

                opal_accel_device_allocators[i]->alc_finalize(opal_accel_device_allocators[i]);
                free(ctx);
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
