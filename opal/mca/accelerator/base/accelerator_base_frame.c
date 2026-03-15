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
#include "opal/mca/allocator/bucket/allocator_bucket_alloc.h"
#include "opal/mca/threads/mutex.h"

/*
 * The following file was created by configure.  It contains extern
 * components and the definition of an array of pointers to each
 * module's public mca_base_module_t struct.
 */
#include "opal/mca/accelerator/base/static-components.h"


opal_accelerator_base_module_t opal_accelerator = {0};
opal_accelerator_base_component_t opal_accelerator_base_selected_component = {{0}};

#define OPAL_ACCELERATOR_MAX_DEVICES 16
#define OPAL_ACCELERATOR_ALLOC_NUM_BUCKETS 8

static mca_allocator_base_module_t *opal_accel_device_allocators[OPAL_ACCELERATOR_MAX_DEVICES];
static opal_mutex_t opal_accel_alloc_lock = OPAL_MUTEX_STATIC_INIT;

typedef struct {
    int dev_id;
} opal_accel_alloc_ctx_t;

static void *opal_accel_seg_alloc(void *ctx, size_t *size)
{
    opal_accel_alloc_ctx_t *ac = (opal_accel_alloc_ctx_t *)ctx;
    void *ptr = NULL;
    if (OPAL_SUCCESS != opal_accelerator.mem_alloc(ac->dev_id, &ptr, *size)) {
        return NULL;
    }
    return ptr;
}

static void opal_accel_seg_free(void *ctx, void *seg)
{
    opal_accel_alloc_ctx_t *ac = (opal_accel_alloc_ctx_t *)ctx;
    opal_accelerator.mem_release(ac->dev_id, seg);
}

mca_allocator_base_module_t *
opal_accelerator_base_get_device_allocator(int dev_id)
{
    mca_allocator_bucket_t *bucket;
    opal_accel_alloc_ctx_t *ctx;

    if (dev_id < 0 || dev_id >= OPAL_ACCELERATOR_MAX_DEVICES) {
        return NULL;
    }
    if (NULL == opal_accelerator.mem_alloc) {
        return NULL;
    }
    if (opal_accel_device_allocators[dev_id] != NULL) {
        return opal_accel_device_allocators[dev_id];
    }

    OPAL_THREAD_LOCK(&opal_accel_alloc_lock);
    if (opal_accel_device_allocators[dev_id] == NULL) {
        ctx = (opal_accel_alloc_ctx_t *)malloc(sizeof(*ctx));
        if (NULL == ctx) {
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        ctx->dev_id = dev_id;
        bucket = mca_allocator_bucket_init(NULL, OPAL_ACCELERATOR_ALLOC_NUM_BUCKETS,
                                           opal_accel_seg_alloc, opal_accel_seg_free);
        if (NULL == bucket) {
            free(ctx);
            OPAL_THREAD_UNLOCK(&opal_accel_alloc_lock);
            return NULL;
        }
        bucket->super.alc_context = ctx;
        opal_accel_device_allocators[dev_id] = &bucket->super;
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
    for (int i = 0; i < OPAL_ACCELERATOR_MAX_DEVICES; i++) {
        if (opal_accel_device_allocators[i] != NULL) {
            opal_accel_alloc_ctx_t *ctx = (opal_accel_alloc_ctx_t *)opal_accel_device_allocators[i]->alc_context;
            opal_accel_device_allocators[i]->alc_finalize(opal_accel_device_allocators[i]);
            free(ctx);
            opal_accel_device_allocators[i] = NULL;
        }
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
