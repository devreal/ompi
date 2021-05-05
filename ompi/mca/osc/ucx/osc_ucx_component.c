/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017. ALL RIGHTS RESERVED.
 * Copyright (c) 2018      Amazon.com, Inc. or its affiliates.  All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "opal/util/printf.h"

#include "ompi/mca/osc/osc.h"
#include "ompi/mca/osc/base/base.h"
#include "ompi/mca/osc/base/osc_base_obj_convert.h"
#include "opal/mca/common/ucx/common_ucx.h"
#include "opal/class/opal_fifo.h"

#include "osc_ucx.h"
#include "osc_ucx_request.h"

#include <stddef.h>

#define memcpy_off(_dst, _src, _len, _off)        \
    memcpy(((char*)(_dst)) + (_off), _src, _len); \
    (_off) += (_len);

opal_mutex_t mca_osc_service_mutex = OPAL_MUTEX_STATIC_INIT;
static void _osc_ucx_init_lock(void)
{
    if(mca_osc_ucx_component.enable_mpi_threads) {
        opal_mutex_lock(&mca_osc_service_mutex);
    }
}
static void _osc_ucx_init_unlock(void)
{
    if(mca_osc_ucx_component.enable_mpi_threads) {
        opal_mutex_unlock(&mca_osc_service_mutex);
    }
}

static opal_lifo_t module_free_list;


static int component_open(void);
static int component_register(void);
static int component_init(bool enable_progress_threads, bool enable_mpi_threads);
static int component_finalize(void);
static int component_query(struct ompi_win_t *win, void **base, size_t size, int disp_unit,
                           struct ompi_communicator_t *comm, struct opal_info_t *info, int flavor);
static int component_select(struct ompi_win_t *win, void **base, size_t size, int disp_unit,
                            struct ompi_communicator_t *comm, struct opal_info_t *info,
                            int flavor, int *model);
static void ompi_osc_ucx_unregister_progress(void);

ompi_osc_ucx_component_t mca_osc_ucx_component = {
    { /* ompi_osc_base_component_t */
        .osc_version = {
            OMPI_OSC_BASE_VERSION_3_0_0,
            .mca_component_name = "ucx",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),
            .mca_open_component = component_open,
            .mca_register_component_params = component_register,
        },
        .osc_data = {
            /* The component is not checkpoint ready */
            MCA_BASE_METADATA_PARAM_NONE
        },
        .osc_init = component_init,
        .osc_query = component_query,
        .osc_select = component_select,
        .osc_finalize = component_finalize,
    },
    .wpool                  = NULL,
    .env_initialized        = false,
    .num_incomplete_req_ops = 0,
    .num_modules            = 0,
    .acc_single_intrinsic   = false
};

ompi_osc_ucx_module_t ompi_osc_ucx_module_template = {
    {
        .osc_win_attach = ompi_osc_ucx_win_attach,
        .osc_win_detach = ompi_osc_ucx_win_detach,
        .osc_free = ompi_osc_ucx_free,

        .osc_put = ompi_osc_ucx_put,
        .osc_get = ompi_osc_ucx_get,
        .osc_accumulate = ompi_osc_ucx_accumulate,
        .osc_compare_and_swap = ompi_osc_ucx_compare_and_swap,
        .osc_fetch_and_op = ompi_osc_ucx_fetch_and_op,
        .osc_get_accumulate = ompi_osc_ucx_get_accumulate,

        .osc_rput = ompi_osc_ucx_rput,
        .osc_rget = ompi_osc_ucx_rget,
        .osc_raccumulate = ompi_osc_ucx_raccumulate,
        .osc_rget_accumulate = ompi_osc_ucx_rget_accumulate,

        .osc_fence = ompi_osc_ucx_fence,

        .osc_start = ompi_osc_ucx_start,
        .osc_complete = ompi_osc_ucx_complete,
        .osc_post = ompi_osc_ucx_post,
        .osc_wait = ompi_osc_ucx_wait,
        .osc_test = ompi_osc_ucx_test,

        .osc_lock = ompi_osc_ucx_lock,
        .osc_unlock = ompi_osc_ucx_unlock,
        .osc_lock_all = ompi_osc_ucx_lock_all,
        .osc_unlock_all = ompi_osc_ucx_unlock_all,

        .osc_sync = ompi_osc_ucx_sync,
        .osc_flush = ompi_osc_ucx_flush,
        .osc_flush_all = ompi_osc_ucx_flush_all,
        .osc_flush_local = ompi_osc_ucx_flush_local,
        .osc_flush_local_all = ompi_osc_ucx_flush_local_all,

        .osc_get_memhandle = ompi_osc_ucx_get_memhandle,
        .osc_release_memhandle = ompi_osc_ucx_release_memhandle,
        .osc_from_memhandle = ompi_osc_ucx_from_memhandle,
    }
};

/* look up parameters for configuring this window.  The code first
   looks in the info structure passed by the user, then it checks
   for a matching MCA variable. */
static bool check_config_value_bool (char *key, opal_info_t *info)
{
    int ret, flag, param;
    bool result = false;
    const bool *flag_value = &result;

    ret = opal_info_get_bool (info, key, &result, &flag);
    if (OMPI_SUCCESS == ret && flag) {
        return result;
    }

    param = mca_base_var_find("ompi", "osc", "ucx", key);
    if (0 <= param) {
        (void) mca_base_var_get_value(param, &flag_value, NULL, NULL);
    }

    return flag_value[0];
}

static int component_open(void) {
    return OMPI_SUCCESS;
}

static int component_register(void) {
    unsigned major          = 0;
    unsigned minor          = 0;
    unsigned release_number = 0;
    char *description_str;

    ucp_get_version(&major, &minor, &release_number);

    mca_osc_ucx_component.priority = UCX_VERSION(major, minor, release_number) >= UCX_VERSION(1, 5, 0) ? 60 : 0;

    opal_asprintf(&description_str, "Priority of the osc/ucx component (default: %d)",
             mca_osc_ucx_component.priority);
    (void) mca_base_component_var_register(&mca_osc_ucx_component.super.osc_version, "priority", description_str,
                                           MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, 0, 0, OPAL_INFO_LVL_3,
                                           MCA_BASE_VAR_SCOPE_GROUP, &mca_osc_ucx_component.priority);
    free(description_str);

    mca_osc_ucx_component.no_locks = false;

    opal_asprintf(&description_str, "Enable optimizations available only if MPI_LOCK is "
             "not used. Info key of same name overrides this value (default: %s)",
             mca_osc_ucx_component.no_locks  ? "true" : "false");
    (void) mca_base_component_var_register(&mca_osc_ucx_component.super.osc_version, "no_locks", description_str,
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_GROUP, &mca_osc_ucx_component.no_locks);
    free(description_str);

    mca_osc_ucx_component.acc_single_intrinsic = false;
    opal_asprintf(&description_str, "Enable optimizations for MPI_Fetch_and_op, MPI_Accumulate, etc for codes "
             "that will not use anything more than a single predefined datatype (default: %s)",
             mca_osc_ucx_component.acc_single_intrinsic  ? "true" : "false");
    (void) mca_base_component_var_register(&mca_osc_ucx_component.super.osc_version, "acc_single_intrinsic",
                                           description_str, MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_GROUP, &mca_osc_ucx_component.acc_single_intrinsic);
    free(description_str);

    opal_common_ucx_mca_var_register(&mca_osc_ucx_component.super.osc_version);

    return OMPI_SUCCESS;
}

static int progress_callback(void) {
    int rc = 0;
    if (mca_osc_ucx_component.wpool != NULL) {
        rc = opal_common_ucx_wpool_progress(mca_osc_ucx_component.wpool);
    }
    return rc;
}

static int component_init(bool enable_progress_threads, bool enable_mpi_threads) {
    mca_osc_ucx_component.enable_mpi_threads = enable_mpi_threads;
    mca_osc_ucx_component.wpool = opal_common_ucx_wpool_allocate();
    opal_common_ucx_mca_register();
    return OMPI_SUCCESS;
}

static int component_finalize(void) {
    opal_common_ucx_mca_deregister();
    if (mca_osc_ucx_component.env_initialized) {
        opal_common_ucx_wpool_finalize(mca_osc_ucx_component.wpool);
    }
    opal_common_ucx_wpool_free(mca_osc_ucx_component.wpool);
    return OMPI_SUCCESS;
}

static int component_query(struct ompi_win_t *win, void **base, size_t size, int disp_unit,
                           struct ompi_communicator_t *comm, struct opal_info_t *info, int flavor) {
    if (MPI_WIN_FLAVOR_SHARED == flavor) return -1;
    return mca_osc_ucx_component.priority;
}

static int exchange_len_info(void *my_info, size_t my_info_len, char **recv_info_ptr,
                             int **disps_ptr, void *metadata)
{
    int ret = OMPI_SUCCESS;
    struct ompi_communicator_t *comm = (struct ompi_communicator_t *)metadata;
    int comm_size = ompi_comm_size(comm);
    int lens[comm_size];
    int total_len, i;

    ret = comm->c_coll->coll_allgather(&my_info_len, 1, MPI_INT,
                                       lens, 1, MPI_INT, comm,
                                       comm->c_coll->coll_allgather_module);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    total_len = 0;
    (*disps_ptr) = (int *)calloc(comm_size, sizeof(int));
    for (i = 0; i < comm_size; i++) {
        (*disps_ptr)[i] = total_len;
        total_len += lens[i];
    }

    (*recv_info_ptr) = (char *)calloc(total_len, sizeof(char));
    ret = comm->c_coll->coll_allgatherv(my_info, my_info_len, MPI_BYTE,
                                        (void *)(*recv_info_ptr), lens, (*disps_ptr), MPI_BYTE,
                                        comm, comm->c_coll->coll_allgatherv_module);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    return ret;
}

static void ompi_osc_ucx_unregister_progress()
{
    int ret;

    /* May be called concurrently - protect */
    _osc_ucx_init_lock();

    mca_osc_ucx_component.num_modules--;
    OSC_UCX_ASSERT(mca_osc_ucx_component.num_modules >= 0);
    if (0 == mca_osc_ucx_component.num_modules) {
        ret = opal_progress_unregister(progress_callback);
        if (OMPI_SUCCESS != ret) {
            OSC_UCX_VERBOSE(1, "opal_progress_unregister failed: %d", ret);
        }
    }

    _osc_ucx_init_unlock();
}

static char* ompi_osc_ucx_set_no_lock_info(opal_infosubscriber_t *obj, char *key, char *value)
{

    struct ompi_win_t *win = (struct ompi_win_t*) obj;
    ompi_osc_ucx_module_t *module = (ompi_osc_ucx_module_t *)win->w_osc_module;
    bool temp;

    temp = opal_str_to_bool(value);

    if (temp && !module->no_locks) {
        /* clean up the lock hash. it is up to the user to ensure no lock is
         * outstanding from this process when setting the info key */
        OBJ_DESTRUCT(&module->outstanding_locks);
        module->no_locks = true;
        win->w_flags |= OMPI_WIN_NO_LOCKS;
    } else if (!temp && module->no_locks) {
        int comm_size = ompi_comm_size (module->comm);
        int ret;

        OBJ_CONSTRUCT(&module->outstanding_locks, opal_hash_table_t);
        ret = opal_hash_table_init (&module->outstanding_locks, comm_size);
        if (OPAL_SUCCESS != ret) {
            module->no_locks = true;
        } else {
            module->no_locks = false;
        }
        win->w_flags &= ~OMPI_WIN_NO_LOCKS;
    }
    module->comm->c_coll->coll_barrier(module->comm, module->comm->c_coll->coll_barrier_module);
    return module->no_locks ? "true" : "false";
}

static int comm_delete_attr_delete_function(
    MPI_Comm comm,
    int comm_keyval,
    void *attribute_val,
    void *extra_state)
{
    (void)comm; (void)comm_keyval; (void)extra_state;
    opal_common_ucx_wpool_t* wpool = (opal_common_ucx_wpool_t*)attribute_val;
    opal_common_ucx_wpool_finalize(wpool);
    return MPI_SUCCESS;
}

static int comm_wpool_key;

static int initialize_env()
{
    int ret;

    /* Lazy initialization of the global state.
      * As not all of the MPI applications are using One-Sided functionality
      * we don't want to initialize in the component_init()
      */

    OBJ_CONSTRUCT(&mca_osc_ucx_component.requests, opal_free_list_t);
    ret = opal_free_list_init (&mca_osc_ucx_component.requests,
                                sizeof(ompi_osc_ucx_request_t),
                                opal_cache_line_size,
                                OBJ_CLASS(ompi_osc_ucx_request_t),
                                0, 0, 8, 0, 8, NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        OSC_UCX_VERBOSE(1, "opal_free_list_init failed: %d", ret);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    ret = opal_common_ucx_wpool_init(mca_osc_ucx_component.wpool,
                                      ompi_proc_world_size(),
                                      mca_osc_ucx_component.enable_mpi_threads);
    if (OMPI_SUCCESS != ret) {
        OSC_UCX_VERBOSE(1, "opal_common_ucx_wpool_init failed: %d", ret);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    OBJ_CONSTRUCT(&module_free_list, opal_lifo_t);

    //PMPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, &comm_delete_attr_delete_function,
    //                        &comm_wpool_key, NULL);

    /* Make sure that all memory updates performed above are globally
      * observable before (mca_osc_ucx_component.env_initialized = true)
      */
    mca_osc_ucx_component.env_initialized = true;

    return OMPI_SUCCESS;
}

static int component_select(struct ompi_win_t *win, void **base, size_t size, int disp_unit,
                            struct ompi_communicator_t *comm, struct opal_info_t *info,
                            int flavor, int *model) {
    ompi_osc_ucx_module_t *module = NULL;
    char *name = NULL;
    long values[2];
    int ret = OMPI_SUCCESS;
    //ucs_status_t status;
    int i, comm_size = ompi_comm_size(comm);
    bool env_initialized = false;
    void *state_base = NULL;
    opal_common_ucx_mem_type_t mem_type;
    uint64_t zero = 0;
    char *my_mem_addr;
    int my_mem_addr_size;
    void * my_info = NULL;
    char *recv_buf = NULL;

    /* the osc/sm component is the exclusive provider for support for
     * shared memory windows */
    if (flavor == MPI_WIN_FLAVOR_SHARED) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /* May be called concurrently - protect */
    _osc_ucx_init_lock();

    if (mca_osc_ucx_component.env_initialized == false) {
        ret = initialize_env();
        env_initialized = true;

        if (OMPI_SUCCESS != ret) {
            goto select_unlock;
        }
    }

    /* Account for the number of active "modules" = MPI windows */
    mca_osc_ucx_component.num_modules++;

    /* If this is the first window to be registered - register the progress
     * callback
     */
    OSC_UCX_ASSERT(mca_osc_ucx_component.num_modules > 0);
    if (1 == mca_osc_ucx_component.num_modules) {
        ret = opal_progress_register(progress_callback);
        if (OMPI_SUCCESS != ret) {
            OSC_UCX_VERBOSE(1, "opal_progress_register failed: %d", ret);
            goto select_unlock;
        }
    }

select_unlock:
    _osc_ucx_init_unlock();
    if (ret) {
        goto error;
    }

    /* create module structure */
    module = (ompi_osc_ucx_module_t *)calloc(1, sizeof(ompi_osc_ucx_module_t));
    if (module == NULL) {
        ret = OMPI_ERR_TEMP_OUT_OF_RESOURCE;
        goto error_nomem;
    }

    /* fill in the function pointer part */
    memcpy(module, &ompi_osc_ucx_module_template, sizeof(ompi_osc_base_module_t));

    ret = ompi_comm_dup(comm, &module->comm);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }

    *model = MPI_WIN_UNIFIED;
    opal_asprintf(&name, "ucx window %d", ompi_comm_get_cid(module->comm));
    ompi_win_set_name(win, name);
    free(name);

    module->flavor = flavor;
    module->size = size;
    module->no_locks = check_config_value_bool ("no_locks", info);
    module->acc_single_intrinsic = check_config_value_bool ("acc_single_intrinsic", info);

    /* share everyone's displacement units. Only do an allgather if
       strictly necessary, since it requires O(p) state. */
    values[0] = disp_unit;
    values[1] = -disp_unit;

    ret = module->comm->c_coll->coll_allreduce(MPI_IN_PLACE, values, 2, MPI_LONG,
                                               MPI_MIN, module->comm,
                                               module->comm->c_coll->coll_allreduce_module);
    if (OMPI_SUCCESS != ret) {
        goto error;
    }

    if (values[0] == -values[1]) { /* everyone has the same disp_unit, we do not need O(p) space */
        module->disp_unit = disp_unit;
    } else { /* different disp_unit sizes, allocate O(p) space to store them */
        module->disp_unit = -1;
        module->disp_units = calloc(comm_size, sizeof(int));
        if (module->disp_units == NULL) {
            ret = OMPI_ERR_TEMP_OUT_OF_RESOURCE;
            goto error;
        }

        ret = module->comm->c_coll->coll_allgather(&disp_unit, 1, MPI_INT,
                                                   module->disp_units, 1, MPI_INT,
                                                   module->comm,
                                                   module->comm->c_coll->coll_allgather_module);
        if (OMPI_SUCCESS != ret) {
            goto error;
        }
    }

    ret = opal_common_ucx_wpctx_create(mca_osc_ucx_component.wpool, comm_size,
                                     &exchange_len_info, (void *)module->comm,
                                     &module->ctx);
    if (OMPI_SUCCESS != ret) {
        goto error;
    }

    if (flavor == MPI_WIN_FLAVOR_ALLOCATE || flavor == MPI_WIN_FLAVOR_CREATE) {
        switch (flavor) {
        case MPI_WIN_FLAVOR_ALLOCATE:
            mem_type = OPAL_COMMON_UCX_MEM_ALLOCATE_MAP;
            break;
        case MPI_WIN_FLAVOR_CREATE:
            mem_type = OPAL_COMMON_UCX_MEM_MAP;
            break;
        }

        ret = opal_common_ucx_wpmem_create(module->ctx, base, size,
                                         mem_type, &exchange_len_info,
                                         (void *)module->comm,
                                           &my_mem_addr, &my_mem_addr_size,
                                           &module->mem);
        if (ret != OMPI_SUCCESS) {
            goto error;
        }

    }

    state_base = (void *)&(module->state);
    ret = opal_common_ucx_wpmem_create(module->ctx, &state_base,
                                     sizeof(ompi_osc_ucx_state_t),
                                     OPAL_COMMON_UCX_MEM_MAP, &exchange_len_info,
                                     (void *)module->comm,
                                       &my_mem_addr, &my_mem_addr_size,
                                       &module->state_mem);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }

    /* exchange window addrs */
    my_info = malloc(2 * sizeof(uint64_t));
    if (my_info == NULL) {
        ret = OMPI_ERR_TEMP_OUT_OF_RESOURCE;
        goto error;
    }

    if (flavor == MPI_WIN_FLAVOR_ALLOCATE || flavor == MPI_WIN_FLAVOR_CREATE) {
        memcpy(my_info, base, sizeof(uint64_t));
    } else {
        memcpy(my_info, &zero, sizeof(uint64_t));
    }
    memcpy((char*)my_info + sizeof(uint64_t), &state_base, sizeof(uint64_t));

    recv_buf = (char *)calloc(comm_size, 2 * sizeof(uint64_t));
    ret = comm->c_coll->coll_allgather((void *)my_info, 2 * sizeof(uint64_t),
                                       MPI_BYTE, recv_buf, 2 * sizeof(uint64_t),
                                       MPI_BYTE, comm, comm->c_coll->coll_allgather_module);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }

    module->addrs = calloc(comm_size, sizeof(uint64_t));
    module->state_addrs = calloc(comm_size, sizeof(uint64_t));
    for (i = 0; i < comm_size; i++) {
        memcpy(&(module->addrs[i]), recv_buf + i * 2 * sizeof(uint64_t), sizeof(uint64_t));
        memcpy(&(module->state_addrs[i]), recv_buf + i * 2 * sizeof(uint64_t) + sizeof(uint64_t), sizeof(uint64_t));
    }
    free(recv_buf);

    /* init window state */
    module->state.lock = TARGET_LOCK_UNLOCKED;
    module->state.post_index = 0;
    memset((void *)module->state.post_state, 0, sizeof(uint64_t) * OMPI_OSC_UCX_POST_PEER_MAX);
    module->state.complete_count = 0;
    module->state.req_flag = 0;
    module->state.acc_lock = TARGET_LOCK_UNLOCKED;
    module->state.dynamic_win_count = 0;
    for (i = 0; i < OMPI_OSC_UCX_ATTACH_MAX; i++) {
        module->local_dynamic_win_info[i].refcnt = 0;
    }
    module->epoch_type.access = NONE_EPOCH;
    module->epoch_type.exposure = NONE_EPOCH;
    module->lock_count = 0;
    module->post_count = 0;
    module->start_group = NULL;
    module->post_group = NULL;
    OBJ_CONSTRUCT(&module->pending_posts, opal_list_t);
    module->start_grp_ranks = NULL;
    module->lock_all_is_nocheck = false;

    if (!module->no_locks) {
        OBJ_CONSTRUCT(&module->outstanding_locks, opal_hash_table_t);
        ret = opal_hash_table_init(&module->outstanding_locks, comm_size);
        if (ret != OPAL_SUCCESS) {
            goto error;
        }
    } else {
        win->w_flags |= OMPI_WIN_NO_LOCKS;
    }

    win->w_osc_module = &module->super;

    opal_infosubscribe_subscribe(&win->super, "no_locks", "false", ompi_osc_ucx_set_no_lock_info);

    /* sync with everyone */

    ret = module->comm->c_coll->coll_barrier(module->comm,
                                             module->comm->c_coll->coll_barrier_module);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }

    return ret;

error:
    if (module->disp_units) free(module->disp_units);
    if (module->comm) ompi_comm_free(&module->comm);
    free(module);

error_nomem:
    if (env_initialized == true) {
        opal_common_ucx_wpool_finalize(mca_osc_ucx_component.wpool);
        OBJ_DESTRUCT(&mca_osc_ucx_component.requests);
        mca_osc_ucx_component.env_initialized = false;
    }

    ompi_osc_ucx_unregister_progress();
    return ret;
}


int ompi_osc_ucx_from_memhandle(struct ompi_win_t *win, size_t size, int disp_unit, int target,
                                struct ompi_win_t *parentwin, struct opal_info_t *info,
                                const char memhandle[], int *model) {
    ompi_osc_ucx_module_t *module = NULL;
    char *name = NULL;
    int ret = OMPI_SUCCESS;
    //ucs_status_t status;
    int i;

#if 0
    bool env_initialized = false;

    /* May be called concurrently - protect */
    _osc_ucx_init_lock();

    if (mca_osc_ucx_component.env_initialized == false) {
        ret = initialize_env();
        env_initialized = true;

        if (OMPI_SUCCESS != ret) {
            goto select_unlock;
        }
    }

    /* Account for the number of active "modules" = MPI windows */
    mca_osc_ucx_component.num_modules++;

    /* If this is the first window to be registered - register the progress
     * callback
     */
    OSC_UCX_ASSERT(mca_osc_ucx_component.num_modules > 0);
    if (1 == mca_osc_ucx_component.num_modules) {
        ret = opal_progress_register(progress_callback);
        if (OMPI_SUCCESS != ret) {
            OSC_UCX_VERBOSE(1, "opal_progress_register failed: %d", ret);
            goto select_unlock;
        }
    }

    int flag;
    opal_common_ucx_wpool_t* wpool = NULL;

    PMPI_Comm_get_attr(comm, comm_wpool_key, &wpool, &flag);

    if (!flag) {
        wpool = calloc(1, sizeof(*wpool));
        ret = opal_common_ucx_wpool_init(wpool,
                                         ompi_proc_world_size(),
                                         mca_osc_ucx_component.enable_mpi_threads);
        if (OMPI_SUCCESS != ret) {
            OSC_UCX_VERBOSE(1, "opal_common_ucx_wpool_init failed: %d", ret);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        PMPI_Comm_set_attr(comm, comm_wpool_key, wpool);
    }
select_unlock:
    _osc_ucx_init_unlock();
    if (ret) {
        goto error;
    }
#endif // 0
    ompi_osc_ucx_module_t *parentmodule = (ompi_osc_ucx_module_t*)parentwin->w_osc_module;
    OBJ_RETAIN(parentmodule->ctx);

    opal_list_item_t *free_item = opal_lifo_pop(&module_free_list);
    if (NULL != free_item) {
        module = (ompi_osc_ucx_module_t*)(((intptr_t)free_item) - offsetof(ompi_osc_ucx_module_t, free_list_item));
    } else {
        /* Allocate a new module */

        /* create module structure */
        module = (ompi_osc_ucx_module_t *)calloc(1, sizeof(ompi_osc_ucx_module_t));
        if (module == NULL) {
            ret = OMPI_ERR_TEMP_OUT_OF_RESOURCE;
            goto error_nomem;
        }

        /* fill in the function pointer part */
        memcpy(module, &ompi_osc_ucx_module_template, sizeof(ompi_osc_base_module_t));

        module->super.osc_get_memhandle = NULL;
        module->super.osc_release_memhandle = NULL;
        module->super.osc_from_memhandle = NULL;

        OBJ_CONSTRUCT(&module->free_list_item, opal_list_item_t);

        module->mem = calloc(1, sizeof(*module->mem));
        OBJ_CONSTRUCT(&module->mem->mutex, opal_mutex_t);
        OBJ_CONSTRUCT(&module->mem->thread_rkey_list, opal_list_t);
        //OBJ_CONSTRUCT(&module->mem->mem_records, opal_list_t);
        module->mem->mem_displs = NULL;
        //OBJ_CONSTRUCT(&module->mem->tls_key, opal_tsd_tracked_key_t);
        //opal_tsd_tracked_key_set_destructor(&module->mem->tls_key, _mem_rec_destructor);

        /* init window state */
        module->state.lock = TARGET_LOCK_UNLOCKED;
        module->state.post_index = 0;
        memset((void *)module->state.post_state, 0, sizeof(uint64_t) * OMPI_OSC_UCX_POST_PEER_MAX);
        module->state.complete_count = 0;
        module->state.req_flag = 0;
        module->state.acc_lock = TARGET_LOCK_UNLOCKED;
        module->state.dynamic_win_count = 0;
        for (i = 0; i < OMPI_OSC_UCX_ATTACH_MAX; i++) {
            module->local_dynamic_win_info[i].refcnt = 0;
        }
        module->epoch_type.access = NONE_EPOCH;
        module->epoch_type.exposure = NONE_EPOCH;
        module->lock_count = 0;
        module->post_count = 0;
        module->start_group = NULL;
        module->post_group = NULL;
        OBJ_CONSTRUCT(&module->pending_posts, opal_list_t);
        module->start_grp_ranks = NULL;
        module->lock_all_is_nocheck = false;

        int comm_size = ompi_comm_size(parentmodule->comm);
        OBJ_CONSTRUCT(&module->outstanding_locks, opal_hash_table_t);
        ret = opal_hash_table_init(&module->outstanding_locks, comm_size);
        if (ret != OPAL_SUCCESS) {
            goto error;
        }

    }

#if 0
    ret = opal_common_ucx_wpctx_create(wpool, comm_size,
                                      NULL, NULL,
                                      &module->ctx);
    if (OMPI_SUCCESS != ret) {
        goto error;
    }
#endif // 0

    module->ctx = parentmodule->ctx;

    module->mem->ctx = module->ctx;

    module->comm = parentmodule->comm;

    *model = MPI_WIN_UNIFIED;
    bool no_attr, lock_shared;
    int flag;
    opal_info_get_bool(info, "mpi_win_no_attr", &no_attr, &flag);
    if (flag && no_attr) {
        opal_asprintf(&name, "ucx window %d (parent %s)", ompi_comm_get_cid(module->comm), parentwin->w_name);
        ompi_win_set_name(win, name);
        free(name);
    }

    module->flavor = MPI_WIN_FLAVOR_MEMHANDLE;
    module->size = size;
    module->no_locks = false;
    module->acc_single_intrinsic = check_config_value_bool("acc_single_intrinsic", info);
    opal_info_get_bool(info, "mpi_lock_shared", &lock_shared, &flag);
    module->always_lock_shared   = flag && lock_shared;
    module->disp_unit = disp_unit;

    bool need_state = !(module->always_lock_shared && module->acc_single_intrinsic);


    const ompi_osc_ucx_memhandle_t *ucx_memhandle = (const ompi_osc_ucx_memhandle_t*)memhandle;

    /* fill in the connection details */
    const void *data_rkey_addr = (ucx_memhandle->_data);

    static int prev_data_rkey_size = 0;

    /** TODO: this is assuming that the rkeys for data and state have always the same size! */
    if (NULL == module->mem->mem_addrs) {
        module->mem->mem_addrs = malloc(ucx_memhandle->data_rkey_size);
        prev_data_rkey_size = ucx_memhandle->data_rkey_size;
    }
    memcpy(module->mem->mem_addrs, data_rkey_addr, ucx_memhandle->data_rkey_size);
    if (prev_data_rkey_size != ucx_memhandle->data_rkey_size) {
        printf("WARN: previous rkey size does not match current rkey size: %d vs %d\n",
               prev_data_rkey_size, ucx_memhandle->data_rkey_size);
    }

    if (need_state) {
        /* allocate state lazily */
        if (NULL == module->state_mem) {
            module->state_mem = calloc(1, sizeof(*module->state_mem));
            OBJ_CONSTRUCT(&module->state_mem->mutex, opal_mutex_t);
            OBJ_CONSTRUCT(&module->state_mem->thread_rkey_list, opal_list_t);
            //OBJ_CONSTRUCT(&module->state_mem->mem_records, opal_list_t);
            //module->state_mem->mem_displs = NULL;
            //OBJ_CONSTRUCT(&module->state_mem->tls_key, opal_tsd_tracked_key_t);
            //opal_tsd_tracked_key_set_destructor(&module->state_mem->tls_key, _mem_rec_destructor);
        }
        module->state_mem->ctx = module->ctx;
        static int prev_state_rkey_size = 0;
        if (NULL == module->state_mem->mem_addrs && ucx_memhandle->state_rkey_size > 0) {
            module->state_mem->mem_addrs = malloc(ucx_memhandle->state_rkey_size);
            prev_state_rkey_size = ucx_memhandle->state_rkey_size;
        }

        if (prev_state_rkey_size < ucx_memhandle->state_rkey_size) {
            printf("WARN: previous rkey size does not match current rkey size: %d vs %d\n",
                  prev_state_rkey_size, ucx_memhandle->state_rkey_size);
        }

        const void *state_rkey_addr = (ucx_memhandle->_data + sizeof(ucp_mem_h) + ucx_memhandle->data_rkey_size);
        memcpy(module->state_mem->mem_addrs, state_rkey_addr, ucx_memhandle->state_rkey_size);

        //ucs_status_t status;
        //status = ucp_ep_rkey_unpack(ep, state_rkey_addr, &module->state_mem->rkey);
    }
    module->has_state = need_state;

    if (module->always_lock_shared) {
        module->epoch_type.access = PASSIVE_EPOCH;
        module->lock_count++;
    } else {
        module->epoch_type.access = NONE_EPOCH;
        module->lock_count = 0;
    }

#if 0
    if (flavor == MPI_WIN_FLAVOR_ALLOCATE || flavor == MPI_WIN_FLAVOR_CREATE) {
        switch (flavor) {
        case MPI_WIN_FLAVOR_ALLOCATE:
            mem_type = OPAL_COMMON_UCX_MEM_ALLOCATE_MAP;
            break;
        case MPI_WIN_FLAVOR_CREATE:
            mem_type = OPAL_COMMON_UCX_MEM_MAP;
            break;
        }

        ret = opal_common_ucx_wpmem_create(module->ctx, base, size,
                                         mem_type, &exchange_len_info,
                                         (void *)module->comm,
                                           &my_mem_addr, &my_mem_addr_size,
                                           &module->mem);
        if (ret != OMPI_SUCCESS) {
            goto error;
        }

    }

    state_base = (void *)&(module->state);
    ret = opal_common_ucx_wpmem_create(module->ctx, &state_base,
                                     sizeof(ompi_osc_ucx_state_t),
                                     OPAL_COMMON_UCX_MEM_MAP, &exchange_len_info,
                                     (void *)module->comm,
                                       &my_mem_addr, &my_mem_addr_size,
                                       &module->state_mem);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }
#endif

#if 0

    /* fill in the data and state addresses */
    free(module->addrs);
    module->addrs = calloc(target+1, sizeof(uint64_t));
    module->addrs[target] = ucx_memhandle->data_addr;

    free(module->state_addrs);
    module->state_addrs = NULL;
    if (ucx_memhandle->state_addr) {
        module->state_addrs = calloc(target+1, sizeof(uint64_t));
        module->state_addrs[target] = ucx_memhandle->state_addr;
    }
#endif // 0
    module->addrs = (uint64_t*)ucx_memhandle->data_addr;
    module->state_addrs = (uint64_t*)ucx_memhandle->state_addr;

#if 0
    /* exchange window addrs */
    my_info = malloc(2 * sizeof(uint64_t));
    if (my_info == NULL) {
        ret = OMPI_ERR_TEMP_OUT_OF_RESOURCE;
        goto error;
    }

    if (flavor == MPI_WIN_FLAVOR_ALLOCATE || flavor == MPI_WIN_FLAVOR_CREATE) {
        memcpy(my_info, base, sizeof(uint64_t));
    } else {
        memcpy(my_info, &zero, sizeof(uint64_t));
    }
    memcpy((char*)my_info + sizeof(uint64_t), &state_base, sizeof(uint64_t));

    recv_buf = (char *)calloc(comm_size, 2 * sizeof(uint64_t));
    ret = comm->c_coll->coll_allgather((void *)my_info, 2 * sizeof(uint64_t),
                                       MPI_BYTE, recv_buf, 2 * sizeof(uint64_t),
                                       MPI_BYTE, comm, comm->c_coll->coll_allgather_module);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }

    module->addrs = calloc(comm_size, sizeof(uint64_t));
    module->state_addrs = calloc(comm_size, sizeof(uint64_t));
    for (i = 0; i < comm_size; i++) {
        memcpy(&(module->addrs[i]), recv_buf + i * 2 * sizeof(uint64_t), sizeof(uint64_t));
        memcpy(&(module->state_addrs[i]), recv_buf + i * 2 * sizeof(uint64_t) + sizeof(uint64_t), sizeof(uint64_t));
    }
    free(recv_buf);
#endif


    if (module->no_locks) {
        win->w_flags |= OMPI_WIN_NO_LOCKS;
    }

    win->w_osc_module = &module->super;

    //opal_infosubscribe_subscribe(&win->super, "no_locks", "false", ompi_osc_ucx_set_no_lock_info);

#if 0
    /* sync with everyone */
    ret = module->comm->c_coll->coll_barrier(module->comm,
                                             module->comm->c_coll->coll_barrier_module);
    if (ret != OMPI_SUCCESS) {
        goto error;
    }
#endif // 0

    return ret;

error:
    if (module->disp_units) free(module->disp_units);
    if (module->addrs) free(module->addrs);
    if (module->state_addrs) free(module->state_addrs);
    free(module);

error_nomem:
#if 0
    if (env_initialized == true) {
        opal_common_ucx_wpool_finalize(wpool);
        OBJ_DESTRUCT(&mca_osc_ucx_component.requests);
        mca_osc_ucx_component.env_initialized = false;
    }
#endif // 0

    ompi_osc_ucx_unregister_progress();
    return ret;
}

int ompi_osc_ucx_get_memhandle(void *base,
                            size_t size,
                            struct opal_info_t *info,
                            struct ompi_win_t *parentwin,
                            char memhandle[],
                            int *memhandle_size)
{
    void *data_rkey_addr;
    size_t data_rkey_addr_len;
    void *state_rkey_addr = NULL;
    size_t state_rkey_addr_len = 0;
    ucs_status_t status;
    int ret = OMPI_SUCCESS;
    ucp_mem_h data_memh, state_memh;

#if 0
    bool env_initialized = false;

    /* May be called concurrently - protect */
    _osc_ucx_init_lock();

    if (mca_osc_ucx_component.env_initialized == false) {
        ret = initialize_env();
        env_initialized = true;

        if (OMPI_SUCCESS != ret) {
            goto select_unlock;
        }
    }

    /* If this is the first window to be registered - register the progress
     * callback
     */
    if (env_initialized) {
        ret = opal_progress_register(progress_callback);
        if (OMPI_SUCCESS != ret) {
            OSC_UCX_VERBOSE(1, "opal_progress_register failed: %d", ret);
            goto select_unlock;
        }
    }

select_unlock:
    _osc_ucx_init_unlock();
    if (ret) {
        return ret;
    }
#endif // 0

    bool acc_single_intrinsic = check_config_value_bool ("acc_single_intrinsic", info);
    bool always_lock_shared   = check_config_value_bool("mpi_lock_shared", info);

    bool need_state = !acc_single_intrinsic || !always_lock_shared;

    /* register data memory */
    ret = _comm_ucx_wpmem_map(mca_osc_ucx_component.wpool, &base, size, &data_memh,
                            OPAL_COMMON_UCX_MEM_MAP);
    if (ret != OPAL_SUCCESS) {
        MCA_COMMON_UCX_VERBOSE(1, "_comm_ucx_mem_map failed: %d", ret);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    status = ucp_rkey_pack(mca_osc_ucx_component.wpool->ucp_ctx, data_memh,
                           &data_rkey_addr, &data_rkey_addr_len);

    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_rkey_pack failed: %d", status);
        ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, data_memh);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* allocate and register state memory */
    //ompi_osc_ucx_state_t *state_base = malloc(sizeof(ompi_osc_ucx_state_t));
    ompi_osc_ucx_state_t *state_base = NULL;

    /* register state memory */
    if (need_state) {
        ret = _comm_ucx_wpmem_map(mca_osc_ucx_component.wpool, (void**)&state_base,
                                  sizeof(ompi_osc_ucx_state_t), &state_memh,
                                  OPAL_COMMON_UCX_MEM_ALLOCATE_MAP);

        /* init window state */
        state_base->lock = TARGET_LOCK_UNLOCKED;
        state_base->post_index = 0;
        memset((void *)state_base->post_state, 0, sizeof(uint64_t) * OMPI_OSC_UCX_POST_PEER_MAX);
        state_base->complete_count = 0;
        state_base->req_flag = 0;
        state_base->acc_lock = TARGET_LOCK_UNLOCKED;
        state_base->dynamic_win_count = 0;

        if (ret != OPAL_SUCCESS) {
            MCA_COMMON_UCX_VERBOSE(1, "_comm_ucx_mem_map failed: %d", ret);
            ucp_rkey_buffer_release(data_rkey_addr);
            ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, data_memh);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }

        status = ucp_rkey_pack(mca_osc_ucx_component.wpool->ucp_ctx, state_memh,
                              &state_rkey_addr, &state_rkey_addr_len);

        if (status != UCS_OK) {
            MCA_COMMON_UCX_VERBOSE(1, "ucp_rkey_pack failed: %d", status);
            ucp_rkey_buffer_release(data_rkey_addr);
            ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, data_memh);
            ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, state_memh);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
    }

    size_t handle_size = sizeof(ompi_osc_ucx_memhandle_t)
                         //+ mca_osc_ucx_component.wpool->recv_waddr_len
                         + data_rkey_addr_len
                         + sizeof(ucp_mem_h) // need to store the local memory handle for cleanup
                         ;
    if (need_state) {
        handle_size += state_rkey_addr_len + sizeof(ucp_mem_h);
    }

    /* now compute the size we need */
    static bool size_printed = false;
    if (!size_printed) {
        printf("Total memhandle size: %zu\n", handle_size);
        size_printed = true;
    }

    ompi_osc_ucx_memhandle_t *ucx_handle = (ompi_osc_ucx_memhandle_t*)memhandle;
    //ucx_handle->recv_worker_addr_len = mca_osc_ucx_component.wpool->recv_waddr_len;
    if (data_rkey_addr_len > UINT16_MAX || state_rkey_addr_len > UINT16_MAX) {
        printf("WARN: UCX rkeys are too large for this implementation!\n");
        ucp_rkey_buffer_release(data_rkey_addr);
        ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, data_memh);
        ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, state_memh);
        return OMPI_ERR_NOT_SUPPORTED;
    }
    ucx_handle->flags = 0;
    ucx_handle->data_rkey_size = data_rkey_addr_len;
    ucx_handle->state_rkey_size = state_rkey_addr_len;
    ucx_handle->data_addr = (uint64_t)base;
    ucx_handle->state_addr = (uint64_t)state_base;
    size_t offset = 0;
    //memcpy(ucx_handle->_data, mca_osc_ucx_component.wpool->recv_waddr, ucx_handle->recv_worker_addr_len);
    //offset += ucx_handle->recv_worker_addr_len;
    memcpy(ucx_handle->_data + offset, data_rkey_addr, data_rkey_addr_len);
    offset += ucx_handle->data_rkey_size;
    memcpy(ucx_handle->_data + offset, &data_memh, sizeof(data_memh));
    ucp_rkey_buffer_release(data_rkey_addr);

    if (need_state) {
        offset += sizeof(data_memh);
        memcpy(ucx_handle->_data + offset, state_rkey_addr, state_rkey_addr_len);
        offset += ucx_handle->state_rkey_size;
        memcpy(ucx_handle->_data + offset, &state_memh, sizeof(state_memh));

        ucx_handle->flags |= OSC_MEMHANDLE_HAS_STATE;
        ucp_rkey_buffer_release(state_rkey_addr);
    }

    *memhandle_size = handle_size;

    return OMPI_SUCCESS;
}

int ompi_osc_ucx_release_memhandle(char memhandle[], ompi_win_t *parentwin)
{
    (void)parentwin;

    ompi_osc_ucx_memhandle_t *ucx_handle = (ompi_osc_ucx_memhandle_t*)memhandle;

    //void *data_rkey_addr;
    //data_rkey_addr = ucx_handle->_data + ucx_handle->recv_worker_addr_len;

    //size_t memh_offset = mca_osc_ucx_component.wpool->recv_waddr_len
    //                     + ucx_handle->data_rkey_size;
    size_t memh_offset = ucx_handle->data_rkey_size;

    /* Unmap data segment */
    ucp_mem_h data_mem_h = *(ucp_mem_h*)(ucx_handle->_data + memh_offset);
    ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, data_mem_h);
    if (ucx_handle->flags & OSC_MEMHANDLE_HAS_STATE) {
        memh_offset += ucx_handle->state_rkey_size + sizeof(ucp_mem_h);
        ucp_mem_h state_mem_h = *(ucp_mem_h*)(ucx_handle->_data + memh_offset);
        /* Unmap state segment */
        ucp_mem_unmap(mca_osc_ucx_component.wpool->ucp_ctx, state_mem_h);
    }

    return OMPI_SUCCESS;
}

int ompi_osc_find_attached_region_position(ompi_osc_dynamic_win_info_t *dynamic_wins,
                                           int min_index, int max_index,
                                           uint64_t base, size_t len, int *insert) {
    int mid_index = (max_index + min_index) >> 1;

    if (min_index > max_index) {
        (*insert) = min_index;
        return -1;
    }

    if (dynamic_wins[mid_index].base > base) {
        return ompi_osc_find_attached_region_position(dynamic_wins, min_index, mid_index-1,
                                                      base, len, insert);
    } else if (base + len < dynamic_wins[mid_index].base + dynamic_wins[mid_index].size) {
        return mid_index;
    } else {
        return ompi_osc_find_attached_region_position(dynamic_wins, mid_index+1, max_index,
                                                      base, len, insert);
    }
}

int ompi_osc_ucx_win_attach(struct ompi_win_t *win, void *base, size_t len) {
    ompi_osc_ucx_module_t *module = (ompi_osc_ucx_module_t*) win->w_osc_module;
    int insert_index = -1, contain_index;
    int ret = OMPI_SUCCESS;

    if (module->state.dynamic_win_count >= OMPI_OSC_UCX_ATTACH_MAX) {
        return OMPI_ERR_TEMP_OUT_OF_RESOURCE;
    }

    if (module->state.dynamic_win_count > 0) {
        contain_index = ompi_osc_find_attached_region_position((ompi_osc_dynamic_win_info_t *)module->state.dynamic_wins,
                                                               0, (int)module->state.dynamic_win_count,
                                                               (uint64_t)base, len, &insert_index);
        if (contain_index >= 0) {
            module->local_dynamic_win_info[contain_index].refcnt++;
            return ret;
        }

        assert(insert_index >= 0 && (uint64_t)insert_index < module->state.dynamic_win_count);

        memmove((void *)&module->local_dynamic_win_info[insert_index+1],
                (void *)&module->local_dynamic_win_info[insert_index],
                (OMPI_OSC_UCX_ATTACH_MAX - (insert_index + 1)) * sizeof(ompi_osc_local_dynamic_win_info_t));
        memmove((void *)&module->state.dynamic_wins[insert_index+1],
                (void *)&module->state.dynamic_wins[insert_index],
                (OMPI_OSC_UCX_ATTACH_MAX - (insert_index + 1)) * sizeof(ompi_osc_dynamic_win_info_t));
    } else {
        insert_index = 0;
    }

    ret = opal_common_ucx_wpmem_create(module->ctx, &base, len,
                                       OPAL_COMMON_UCX_MEM_MAP, &exchange_len_info,
                                       (void *)module->comm,
                                       &(module->local_dynamic_win_info[insert_index].my_mem_addr),
                                       &(module->local_dynamic_win_info[insert_index].my_mem_addr_size),
                                       &(module->local_dynamic_win_info[insert_index].mem));
    if (ret != OMPI_SUCCESS) {
        return ret;
    }

    module->state.dynamic_wins[insert_index].base = (uint64_t)base;
    module->state.dynamic_wins[insert_index].size = len;

    memcpy((char *)(module->state.dynamic_wins[insert_index].mem_addr),
           (char *)module->local_dynamic_win_info[insert_index].my_mem_addr,
           module->local_dynamic_win_info[insert_index].my_mem_addr_size);

    module->local_dynamic_win_info[insert_index].refcnt++;
    module->state.dynamic_win_count++;

    return ret;
}

int ompi_osc_ucx_win_detach(struct ompi_win_t *win, const void *base) {
    ompi_osc_ucx_module_t *module = (ompi_osc_ucx_module_t*) win->w_osc_module;
    int insert, contain;

    assert(module->state.dynamic_win_count > 0);

    contain = ompi_osc_find_attached_region_position((ompi_osc_dynamic_win_info_t *)module->state.dynamic_wins,
                                                     0, (int)module->state.dynamic_win_count,
                                                     (uint64_t)base, 1, &insert);
    assert(contain >= 0 && (uint64_t)contain < module->state.dynamic_win_count);

    /* if we can't find region - just exit */
    if (contain < 0) {
        return OMPI_SUCCESS;
    }

    module->local_dynamic_win_info[contain].refcnt--;
    if (module->local_dynamic_win_info[contain].refcnt == 0) {
        opal_common_ucx_wpmem_free(module->local_dynamic_win_info[contain].mem);
        memmove((void *)&(module->local_dynamic_win_info[contain]),
                (void *)&(module->local_dynamic_win_info[contain+1]),
                (OMPI_OSC_UCX_ATTACH_MAX - (contain + 1)) * sizeof(ompi_osc_local_dynamic_win_info_t));
        memmove((void *)&module->state.dynamic_wins[contain],
                (void *)&module->state.dynamic_wins[contain+1],
                (OMPI_OSC_UCX_ATTACH_MAX - (contain + 1)) * sizeof(ompi_osc_dynamic_win_info_t));

        module->state.dynamic_win_count--;
    }

    return OMPI_SUCCESS;
}

int ompi_osc_ucx_free(struct ompi_win_t *win) {
    ompi_osc_ucx_module_t *module = (ompi_osc_ucx_module_t*) win->w_osc_module;
    int ret = OMPI_SUCCESS;

    if (module->flavor == MPI_WIN_FLAVOR_MEMHANDLE) {
        opal_common_ucx_wpctx_release(module->ctx);
        module->ctx = NULL;
        /* need to clean up the memory tls */
        opal_common_ucx_ep_rkey_t *pair;
        while (NULL != (pair = (opal_common_ucx_ep_rkey_t *)opal_list_remove_first(&module->mem->thread_rkey_list))) {
            OBJ_RELEASE(pair); // rkey destruction is done in the dtor
        }
        if (NULL != module->state_mem) {
            while (NULL != (pair = (opal_common_ucx_ep_rkey_t *)opal_list_remove_first(&module->state_mem->thread_rkey_list))) {
                OBJ_RELEASE(pair); // rkey destruction is done in the dtor
            }
        }

        opal_lifo_push(&module_free_list, &module->free_list_item);
        return OMPI_SUCCESS;
    }

    if (!module->always_lock_shared) {
        assert(module->lock_count == 0);
    }
    assert(opal_list_is_empty(&module->pending_posts) == true);
    if(!module->no_locks) {
        OBJ_DESTRUCT(&module->outstanding_locks);
    }
    OBJ_DESTRUCT(&module->pending_posts);

    opal_common_ucx_wpmem_flush(module->mem, OPAL_COMMON_UCX_SCOPE_WORKER, 0);

    if (module->flavor != MPI_WIN_FLAVOR_MEMHANDLE) {
        ret = module->comm->c_coll->coll_barrier(module->comm,
                                                module->comm->c_coll->coll_barrier_module);
        if (ret != OMPI_SUCCESS) {
            return ret;
        }
        ompi_comm_free(&module->comm);
        opal_common_ucx_wpmem_free(module->state_mem);
        opal_common_ucx_wpmem_free(module->mem);
    }

    free(module->addrs);
    free(module->state_addrs);


    opal_common_ucx_wpctx_release(module->ctx);

    if (module->disp_units) {
        free(module->disp_units);
    }

    free(module);
    ompi_osc_ucx_unregister_progress();

    return ret;
}
