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
#include <string.h>
#include "coll_sched.h"
#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"

static int mca_coll_sched_module_enable(mca_coll_base_module_t *module,
                                         struct ompi_communicator_t *comm);
static int mca_coll_sched_module_disable(mca_coll_base_module_t *module,
                                          struct ompi_communicator_t *comm);

int mca_coll_sched_init_query(bool enable_progress_threads, bool enable_mpi_threads)
{
    return OMPI_SUCCESS;
}

mca_coll_base_module_t *
mca_coll_sched_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    /* Only intra-communicators; algorithms assume intra semantics */
    if (OMPI_COMM_IS_INTER(comm)) {
        return NULL;
    }
    /* Need at least 2 processes */
    if (ompi_comm_size(comm) < 2) {
        return NULL;
    }

    mca_coll_sched_module_t *m = OBJ_NEW(mca_coll_sched_module_t);
    if (NULL == m) {
        return NULL;
    }

    *priority = mca_coll_sched_priority;

    m->super.coll_module_enable  = mca_coll_sched_module_enable;
    m->super.coll_module_disable = mca_coll_sched_module_disable;

    m->super.coll_allreduce = mca_coll_sched_allreduce_intra;
    m->super.coll_reduce    = mca_coll_sched_reduce_intra;
    m->super.coll_bcast     = mca_coll_sched_bcast_intra;

    return &m->super;
}

/* Save current binding for api into module's c_coll, then install our function. */
#define SCHED_INSTALL(comm, m, api, fn)                                        \
    do {                                                                        \
        if ((comm)->c_coll->coll_##api) {                                       \
            MCA_COLL_SAVE_API(comm, api,                                        \
                              (m)->c_coll.coll_##api,                           \
                              (m)->c_coll.coll_##api##_module, "sched");        \
            MCA_COLL_INSTALL_API(comm, api, fn, &(m)->super, "sched");          \
        }                                                                       \
    } while (0)

/* Restore the saved binding if we still own it. */
#define SCHED_UNINSTALL(comm, m, api)                                           \
    do {                                                                        \
        if (&(m)->super == (comm)->c_coll->coll_##api##_module) {               \
            MCA_COLL_INSTALL_API(comm, api,                                     \
                                 (m)->c_coll.coll_##api,                        \
                                 (m)->c_coll.coll_##api##_module, "sched");     \
            (m)->c_coll.coll_##api##_module = NULL;                             \
            (m)->c_coll.coll_##api          = NULL;                             \
        }                                                                       \
    } while (0)

static int
mca_coll_sched_module_enable(mca_coll_base_module_t *module,
                              struct ompi_communicator_t *comm)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;

    SCHED_INSTALL(comm, m, allreduce, mca_coll_sched_allreduce_intra);
    SCHED_INSTALL(comm, m, reduce,    mca_coll_sched_reduce_intra);
    SCHED_INSTALL(comm, m, bcast,     mca_coll_sched_bcast_intra);

    /* Create PML executor as the default backend */
    m->executors[0] = ompi_coll_sched_exec_pml_create();
    if (NULL == m->executors[0]) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    m->num_executors = 1;

    return OMPI_SUCCESS;
}

static int
mca_coll_sched_module_disable(mca_coll_base_module_t *module,
                               struct ompi_communicator_t *comm)
{
    mca_coll_sched_module_t *m = (mca_coll_sched_module_t *) module;

    SCHED_UNINSTALL(comm, m, allreduce);
    SCHED_UNINSTALL(comm, m, reduce);
    SCHED_UNINSTALL(comm, m, bcast);

    /* Free cached schedules */
    for (int i = 0; i < m->cache_count; i++) {
        ompi_coll_sched_free(m->cache[i].sched);
        m->cache[i].sched = NULL;
    }
    m->cache_count = 0;

    /* Free executors */
    for (int i = 0; i < m->num_executors; i++) {
        if (m->executors[i] && m->executors[i]->free) {
            m->executors[i]->free(m->executors[i]);
            m->executors[i] = NULL;
        }
    }
    m->num_executors = 0;

    return OMPI_SUCCESS;
}
