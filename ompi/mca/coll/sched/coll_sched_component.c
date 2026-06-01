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
#include "coll_sched.h"
#include "mpi.h"
#include "ompi/mca/coll/coll.h"

const char *mca_coll_sched_component_version_string =
    "Open MPI schedule-IR collective MCA component version " OMPI_VERSION;

int mca_coll_sched_priority = 20;

static int sched_register(void);

const mca_coll_base_component_3_0_0_t mca_coll_sched_component = {
    .collm_version = {
        MCA_COLL_BASE_VERSION_3_0_0,
        .mca_component_name = "sched",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),
        .mca_register_component_params = sched_register,
    },
    .collm_data = {
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },
    .collm_init_query = mca_coll_sched_init_query,
    .collm_comm_query  = mca_coll_sched_comm_query,
};
MCA_BASE_COMPONENT_INIT(ompi, coll, sched)

static int sched_register(void)
{
    mca_coll_sched_priority = 20;
    (void) mca_base_component_var_register(
        &mca_coll_sched_component.collm_version, "priority",
        "Priority of the sched coll component",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
        OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_ALL,
        &mca_coll_sched_priority);
    return OMPI_SUCCESS;
}

static void sched_module_construct(mca_coll_sched_module_t *m)
{
    memset(&m->c_coll, 0, sizeof(m->c_coll));
    m->local_comm   = NULL;
    m->socket_comm  = NULL;
    m->numa_comm    = NULL;
    m->l3_comm      = NULL;
    m->num_nodes    = 1;
    m->cache_count  = 0;
    m->num_executors = 0;
}

static void sched_module_destruct(mca_coll_sched_module_t *m)
{
    /* schedules in cache are freed by module_disable */
    for (int i = 0; i < m->num_executors; i++) {
        if (m->executors[i] && m->executors[i]->free) {
            m->executors[i]->free(m->executors[i]);
        }
    }
}

OBJ_CLASS_INSTANCE(mca_coll_sched_module_t,
                   mca_coll_base_module_t,
                   sched_module_construct,
                   sched_module_destruct);
