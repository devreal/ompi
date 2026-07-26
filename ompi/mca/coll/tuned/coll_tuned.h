/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2019      Mellanox Technologies. All rights reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_TUNED_EXPORT_H
#define MCA_COLL_TUNED_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/request/request.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/output.h"
#include "opal/mca/accelerator/accelerator.h"
#include "opal/mca/accelerator/base/base.h"
#include "ompi/op/op_gpu_session.h"

/* also need the dynamic rule structures */
#include "coll_tuned_dynamic_rules.h"

BEGIN_C_DECLS

#define COLL_TUNED_TRACING_VERBOSE 50

/* these are the same across all modules and are loaded at component query time */
extern int   ompi_coll_tuned_stream;
extern int   ompi_coll_tuned_priority;
extern bool  ompi_coll_tuned_use_dynamic_rules;
extern char* ompi_coll_tuned_dynamic_rules_filename;
extern int   ompi_coll_tuned_init_tree_fanout;
extern int   ompi_coll_tuned_init_chain_fanout;
extern int   ompi_coll_tuned_init_max_requests;
extern int   ompi_coll_tuned_alltoall_max_requests;
extern int   ompi_coll_tuned_scatter_intermediate_msg;
extern int   ompi_coll_tuned_scatter_large_msg;
extern int   ompi_coll_tuned_scatter_min_procs;
extern int   ompi_coll_tuned_scatter_blocking_send_ratio;

/* Message size (bytes), below which a reduction on device-resident buffers
 * is done by staging through host memory (plain CPU op) rather than driving
 * the GPU op component's persistent kernel. Below the crossover, the
 * host<->device control-slot round trip a persistent kernel needs (see
 * ompi/op/op_gpu_session.h) is expected to cost more than just bouncing the
 * (small) buffers to host and reducing them there; above it, avoiding a full
 * buffer copy in each direction is expected to win. Default 0 preserves the
 * original always-use-the-device behavior until an estimate is computed (see
 * ompi_coll_tuned_gpu_get_threshold() in coll_tuned_gpu.c) or the user sets
 * this explicitly, either of which takes precedence over the default. */
extern size_t ompi_coll_tuned_gpu_reduce_threshold;
/* mca_base_var index of the variable above -- used by
 * ompi_coll_tuned_gpu_get_threshold() to tell whether the user explicitly
 * set it (mca_base_var_get_value's source output) vs. it's still at the
 * compiled-in default. */
extern int ompi_coll_tuned_gpu_reduce_threshold_index;

/* Analytical-estimate inputs -- see coll_tuned_gpu.c. */
extern int ompi_coll_tuned_gpu_host_reduce_bw_mbs;
extern int ompi_coll_tuned_gpu_ctrl_latency_usec;
extern int ompi_coll_tuned_gpu_host_stage_latency_usec;

/*
 * Return the reduction device-vs-host threshold (bytes) to use for dev_id.
 *
 * If the user has explicitly set coll_tuned_gpu_reduce_threshold (via --mca,
 * environment, or a file), that value always wins and is returned as-is,
 * regardless of dev_id. Otherwise, this analytically estimates a crossover
 * from dev_id's device/PCIe bandwidth (queried once per device and cached)
 * and the gpu_host_reduce_bw_mbs/gpu_ctrl_latency_usec/
 * gpu_host_stage_latency_usec tunables; if the bandwidth query isn't
 * supported (e.g. no matching op component, or PCIe link info unavailable),
 * falls back to the compiled-in default (0).
 */
size_t ompi_coll_tuned_gpu_get_threshold(int dev_id);

/*
 * COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, sbuf_bytes, rbuf_bytes,
 *                              cmp_bytes, rc, do_this_call)
 *
 * Decides, for one reduction-based collective invocation, whether to run on
 * the device or stage through the host, then performs whichever the
 * decision calls for and invokes do_this_call exactly once. sbuf_bytes and
 * rbuf_bytes may differ (e.g. reduce_scatter: sbuf holds the full
 * pre-reduction array, rbuf only this rank's post-scatter slice); cmp_bytes
 * is what gets compared against the threshold.
 *
 *   - sbuf/rbuf not device memory: unchanged from before this macro existed
 *     -- do_this_call runs with the original pointers and session == NULL.
 *   - device memory, cmp_bytes >= ompi_coll_tuned_gpu_reduce_threshold:
 *     unchanged -- a GPU session is created (as today) and passed to
 *     do_this_call via `session`.
 *   - device memory, cmp_bytes < ompi_coll_tuned_gpu_reduce_threshold:
 *     sbuf/rbuf are copied to host scratch buffers (sbuf_bytes/rbuf_bytes
 *     each), do_this_call runs against those with session == NULL (so the
 *     algorithm takes its normal host-op code path), and the result is
 *     copied back into the caller's original (device) rbuf afterward.
 *
 * do_this_call must reference the buffers as `_sbuf`/`_rbuf` and the session
 * as `session` -- both are bound by this macro in its scope. sbuf may be
 * MPI_IN_PLACE.
 *
 * COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, total_bytes, rc, do_this_call)
 * is the common case where sbuf and rbuf are the same total size (allreduce,
 * reduce, scan, exscan).
 */
#define COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, sbuf_bytes, rbuf_bytes,             \
                                      cmp_bytes, rc, do_this_call)                              \
    do {                                                                                       \
        ompi_op_gpu_session_t *session = NULL;                                                 \
        int _cgd_dev_id = MCA_ACCELERATOR_NO_DEVICE_ID;                                        \
        uint64_t _cgd_flags;                                                                   \
        bool _cgd_is_device =                                                                  \
            (((sbuf) != MPI_IN_PLACE &&                                                        \
              opal_accelerator.check_addr((sbuf), &_cgd_dev_id, &_cgd_flags) > 0) ||            \
             opal_accelerator.check_addr((rbuf), &_cgd_dev_id, &_cgd_flags) > 0);               \
        void *_cgd_hsbuf = NULL, *_cgd_hrbuf = NULL;                                            \
        const void *_sbuf = (sbuf);                                                             \
        void *_rbuf = (rbuf);                                                                   \
        (rc) = OMPI_SUCCESS;                                                                    \
        if (_cgd_is_device && (cmp_bytes) < ompi_coll_tuned_gpu_get_threshold(_cgd_dev_id)) {   \
            _cgd_hrbuf = malloc(rbuf_bytes);                                                    \
            if ((sbuf) != MPI_IN_PLACE) { _cgd_hsbuf = malloc(sbuf_bytes); }                    \
            if (NULL == _cgd_hrbuf || ((sbuf) != MPI_IN_PLACE && NULL == _cgd_hsbuf)) {         \
                (rc) = OMPI_ERR_OUT_OF_RESOURCE;                                                \
            } else {                                                                            \
                if (NULL != _cgd_hsbuf) {                                                       \
                    opal_accelerator.mem_copy(MCA_ACCELERATOR_NO_DEVICE_ID, _cgd_dev_id,        \
                                               _cgd_hsbuf, (sbuf), (sbuf_bytes),                \
                                               MCA_ACCELERATOR_TRANSFER_DTOH);                  \
                    _sbuf = _cgd_hsbuf;                                                         \
                }                                                                               \
                opal_accelerator.mem_copy(MCA_ACCELERATOR_NO_DEVICE_ID, _cgd_dev_id,            \
                                           _cgd_hrbuf, (rbuf), (rbuf_bytes),                    \
                                           MCA_ACCELERATOR_TRANSFER_DTOH);                      \
                _rbuf = _cgd_hrbuf;                                                             \
            }                                                                                   \
        } else if (_cgd_is_device) {                                                            \
            session = ompi_op_gpu_session_begin((op), (dtype), _cgd_dev_id);                    \
        }                                                                                        \
        if (OMPI_SUCCESS == (rc)) {                                                             \
            (rc) = do_this_call;                                                                \
            ompi_op_gpu_session_end(session);                                                   \
            if (NULL != _cgd_hrbuf) {                                                           \
                opal_accelerator.mem_copy(_cgd_dev_id, MCA_ACCELERATOR_NO_DEVICE_ID, (rbuf),     \
                                           _cgd_hrbuf, (rbuf_bytes),                            \
                                           MCA_ACCELERATOR_TRANSFER_HTOD);                      \
            }                                                                                    \
        }                                                                                        \
        if (NULL != _cgd_hrbuf) { free(_cgd_hrbuf); }                                          \
        if (NULL != _cgd_hsbuf) { free(_cgd_hsbuf); }                                          \
    } while (0)

#define COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, total_bytes, rc, do_this_call)           \
    COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, total_bytes, total_bytes,               \
                                  total_bytes, rc, do_this_call)

/* forced algorithm choices */
/* this structure is for storing the indexes to the forced algorithm mca params... */
/* we get these at component query (so that registered values appear in ompi_infoi) */
struct coll_tuned_force_algorithm_mca_param_indices_t {
    int  algorithm_param_index;      /* which algorithm you want to force */
    int  segsize_param_index;        /* segsize to use (if supported), 0 = no segmentation */
    int  tree_fanout_param_index;    /* tree fanout/in to use */
    int  chain_fanout_param_index;   /* K-chain fanout/in to use */
    int  max_requests_param_index;   /* Maximum number of outstanding send or recv requests */
};
typedef struct coll_tuned_force_algorithm_mca_param_indices_t coll_tuned_force_algorithm_mca_param_indices_t;


/* the following type is for storing actual value obtained from the MCA on each tuned module */
/* via their mca param indices lookup in the component */
/* this structure is stored once per collective type per communicator... */
struct coll_tuned_force_algorithm_params_t {
    int  algorithm;      /* which algorithm you want to force */
    int  segsize;        /* segsize to use (if supported), 0 = no segmentation */
    int  tree_fanout;    /* tree fanout/in to use */
    int  chain_fanout;   /* K-chain fanout/in to use */
    int  max_requests;   /* Maximum number of outstanding send or recv requests */
};
typedef struct coll_tuned_force_algorithm_params_t coll_tuned_force_algorithm_params_t;

/* the indices to the MCA params so that modules can look them up at open / comm create time  */
extern coll_tuned_force_algorithm_mca_param_indices_t ompi_coll_tuned_forced_params[COLLCOUNT];
/* the actual max algorithm values (readonly), loaded at component open */
extern int ompi_coll_tuned_forced_max_algorithms[COLLCOUNT];

/*
 * coll API functions
 */

/* API functions */

int ompi_coll_tuned_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads);

mca_coll_base_module_t *
ompi_coll_tuned_comm_query(struct ompi_communicator_t *comm, int *priority);

/* API functions of decision functions and any implementations */

/*
 * Note this gets long as we have to have a prototype for each
 * MPI collective 4 times.. 2 for the comm type and 2 for each decision
 * type.
 * we might cut down the decision prototypes by conditional compiling
 */

/* All Gather */
int ompi_coll_tuned_allgather_intra_dec_fixed(ALLGATHER_ARGS);
int ompi_coll_tuned_allgather_intra_dec_dynamic(ALLGATHER_ARGS);
int ompi_coll_tuned_allgather_intra_do_this(ALLGATHER_ARGS, int algorithm, int faninout, int segsize, mca_allocator_base_module_t *allocator);
int ompi_coll_tuned_allgather_intra_check_forced_init(coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* All GatherV */
int ompi_coll_tuned_allgatherv_intra_dec_fixed(ALLGATHERV_ARGS);
int ompi_coll_tuned_allgatherv_intra_dec_dynamic(ALLGATHERV_ARGS);
int ompi_coll_tuned_allgatherv_intra_do_this(ALLGATHERV_ARGS, int algorithm, int faninout, int segsize);
int ompi_coll_tuned_allgatherv_intra_check_forced_init(coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* All Reduce */
int ompi_coll_tuned_allreduce_intra_dec_fixed(ALLREDUCE_ARGS);
int ompi_coll_tuned_allreduce_intra_disjoint_dec_fixed(ALLREDUCE_ARGS);
int ompi_coll_tuned_allreduce_intra_dec_dynamic(ALLREDUCE_ARGS);
int ompi_coll_tuned_allreduce_intra_do_this(ALLREDUCE_ARGS, int algorithm, int faninout, int segsize, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_allreduce_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* AlltoAll */
int ompi_coll_tuned_alltoall_intra_dec_fixed(ALLTOALL_ARGS);
int ompi_coll_tuned_alltoall_intra_dec_dynamic(ALLTOALL_ARGS);
int ompi_coll_tuned_alltoall_intra_do_this(ALLTOALL_ARGS, int algorithm, int faninout, int segsize, int max_requests);
int ompi_coll_tuned_alltoall_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* AlltoAllV */
int ompi_coll_tuned_alltoallv_intra_dec_fixed(ALLTOALLV_ARGS);
int ompi_coll_tuned_alltoallv_intra_dec_dynamic(ALLTOALLV_ARGS);
int ompi_coll_tuned_alltoallv_intra_do_this(ALLTOALLV_ARGS, int algorithm);
int ompi_coll_tuned_alltoallv_intra_check_forced_init(coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Barrier */
int ompi_coll_tuned_barrier_intra_dec_fixed(BARRIER_ARGS);
int ompi_coll_tuned_barrier_intra_dec_dynamic(BARRIER_ARGS);
int ompi_coll_tuned_barrier_intra_do_this(BARRIER_ARGS, int algorithm, int faninout, int segsize);
int ompi_coll_tuned_barrier_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Bcast */
int ompi_coll_tuned_bcast_intra_dec_fixed(BCAST_ARGS);
int ompi_coll_tuned_bcast_intra_disjoint_dec_fixed(BCAST_ARGS);
int ompi_coll_tuned_bcast_intra_dec_dynamic(BCAST_ARGS);
int ompi_coll_tuned_bcast_intra_do_this(BCAST_ARGS, int algorithm, int faninout, int segsize);
int ompi_coll_tuned_bcast_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Gather */
int ompi_coll_tuned_gather_intra_dec_fixed(GATHER_ARGS);
int ompi_coll_tuned_gather_intra_dec_dynamic(GATHER_ARGS);
int ompi_coll_tuned_gather_intra_do_this(GATHER_ARGS, int algorithm, int faninout, int segsize, mca_allocator_base_module_t *allocator);
int ompi_coll_tuned_gather_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Reduce */
int ompi_coll_tuned_reduce_intra_dec_fixed(REDUCE_ARGS);
int ompi_coll_tuned_reduce_intra_dec_dynamic(REDUCE_ARGS);
int ompi_coll_tuned_reduce_intra_do_this(REDUCE_ARGS, int algorithm, int faninout, int segsize, int max_oustanding_reqs, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_reduce_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Reduce_scatter */
int ompi_coll_tuned_reduce_scatter_intra_dec_fixed(REDUCESCATTER_ARGS);
int ompi_coll_tuned_reduce_scatter_intra_dec_dynamic(REDUCESCATTER_ARGS);
int ompi_coll_tuned_reduce_scatter_intra_do_this(REDUCESCATTER_ARGS, int algorithm, int faninout, int segsize, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_reduce_scatter_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Reduce_scatter_block */
int ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed(REDUCESCATTERBLOCK_ARGS);
int ompi_coll_tuned_reduce_scatter_block_intra_dec_dynamic(REDUCESCATTERBLOCK_ARGS);
int ompi_coll_tuned_reduce_scatter_block_intra_do_this(REDUCESCATTERBLOCK_ARGS, int algorithm, int faninout, int segsize, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_reduce_scatter_block_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Scatter */
int ompi_coll_tuned_scatter_intra_dec_fixed(SCATTER_ARGS);
int ompi_coll_tuned_scatter_intra_dec_dynamic(SCATTER_ARGS);
int ompi_coll_tuned_scatter_intra_do_this(SCATTER_ARGS, int algorithm, int faninout, int segsize, mca_allocator_base_module_t *allocator);
int ompi_coll_tuned_scatter_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Exscan */
int ompi_coll_tuned_exscan_intra_dec_fixed(EXSCAN_ARGS);
int ompi_coll_tuned_exscan_intra_dec_dynamic(EXSCAN_ARGS);
int ompi_coll_tuned_exscan_intra_do_this(EXSCAN_ARGS, int algorithm, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_exscan_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

/* Scan */
int ompi_coll_tuned_scan_intra_dec_fixed(SCAN_ARGS);
int ompi_coll_tuned_scan_intra_dec_dynamic(SCAN_ARGS);
int ompi_coll_tuned_scan_intra_do_this(SCAN_ARGS, int algorithm, ompi_op_gpu_session_t *session);
int ompi_coll_tuned_scan_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices);

struct mca_coll_tuned_component_t {
	/** Base coll component */
	mca_coll_base_component_3_0_0_t super;

	/** MCA parameter: Priority of this component */
	int tuned_priority;

	/** global stuff that I need the component to store */

	/* MCA parameters first */

	/* cached decision table stuff (moved from MCW module) */
	ompi_coll_alg_rule_t *all_base_rules;
};
/**
 * Convenience typedef
 */
typedef struct mca_coll_tuned_component_t mca_coll_tuned_component_t;

/**
 * Global component instance
 */
OMPI_DECLSPEC extern mca_coll_tuned_component_t mca_coll_tuned_component;

struct mca_coll_tuned_module_t {
    mca_coll_base_module_t super;

    /* for forced algorithms we store the information on the module */
    /* previously we only had one shared copy, ops, it really is per comm/module */
    coll_tuned_force_algorithm_params_t user_forced[COLLCOUNT];

    /* the communicator rules for each MPI collective for ONLY my comsize */
    ompi_coll_com_rule_t *com_rules[COLLCOUNT];
};
typedef struct mca_coll_tuned_module_t mca_coll_tuned_module_t;
OBJ_CLASS_DECLARATION(mca_coll_tuned_module_t);

int coll_tuned_alg_from_str(int collective_id, const char *alg_name, int *alg_index);
int coll_tuned_alg_to_str(int collective_id, int alg_value, char **alg_string);
int coll_tuned_alg_register_options(int collective_id, mca_base_var_enum_t *options);


#endif  /* MCA_COLL_TUNED_EXPORT_H */
