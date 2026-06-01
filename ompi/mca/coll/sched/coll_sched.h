/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_SCHED_H
#define MCA_COLL_SCHED_H

#include "ompi_config.h"
#include "mpi.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/request/request.h"
#include "opal/class/opal_object.h"

BEGIN_C_DECLS

/* ── Buffer reference ──────────────────────────────────────────────────────
 *
 * A bufref describes a contiguous region of a buffer.  The buffer is
 * logically divided into num_slots equal partitions; this reference
 * addresses partition [slot].
 *
 * At execute time the executor resolves it:
 *   unit     = ceil(count / num_slots)
 *   offset   = slot * unit             (in elements)
 *   n_elems  = min(unit, count - offset)
 *
 * num_slots == 1 selects the entire buffer (unit = count).
 */

#define OMPI_COLL_SCHED_BUF_SEND  0   /* user sbuf */
#define OMPI_COLL_SCHED_BUF_RECV  1   /* user rbuf */
#define OMPI_COLL_SCHED_BUF_TEMP(n) (2 + (n))

typedef struct {
    int buf_id;    /* OMPI_COLL_SCHED_BUF_* or OMPI_COLL_SCHED_BUF_TEMP(n) */
    int slot;      /* which partition [0 .. num_slots-1] */
    int num_slots; /* total equal partitions; 1 = whole buffer */
} ompi_coll_sched_bufref_t;

#define ompi_coll_sched_bufref_whole(buf_id_) \
    ((ompi_coll_sched_bufref_t){(buf_id_), 0, 1})

#define ompi_coll_sched_bufref_slot(buf_id_, slot_, num_slots_) \
    ((ompi_coll_sched_bufref_t){(buf_id_), (slot_), (num_slots_)})

/* ── Operation ─────────────────────────────────────────────────────────── */

typedef enum {
    OMPI_COLL_SCHED_OP_SEND,     /* non-blocking point-to-point send   */
    OMPI_COLL_SCHED_OP_RECV,     /* non-blocking point-to-point recv   */
    OMPI_COLL_SCHED_OP_REDUCE,   /* local 2-buffer reduction: dst = op(src, dst) */
    OMPI_COLL_SCHED_OP_REDUCE3,  /* local 3-buffer reduction: dst = op(src1, src2) */
    OMPI_COLL_SCHED_OP_COPY,     /* local buffer copy                  */
} ompi_coll_sched_optype_t;

typedef struct {
    ompi_coll_sched_optype_t type;
    int                      comm_slot; /* index into comms[] passed to execute */
    union {
        struct {
            ompi_coll_sched_bufref_t buf;
            int                      peer;
        } send, recv;
        struct {
            ompi_coll_sched_bufref_t src;
            ompi_coll_sched_bufref_t dst;
        } reduce, copy;
        struct {
            ompi_coll_sched_bufref_t src1; /* received data */
            ompi_coll_sched_bufref_t src2; /* own contribution */
            ompi_coll_sched_bufref_t dst;  /* accumulation target */
        } reduce3;
    };
} ompi_coll_sched_op_t;

/* ── Step ──────────────────────────────────────────────────────────────────
 *
 * A step groups ops that may be initiated together.  Network ops (send/recv)
 * are issued non-blocking; the executor waits for them before executing
 * local ops (reduce/copy) within the same step.  barrier=true additionally
 * guarantees all work in this step completes before the next step starts
 * (relevant when a later step reads data written by this step's local ops).
 */
typedef struct {
    int                   num_ops;
    ompi_coll_sched_op_t *ops;
    bool                  barrier;
} ompi_coll_sched_step_t;

/* ── Schedule ──────────────────────────────────────────────────────────────
 *
 * Temp buffer sizing:
 *   temp_full_size[i] == true  → count elements   (whole message)
 *   temp_full_size[i] == false → ceil(count/n) elements  (one chunk)
 * where n = ompi_comm_size(comms[0]).
 */

#define OMPI_COLL_SCHED_MAX_TEMP_BUFS 4

typedef struct ompi_coll_sched_t {
    int                    num_steps;
    ompi_coll_sched_step_t *steps;

    int                    num_temp_bufs;
    bool                   temp_full_size[OMPI_COLL_SCHED_MAX_TEMP_BUFS];

    int                    num_comm_slots; /* 1 for flat; 2+ for hierarchical */
} ompi_coll_sched_t;

/* ── Alloc / free / builders ────────────────────────────────────────────── */

ompi_coll_sched_t *ompi_coll_sched_alloc(int num_steps);
void               ompi_coll_sched_free(ompi_coll_sched_t *sched);

/* Initialise step step_idx with capacity for num_ops operations. */
int  ompi_coll_sched_step_init(ompi_coll_sched_t *sched, int step_idx,
                                int num_ops, bool barrier);

/* Register a temp buffer; returns its buf_id (2+) or OMPI_ERR_OUT_OF_RESOURCE */
int  ompi_coll_sched_add_temp_buf(ompi_coll_sched_t *sched, bool full_size);

/* Fill op slot op_idx inside step step_idx */
void ompi_coll_sched_op_send  (ompi_coll_sched_t *s, int step, int op_idx,
                                int comm_slot, int peer, ompi_coll_sched_bufref_t buf);
void ompi_coll_sched_op_recv  (ompi_coll_sched_t *s, int step, int op_idx,
                                int comm_slot, int peer, ompi_coll_sched_bufref_t buf);
void ompi_coll_sched_op_reduce(ompi_coll_sched_t *s, int step, int op_idx,
                                ompi_coll_sched_bufref_t src, ompi_coll_sched_bufref_t dst);
void ompi_coll_sched_op_reduce3(ompi_coll_sched_t *s, int step, int op_idx,
                                 ompi_coll_sched_bufref_t src1,
                                 ompi_coll_sched_bufref_t src2,
                                 ompi_coll_sched_bufref_t dst);
void ompi_coll_sched_op_copy  (ompi_coll_sched_t *s, int step, int op_idx,
                                ompi_coll_sched_bufref_t src, ompi_coll_sched_bufref_t dst);

/* ── Algorithm builders ─────────────────────────────────────────────────── */

ompi_coll_sched_t *ompi_coll_sched_build_allreduce_ring(int rank, int n);
ompi_coll_sched_t *ompi_coll_sched_build_allreduce_recursivedoubling(int rank, int n);
ompi_coll_sched_t *ompi_coll_sched_build_reduce_binomial(int rank, int n, int root);
ompi_coll_sched_t *ompi_coll_sched_build_bcast_binomial(int rank, int n, int root);

/* ── Executor interface ─────────────────────────────────────────────────── */

struct ompi_coll_sched_exec_t;

typedef bool (*ompi_coll_sched_exec_can_fn_t)(
    struct ompi_coll_sched_exec_t *exec,
    const ompi_coll_sched_t       *sched,
    struct ompi_communicator_t   **comms,
    struct ompi_datatype_t        *dtype,
    struct ompi_op_t              *op);

typedef int (*ompi_coll_sched_exec_run_fn_t)(
    struct ompi_coll_sched_exec_t *exec,
    const ompi_coll_sched_t       *sched,
    struct ompi_communicator_t   **comms,
    const void *sbuf, void *rbuf,
    size_t count,
    struct ompi_datatype_t        *dtype,
    struct ompi_op_t              *op,
    int                            base_tag);

typedef void (*ompi_coll_sched_exec_free_fn_t)(struct ompi_coll_sched_exec_t *exec);

typedef struct ompi_coll_sched_exec_t {
    ompi_coll_sched_exec_can_fn_t  can_execute;
    ompi_coll_sched_exec_run_fn_t  execute;
    ompi_coll_sched_exec_free_fn_t free;
} ompi_coll_sched_exec_t;

ompi_coll_sched_exec_t *ompi_coll_sched_exec_pml_create(void);

/* ── Schedule cache ─────────────────────────────────────────────────────── */

typedef struct {
    int                colltype; /* COLLTYPE_T value */
    int                variant;
    int                root;     /* -1 = root-independent */
    int                param;    /* radix or other algo param; 0 if unused */
    ompi_coll_sched_t *sched;
} ompi_coll_sched_cache_entry_t;

#define OMPI_COLL_SCHED_CACHE_SIZE 16

/* ── Module ─────────────────────────────────────────────────────────────── */

#define OMPI_COLL_SCHED_MAX_EXECUTORS 4

typedef struct mca_coll_sched_module_t {
    mca_coll_base_module_t    super;
    mca_coll_base_comm_coll_t c_coll;   /* saved lower-priority functions */

    /* Topology sub-communicators; NULL if single-node / unavailable */
    struct ompi_communicator_t *local_comm;  /* MPI_COMM_TYPE_SHARED */
    struct ompi_communicator_t *socket_comm; /* OMPI_COMM_TYPE_SOCKET */
    struct ompi_communicator_t *numa_comm;   /* OMPI_COMM_TYPE_NUMA */
    struct ompi_communicator_t *l3_comm;     /* OMPI_COMM_TYPE_L3CACHE */
    int                         num_nodes;

    /* Schedule cache (linear scan, small N) */
    ompi_coll_sched_cache_entry_t cache[OMPI_COLL_SCHED_CACHE_SIZE];
    int                           cache_count;

    /* Executor chain: tried in order; first can_execute() match wins */
    ompi_coll_sched_exec_t *executors[OMPI_COLL_SCHED_MAX_EXECUTORS];
    int                     num_executors;
} mca_coll_sched_module_t;

OMPI_DECLSPEC OBJ_CLASS_DECLARATION(mca_coll_sched_module_t);

ompi_coll_sched_t *ompi_coll_sched_cache_get(mca_coll_sched_module_t *m,
                                               int colltype, int variant,
                                               int root, int param);
int ompi_coll_sched_cache_put(mca_coll_sched_module_t *m,
                               int colltype, int variant,
                               int root, int param,
                               ompi_coll_sched_t *sched);

ompi_coll_sched_exec_t *ompi_coll_sched_select_exec(
    mca_coll_sched_module_t  *m,
    const ompi_coll_sched_t  *sched,
    struct ompi_communicator_t **comms,
    struct ompi_datatype_t    *dtype,
    struct ompi_op_t          *op);

/* ── Component globals ──────────────────────────────────────────────────── */

OMPI_DECLSPEC extern const mca_coll_base_component_3_0_0_t mca_coll_sched_component;
extern int mca_coll_sched_priority;

int mca_coll_sched_init_query(bool enable_progress_threads, bool enable_mpi_threads);
mca_coll_base_module_t *mca_coll_sched_comm_query(struct ompi_communicator_t *comm,
                                                    int *priority);

/* Per-collective dispatch functions */
int mca_coll_sched_allreduce_intra(ALLREDUCE_ARGS);
int mca_coll_sched_reduce_intra(REDUCE_ARGS);
int mca_coll_sched_bcast_intra(BCAST_ARGS);

END_C_DECLS

#endif /* MCA_COLL_SCHED_H */
