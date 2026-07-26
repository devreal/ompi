/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2020-2025 Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "opal/mca/accelerator/accelerator.h"
#include "opal/mca/accelerator/base/base.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op.h"
#include "ompi/op/op_gpu_session.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "coll_tuned.h"

/*
 * Notes on evaluation rules and ordering
 *
 * The order is:
 *      use file based rules if presented (-coll_tuned_dynamic_rules_filename = rules)
 * Else
 *      use forced rules (-coll_tuned_dynamic_ALG_intra_algorithm = algorithm-number)
 * Else
 *      use fixed (compiled) rule set (or nested ifs)
 *
 */

/*
 *  allreduce_intra
 *
 *  Function:   - allreduce using other MPI collectives
 *  Accepts:    - same as MPI_Allreduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
int
ompi_coll_tuned_allreduce_intra_dec_dynamic (const void *sbuf, void *rbuf, size_t count,
                                             struct ompi_datatype_t *dtype,
                                             struct ompi_op_t *op,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "ompi_coll_tuned_allreduce_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLREDUCE].algorithm) {
        size_t _dsize;
        int rc;
        ompi_datatype_type_size(dtype, &_dsize);
        _dsize *= count;
        COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _dsize, rc,
            ompi_coll_tuned_allreduce_intra_do_this(_sbuf, _rbuf, count, dtype, op, comm, module,
                                                    tuned_module->user_forced[ALLREDUCE].algorithm,
                                                    tuned_module->user_forced[ALLREDUCE].tree_fanout,
                                                    tuned_module->user_forced[ALLREDUCE].segsize,
                                                    session));
        return rc;
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[ALLREDUCE]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLREDUCE],
                                                        dsize, &faninout, &segsize, &ignoreme);

        if (alg) {
            int rc;
            COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, dsize, rc,
                ompi_coll_tuned_allreduce_intra_do_this(_sbuf, _rbuf, count, dtype, op,
                                                        comm, module,
                                                        alg, faninout, segsize, session));
            return rc;
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_allreduce_intra_dec_fixed (sbuf, rbuf, count, dtype, op,
                                                      comm, module);
}

/*
 *    alltoall_intra_dec
 *
 *    Function:    - seletects alltoall algorithm to use
 *    Accepts:    - same arguments as MPI_Alltoall()
 *    Returns:    - MPI_SUCCESS or error code (passed from the alltoall implementation)
 */

int ompi_coll_tuned_alltoall_intra_dec_dynamic(const void *sbuf, size_t scount,
                                               struct ompi_datatype_t *sdtype,
                                               void* rbuf, size_t rcount,
                                               struct ompi_datatype_t *rdtype,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "ompi_coll_tuned_alltoall_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLTOALL].algorithm) {
        return ompi_coll_tuned_alltoall_intra_do_this(sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module,
                                                      tuned_module->user_forced[ALLTOALL].algorithm,
                                                      tuned_module->user_forced[ALLTOALL].tree_fanout,
                                                      tuned_module->user_forced[ALLTOALL].segsize,
                                                      tuned_module->user_forced[ALLTOALL].max_requests);
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[ALLTOALL]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int comsize;
        int alg, faninout, segsize, max_requests;
        size_t dsize;

        ompi_datatype_type_size (sdtype, &dsize);
        comsize = ompi_comm_size(comm);
        dsize *= (ptrdiff_t)comsize * (ptrdiff_t)scount;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLTOALL],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return ompi_coll_tuned_alltoall_intra_do_this (sbuf, scount, sdtype,
                                                           rbuf, rcount, rdtype,
                                                           comm, module,
                                                           alg, faninout, segsize, max_requests);
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_alltoall_intra_dec_fixed (sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     comm, module);
}

/*
 *    Function:   - selects alltoallv algorithm to use
 *    Accepts:    - same arguments as MPI_Alltoallv()
 *    Returns:    - MPI_SUCCESS or error code
 */

int ompi_coll_tuned_alltoallv_intra_dec_dynamic(const void *sbuf, ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                                                struct ompi_datatype_t *sdtype,
                                                void* rbuf, ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                                struct ompi_datatype_t *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "ompi_coll_tuned_alltoallv_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLTOALLV].algorithm) {
        return ompi_coll_tuned_alltoallv_intra_do_this(sbuf, scounts, sdisps, sdtype,
                                                       rbuf, rcounts, rdisps, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLTOALLV].algorithm);
    }

    /**
     * check to see if we have some filebased rules. As we don't have global
     * knowledge about the total amount of data, use the first available rule.
     * This allow the users to specify the alltoallv algorithm to be used only
     * based on the communicator size.
     */
    if (tuned_module->com_rules[ALLTOALLV]) {
        int alg, faninout, segsize, max_requests;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLTOALLV],
                                                        0, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return ompi_coll_tuned_alltoallv_intra_do_this (sbuf, scounts, sdisps, sdtype,
                                                            rbuf, rcounts, rdisps, rdtype,
                                                            comm, module,
                                                            alg);
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_alltoallv_intra_dec_fixed(sbuf, scounts, sdisps, sdtype,
                                                     rbuf, rcounts, rdisps, rdtype,
                                                     comm, module);
}

/*
 *    barrier_intra_dec
 *
 *    Function:    - seletects barrier algorithm to use
 *    Accepts:    - same arguments as MPI_Barrier()
 *    Returns:    - MPI_SUCCESS or error code (passed from the barrier implementation)
 */
int ompi_coll_tuned_barrier_intra_dec_dynamic(struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "ompi_coll_tuned_barrier_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[BARRIER].algorithm) {
        return ompi_coll_tuned_barrier_intra_do_this(comm, module,
                                                     tuned_module->user_forced[BARRIER].algorithm,
                                                     tuned_module->user_forced[BARRIER].tree_fanout,
                                                     tuned_module->user_forced[BARRIER].segsize);
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[BARRIER]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[BARRIER],
                                                        0, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return ompi_coll_tuned_barrier_intra_do_this (comm, module,
                                                          alg, faninout, segsize);
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_barrier_intra_dec_fixed (comm, module);
}

/*
 *   bcast_intra_dec
 *
 *   Function:   - selects broadcast algorithm to use
 *   Accepts:   - same arguments as MPI_Bcast()
 *   Returns:   - MPI_SUCCESS or error code (passed from the bcast implementation)
 */
int ompi_coll_tuned_bcast_intra_dec_dynamic(void *buf, size_t count,
                                            struct ompi_datatype_t *dtype, int root,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "coll:tuned:bcast_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[BCAST].algorithm) {
        return ompi_coll_tuned_bcast_intra_do_this(buf, count, dtype,
                                                   root, comm, module,
                                                   tuned_module->user_forced[BCAST].algorithm,
                                                   tuned_module->user_forced[BCAST].chain_fanout,
                                                   tuned_module->user_forced[BCAST].segsize);
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[BCAST]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[BCAST],
                                                        dsize, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return ompi_coll_tuned_bcast_intra_do_this (buf, count, dtype, root,
                                                        comm, module,
                                                        alg, faninout, segsize);
        } /* found a method */
    } /*end if any com rules to check */


    return ompi_coll_tuned_bcast_intra_dec_fixed (buf, count, dtype, root,
                                                  comm, module);
}

/*
 *    reduce_intra_dec
 *
 *    Function:    - seletects reduce algorithm to use
 *    Accepts:    - same arguments as MPI_reduce()
 *    Returns:    - MPI_SUCCESS or error code (passed from the reduce implementation)
 *
 */
int ompi_coll_tuned_reduce_intra_dec_dynamic( const void *sbuf, void *rbuf,
                                              size_t count, struct ompi_datatype_t* dtype,
                                              struct ompi_op_t* op, int root,
                                              struct ompi_communicator_t* comm,
                                              mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "coll:tuned:reduce_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCE].algorithm) {
        size_t _dsize;
        int rc;
        ompi_datatype_type_size(dtype, &_dsize);
        _dsize *= count;
        COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _dsize, rc,
            ompi_coll_tuned_reduce_intra_do_this(_sbuf, _rbuf, count, dtype,
                                                 op, root, comm, module,
                                                 tuned_module->user_forced[REDUCE].algorithm,
                                                 tuned_module->user_forced[REDUCE].chain_fanout,
                                                 tuned_module->user_forced[REDUCE].segsize,
                                                 tuned_module->user_forced[REDUCE].max_requests,
                                                 session));
        return rc;
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCE]) {

        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, max_requests;
        size_t dsize;

        ompi_datatype_type_size(dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[REDUCE],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            int rc;
            COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, dsize, rc,
                ompi_coll_tuned_reduce_intra_do_this(_sbuf, _rbuf, count, dtype,
                                                     op, root, comm, module,
                                                     alg, faninout,
                                                     segsize, max_requests, session));
            return rc;
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_reduce_intra_dec_fixed (sbuf, rbuf, count, dtype,
                                                   op, root, comm, module);
}

/*
 *    reduce_scatter_intra_dec
 *
 *    Function:   - seletects reduce_scatter algorithm to use
 *    Accepts:    - same arguments as MPI_Reduce_scatter()
 *    Returns:    - MPI_SUCCESS or error code (passed from
 *                  the reduce_scatter implementation)
 *
 */
int ompi_coll_tuned_reduce_scatter_intra_dec_dynamic(const void *sbuf, void *rbuf,
                                                     ompi_count_array_t rcounts,
                                                     struct ompi_datatype_t *dtype,
                                                     struct ompi_op_t *op,
                                                     struct ompi_communicator_t *comm,
                                                     mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "coll:tuned:reduce_scatter_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCESCATTER].algorithm) {
        size_t _dsize, _sbuf_dsize, _rbuf_dsize;
        int rc;
        ompi_datatype_type_size(dtype, &_dsize);
        _sbuf_dsize = 0;
        {
            int _i, _size = ompi_comm_size(comm);
            for (_i = 0; _i < _size; _i++) { _sbuf_dsize += ompi_count_array_get(rcounts, _i); }
        }
        _rbuf_dsize = _dsize * ompi_count_array_get(rcounts, ompi_comm_rank(comm));
        _sbuf_dsize *= _dsize;
        COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, _sbuf_dsize, _rbuf_dsize, _sbuf_dsize, rc,
            ompi_coll_tuned_reduce_scatter_intra_do_this(_sbuf, _rbuf, rcounts, dtype,
                                                         op, comm, module,
                                                         tuned_module->user_forced[REDUCESCATTER].algorithm,
                                                         tuned_module->user_forced[REDUCESCATTER].chain_fanout,
                                                         tuned_module->user_forced[REDUCESCATTER].segsize,
                                                         session));
        return rc;
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCESCATTER]) {
        /* we do, so calc the message size or what ever we need and use
           this for the evaluation */
        int alg, faninout, segsize, ignoreme, i, count, size;
        size_t dsize, elemsize;
        size = ompi_comm_size(comm);
        for (i = 0, count = 0; i < size; i++) { count += ompi_count_array_get(rcounts, i);}
        ompi_datatype_type_size (dtype, &elemsize);
        dsize = elemsize * count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[REDUCESCATTER],
                                                        dsize, &faninout,
                                                        &segsize, &ignoreme);
        if (alg) {
            size_t _rbuf_dsize = elemsize * ompi_count_array_get(rcounts, ompi_comm_rank(comm));
            int rc;
            COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, dsize, _rbuf_dsize, dsize, rc,
                ompi_coll_tuned_reduce_scatter_intra_do_this(_sbuf, _rbuf, rcounts, dtype,
                                                             op, comm, module,
                                                             alg, faninout, segsize, session));
            return rc;
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_reduce_scatter_intra_dec_fixed (sbuf, rbuf, rcounts,
                                                           dtype, op, comm, module);
}

/*
 *    reduce_scatter_block_intra_dec
 *
 *    Function:   - seletects reduce_scatter_block algorithm to use
 *    Accepts:    - same arguments as MPI_Reduce_scatter_block()
 *    Returns:    - MPI_SUCCESS or error code (passed from
 *                  the reduce_scatter implementation)
 *
 */
int ompi_coll_tuned_reduce_scatter_block_intra_dec_dynamic(const void *sbuf, void *rbuf,
                                                           size_t rcount,
                                                           struct ompi_datatype_t *dtype,
                                                           struct ompi_op_t *op,
                                                           struct ompi_communicator_t *comm,
                                                           mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
        "coll:tuned:reduce_scatter_block_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCESCATTERBLOCK].algorithm) {
        size_t _elemsize, _rbuf_dsize, _sbuf_dsize;
        int rc;
        ompi_datatype_type_size(dtype, &_elemsize);
        _rbuf_dsize = _elemsize * rcount;
        _sbuf_dsize = _rbuf_dsize * (size_t)ompi_comm_size(comm);
        COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, _sbuf_dsize, _rbuf_dsize, _sbuf_dsize, rc,
            ompi_coll_tuned_reduce_scatter_block_intra_do_this(_sbuf, _rbuf, rcount, dtype,
                                                               op, comm, module,
                                                               tuned_module->user_forced[REDUCESCATTERBLOCK].algorithm,
                                                               tuned_module->user_forced[REDUCESCATTERBLOCK].chain_fanout,
                                                               tuned_module->user_forced[REDUCESCATTERBLOCK].segsize,
                                                               session));
        return rc;
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCESCATTERBLOCK]) {
        /* we do, so calc the message size or what ever we need and use
           this for the evaluation */
        int alg, faninout, segsize, ignoreme, size;
        size_t dsize, elemsize;
        size = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &elemsize);
        dsize = elemsize * rcount * size;

        alg = ompi_coll_tuned_get_target_method_params(tuned_module->com_rules[REDUCESCATTERBLOCK],
                                                       dsize, &faninout,
                                                       &segsize, &ignoreme);
        if (alg) {
            size_t _rbuf_dsize = elemsize * rcount;
            int rc;
            COLL_TUNED_GPU_DISPATCH_ASYM(op, dtype, sbuf, rbuf, dsize, _rbuf_dsize, dsize, rc,
                ompi_coll_tuned_reduce_scatter_block_intra_do_this(_sbuf, _rbuf, rcount, dtype,
                                                                   op, comm, module,
                                                                   alg, faninout, segsize, session));
            return rc;
        } /* found a method */
    } /* end if any com rules to check */

    return ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed (sbuf, rbuf, rcount,
                                                                 dtype, op, comm, module);
}

/*
 *    allgather_intra_dec
 *
 *    Function:    - seletects allgather algorithm to use
 *    Accepts:    - same arguments as MPI_Allgather()
 *    Returns:    - MPI_SUCCESS or error code (passed from the selected
 *                        allgather function).
 */

int ompi_coll_tuned_allgather_intra_dec_dynamic(const void *sbuf, size_t scount,
                                                struct ompi_datatype_t *sdtype,
                                                void* rbuf, size_t rcount,
                                                struct ompi_datatype_t *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_allgather_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLGATHER].algorithm) {
        mca_allocator_base_module_t *allocator = NULL;
        int _dev_id = MCA_ACCELERATOR_NO_DEVICE_ID;
        uint64_t _flags;
        if ((sbuf != MPI_IN_PLACE &&
             opal_accelerator.check_addr(sbuf, &_dev_id, &_flags) > 0) ||
            opal_accelerator.check_addr(rbuf, &_dev_id, &_flags) > 0) {
            allocator = opal_accelerator_base_get_device_allocator(_dev_id);
        }
        return ompi_coll_tuned_allgather_intra_do_this(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLGATHER].algorithm,
                                                       tuned_module->user_forced[ALLGATHER].tree_fanout,
                                                       tuned_module->user_forced[ALLGATHER].segsize,
                                                       allocator);
    }

    if (tuned_module->com_rules[ALLGATHER]) {
        /* We have file based rules:
           - calculate message size and other necessary information */
        int comsize;
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (sdtype, &dsize);
        comsize = ompi_comm_size(comm);
        dsize *= (ptrdiff_t)comsize * (ptrdiff_t)scount;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLGATHER],
                                                        dsize, &faninout, &segsize, &ignoreme);
        if (alg) {
            mca_allocator_base_module_t *allocator = NULL;
            int _dev_id = MCA_ACCELERATOR_NO_DEVICE_ID;
            uint64_t _flags;
            if ((sbuf != MPI_IN_PLACE &&
                 opal_accelerator.check_addr(sbuf, &_dev_id, &_flags) > 0) ||
                opal_accelerator.check_addr(rbuf, &_dev_id, &_flags) > 0) {
                allocator = opal_accelerator_base_get_device_allocator(_dev_id);
            }
            return ompi_coll_tuned_allgather_intra_do_this(sbuf, scount, sdtype,
                                                           rbuf, rcount, rdtype,
                                                           comm, module,
                                                           alg, faninout, segsize, allocator);
        }
    }

    /* Use default decision */
    return ompi_coll_tuned_allgather_intra_dec_fixed (sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module);
}

/*
 *    allgatherv_intra_dec
 *
 *    Function:    - seletects allgatherv algorithm to use
 *    Accepts:    - same arguments as MPI_Allgatherv()
 *    Returns:    - MPI_SUCCESS or error code (passed from the selected
 *                        allgatherv function).
 */

int ompi_coll_tuned_allgatherv_intra_dec_dynamic(const void *sbuf, size_t scount,
                                                 struct ompi_datatype_t *sdtype,
                                                 void* rbuf, ompi_count_array_t rcounts,
                                                 ompi_disp_array_t rdispls,
                                                 struct ompi_datatype_t *rdtype,
                                                 struct ompi_communicator_t *comm,
                                                 mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_allgatherv_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLGATHERV].algorithm) {
        /* User-forced algorithm */
        return ompi_coll_tuned_allgatherv_intra_do_this(sbuf, scount, sdtype,
                                                        rbuf, rcounts, rdispls, rdtype,
                                                        comm, module,
                                                        tuned_module->user_forced[ALLGATHERV].algorithm,
                                                        tuned_module->user_forced[ALLGATHERV].tree_fanout,
                                                        tuned_module->user_forced[ALLGATHERV].segsize);
    }

    if (tuned_module->com_rules[ALLGATHERV]) {
        /* We have file based rules:
           - calculate message size and other necessary information */
        int comsize, i;
        int alg, faninout, segsize, ignoreme;
        size_t dsize, total_size, per_rank_size;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        total_size = 0;
        for (i = 0; i < comsize; i++) { total_size += dsize * ompi_count_array_get(rcounts, i); }

        per_rank_size = total_size / comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLGATHERV],
                                                        per_rank_size, &faninout, &segsize, &ignoreme);
        if (alg) {
            /* we have found a valid choice from the file based rules for
               this message size */
            return ompi_coll_tuned_allgatherv_intra_do_this (sbuf, scount, sdtype,
                                                             rbuf, rcounts,
                                                             rdispls, rdtype,
                                                             comm, module,
                                                             alg, faninout, segsize);
        }
    }
    /* Use default decision */
    return ompi_coll_tuned_allgatherv_intra_dec_fixed (sbuf, scount, sdtype,
                                                       rbuf, rcounts,
                                                       rdispls, rdtype,
                                                       comm, module);
}

int ompi_coll_tuned_gather_intra_dec_dynamic(const void *sbuf, size_t scount,
                                             struct ompi_datatype_t *sdtype,
                                             void* rbuf, size_t rcount,
                                             struct ompi_datatype_t *rdtype,
                                             int root,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_gather_intra_dec_dynamic"));

    mca_allocator_base_module_t *allocator = NULL;

    /* Scratch buffer is used for data movement only (no ompi_op_reduce).
     * Use device allocator when user buffers are on device. */
    {
        int _dev_id = MCA_ACCELERATOR_NO_DEVICE_ID;
        uint64_t _flags;
        if ((sbuf != MPI_IN_PLACE &&
             opal_accelerator.check_addr(sbuf, &_dev_id, &_flags) > 0) ||
            opal_accelerator.check_addr(rbuf, &_dev_id, &_flags) > 0) {
            allocator = opal_accelerator_base_get_device_allocator(_dev_id);
        }
    }

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[GATHER].algorithm) {
        return ompi_coll_tuned_gather_intra_do_this(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module,
                                                    tuned_module->user_forced[GATHER].algorithm,
                                                    tuned_module->user_forced[GATHER].tree_fanout,
                                                    tuned_module->user_forced[GATHER].segsize,
                                                    allocator);
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[GATHER]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        dsize *= scount * comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[GATHER],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            return ompi_coll_tuned_gather_intra_do_this(sbuf, scount, sdtype,
                                                        rbuf, rcount, rdtype,
                                                        root, comm, module,
                                                        alg, faninout, segsize, allocator);
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_gather_intra_dec_fixed (sbuf, scount, sdtype,
                                                   rbuf, rcount, rdtype,
                                                   root, comm, module);
}

int ompi_coll_tuned_scatter_intra_dec_dynamic(const void *sbuf, size_t scount,
                                              struct ompi_datatype_t *sdtype,
                                              void* rbuf, size_t rcount,
                                              struct ompi_datatype_t *rdtype,
                                              int root, struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_scatter_intra_dec_dynamic"));

    mca_allocator_base_module_t *allocator = NULL;

    /* Scratch buffer is used for data movement only (no ompi_op_reduce).
     * Use device allocator when user buffers are on device. */
    {
        int _dev_id = MCA_ACCELERATOR_NO_DEVICE_ID;
        uint64_t _flags;
        if ((sbuf != MPI_IN_PLACE &&
             opal_accelerator.check_addr(sbuf, &_dev_id, &_flags) > 0) ||
            opal_accelerator.check_addr(rbuf, &_dev_id, &_flags) > 0) {
            allocator = opal_accelerator_base_get_device_allocator(_dev_id);
        }
    }

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[SCATTER].algorithm) {
        return ompi_coll_tuned_scatter_intra_do_this(sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     root, comm, module,
                                                     tuned_module->user_forced[SCATTER].algorithm,
                                                     tuned_module->user_forced[SCATTER].chain_fanout,
                                                     tuned_module->user_forced[SCATTER].segsize,
                                                     allocator);
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[SCATTER]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        dsize *= scount * comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[SCATTER],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            return ompi_coll_tuned_scatter_intra_do_this(sbuf, scount, sdtype,
                                                         rbuf, rcount, rdtype,
                                                         root, comm, module,
                                                         alg, faninout, segsize, allocator);
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_tuned_scatter_intra_dec_fixed (sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module);
}

int ompi_coll_tuned_exscan_intra_dec_dynamic(const void *sbuf, void* rbuf, size_t count,
                                              struct ompi_datatype_t *dtype,
                                              struct ompi_op_t *op,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_exscan_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[EXSCAN].algorithm) {
        size_t _bufsize;
        int rc;
        ompi_datatype_type_size(dtype, &_bufsize);
        _bufsize *= count;
        COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _bufsize, rc,
            ompi_coll_tuned_exscan_intra_do_this(_sbuf, _rbuf, count, dtype,
                                                 op, comm, module,
                                                 tuned_module->user_forced[EXSCAN].algorithm,
                                                 session));
        return rc;
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[EXSCAN]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[EXSCAN],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            size_t _bufsize;
            int rc;
            ompi_datatype_type_size(dtype, &_bufsize);
            _bufsize *= count;
            COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _bufsize, rc,
                ompi_coll_tuned_exscan_intra_do_this(_sbuf, _rbuf, count, dtype,
                                                     op, comm, module,
                                                     alg, session));
            return rc;
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_base_exscan_intra_linear(sbuf, rbuf, count, dtype,
                                              op, comm, module, NULL);
}

int ompi_coll_tuned_scan_intra_dec_dynamic(const void *sbuf, void* rbuf, size_t count,
                                           struct ompi_datatype_t *dtype,
                                           struct ompi_op_t *op,
                                           struct ompi_communicator_t *comm,
                                           mca_coll_base_module_t *module)
{
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT_VERBOSE((COLL_TUNED_TRACING_VERBOSE, ompi_coll_tuned_stream,
                 "ompi_coll_tuned_scan_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[SCAN].algorithm) {
        size_t _bufsize;
        int rc;
        ompi_datatype_type_size(dtype, &_bufsize);
        _bufsize *= count;
        COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _bufsize, rc,
            ompi_coll_tuned_scan_intra_do_this(_sbuf, _rbuf, count, dtype,
                                               op, comm, module,
                                               tuned_module->user_forced[SCAN].algorithm,
                                               session));
        return rc;
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[SCAN]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[SCAN],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            size_t _bufsize;
            int rc;
            ompi_datatype_type_size(dtype, &_bufsize);
            _bufsize *= count;
            COLL_TUNED_GPU_DISPATCH(op, dtype, sbuf, rbuf, _bufsize, rc,
                ompi_coll_tuned_scan_intra_do_this(_sbuf, _rbuf, count, dtype,
                                                   op, comm, module,
                                                   alg, session));
            return rc;
        } /* found a method */
    } /*end if any com rules to check */

    return ompi_coll_base_scan_intra_linear(sbuf, rbuf, count, dtype,
                                            op, comm, module, NULL);
}
