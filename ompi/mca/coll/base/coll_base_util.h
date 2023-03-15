/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2007 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2014-2020 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BASE_UTIL_EXPORT_H
#define MCA_COLL_BASE_UTIL_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/request/request.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/op/op.h"
#include "ompi/mca/pml/pml.h"
#include "opal/mca/accelerator/accelerator.h"

BEGIN_C_DECLS

/**
 * The largest array we need to track collective temporary memory. Right now
 * the record is for ialltoallw, for the array of send and receive types,
 * count and displacements.
 */
#define OMPI_REQ_NB_RELEASE_ARRAYS 7

/**
 * Request structure to be returned by non-blocking
 * collective operations.
 */
struct ompi_coll_base_nbc_request_t {
    ompi_request_t super;
    union {
        ompi_request_complete_fn_t req_complete_cb;
        ompi_request_free_fn_t req_free;
    } cb;
    void *req_complete_cb_data;
    struct {
        union {
            struct {
                ompi_op_t *op;
                ompi_datatype_t *datatype;
            } op;
            struct {
                ompi_datatype_t *stype;
                ompi_datatype_t *rtype;
            } types;
            struct {
                opal_object_t *objs[2];
            } objs;
            struct {
                ompi_datatype_t * const *stypes;
                ompi_datatype_t * const *rtypes;
                int scount;
                int rcount;
            } vecs;
        } refcounted;
        void* release_arrays[OMPI_REQ_NB_RELEASE_ARRAYS];
    } data;
};

OMPI_DECLSPEC OBJ_CLASS_DECLARATION(ompi_coll_base_nbc_request_t);

static inline int32_t
ompi_coll_base_nbc_reserve_tags(ompi_communicator_t* comm, int32_t reserve)
{
    int32_t tag, old_tag;
    assert( reserve > 0 );
  reread_tag:  /* In case we fail to atomically update the tag */
    tag = old_tag = comm->c_nbc_tag;
    if ((tag - reserve) < MCA_COLL_BASE_TAG_NONBLOCKING_END) {
        tag = MCA_COLL_BASE_TAG_NONBLOCKING_BASE;
    }
    if( !OPAL_ATOMIC_COMPARE_EXCHANGE_STRONG_32(&comm->c_nbc_tag, &old_tag, tag - reserve) ) {
        goto reread_tag;
    }
    return tag;
}

typedef struct ompi_coll_base_nbc_request_t ompi_coll_base_nbc_request_t;

/*
 * Structure to store an available module
 */
struct mca_coll_base_avail_coll_t {
    opal_list_item_t super;

    int ac_priority;
    mca_coll_base_module_t *ac_module;
    const char * ac_component_name;
};
typedef struct mca_coll_base_avail_coll_t mca_coll_base_avail_coll_t;
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(mca_coll_base_avail_coll_t);

/**
 * A MPI_like function doing a send and a receive simultaneously.
 * If one of the communications results in a zero-byte message the
 * communication is ignored, and no message will cross to the peer.
 */
int ompi_coll_base_sendrecv_actual( const void* sendbuf, size_t scount,
                                    ompi_datatype_t* sdatatype,
                                    int dest, int stag,
                                    void* recvbuf, size_t rcount,
                                    ompi_datatype_t* rdatatype,
                                    int source, int rtag,
                                    struct ompi_communicator_t* comm,
                                    ompi_status_public_t* status );


/**
 * Similar to the function above this implementation of send-receive
 * do not generate communications for zero-bytes messages. Thus, it is
 * improper to use in the context of some algorithms for collective
 * communications.
 */
static inline int
ompi_coll_base_sendrecv( void* sendbuf, size_t scount, ompi_datatype_t* sdatatype,
                          int dest, int stag,
                          void* recvbuf, size_t rcount, ompi_datatype_t* rdatatype,
                          int source, int rtag,
                          struct ompi_communicator_t* comm,
                          ompi_status_public_t* status, int myid )
{
    if ((dest == source) && (source == myid)) {
        return (int) ompi_datatype_sndrcv(sendbuf, (int32_t) scount, sdatatype,
                                          recvbuf, (int32_t) rcount, rdatatype);
    }
    return ompi_coll_base_sendrecv_actual (sendbuf, scount, sdatatype,
                                           dest, stag,
                                           recvbuf, rcount, rdatatype,
                                           source, rtag, comm, status);
}

/**
 * ompi_mirror_perm: Returns mirror permutation of nbits low-order bits
 *                   of x [*].
 * [*] Warren Jr., Henry S. Hacker's Delight (2ed). 2013.
 *     Chapter 7. Rearranging Bits and Bytes.
 */
unsigned int ompi_mirror_perm(unsigned int x, int nbits);

/*
 * ompi_rounddown: Rounds a number down to nearest multiple.
 *     rounddown(10,4) = 8, rounddown(6,3) = 6, rounddown(14,3) = 12
 */
int ompi_rounddown(int num, int factor);

/**
 * If necessary, retain op and store it in the
 * request object, which should be of type ompi_coll_base_nbc_request_t
 * (will be cast internally).
 */
int ompi_coll_base_retain_op( ompi_request_t *request,
                              ompi_op_t *op,
                              ompi_datatype_t *type);

/**
 * If necessary, retain the datatypes and store them in the
 * request object, which should be of type ompi_coll_base_nbc_request_t
 * (will be cast internally).
 */
int ompi_coll_base_retain_datatypes( ompi_request_t *request,
                                     ompi_datatype_t *stype,
                                     ompi_datatype_t *rtype);

/**
 * If necessary, retain the datatypes and store them in the
 * request object, which should be of type ompi_coll_base_nbc_request_t
 * (will be cast internally).
 */
int ompi_coll_base_retain_datatypes_w( ompi_request_t *request,
                                       ompi_datatype_t * const stypes[],
                                       ompi_datatype_t * const rtypes[],
                                       bool use_topo);

/* File reading function */
int ompi_coll_base_file_getnext_long(FILE *fptr, int *fileline, long* val);
int ompi_coll_base_file_getnext_size_t(FILE *fptr, int *fileline, size_t* val);
int ompi_coll_base_file_getnext_string(FILE *fptr, int *fileline, char** val);
/* peek at the next valid token to see if it begins with the expected value. If yes
 * eat the value, otherwise put it back into the file.
 */
int ompi_coll_base_file_peek_next_char_is(FILE *fptr, int *fileline, int expected);

/* Miscellaneous function */
const char* mca_coll_base_colltype_to_str(int collid);
int mca_coll_base_name_to_colltype(const char* name);

/* device/host memory allocation functions */

/**
 * Returns a pointer to memory in the same memory domain as the receive or send buffer.
 * Device memory is allocated if either the receive buffer or the send buffer are
 * located on the device and if the op supports on-device reductions on the datatype.
 * If memory is allocated on the host, device will be set to -1.
 */
static inline
void* ompi_coll_base_allocate_op_tmpbuf(
    const void *sendbuf, const void *recvbuf, size_t size,
    const struct ompi_op_t *op, const struct ompi_datatype_t *dtype,
    int *device)
{
    void *res = NULL;
    uint64_t flags;
    *device = -1;
    if ((NULL == op && NULL == dtype) || ompi_op_supports_device(op, dtype)) {
        /* if the recvbuf is on the device we take that device */
        if (NULL != recvbuf && 0 < opal_accelerator.check_addr(recvbuf, device, &flags)) {
            if (OPAL_SUCCESS != opal_accelerator.mem_alloc(*device, &res, size)) {
                /* fall back to the host */
                res = NULL;
                *device = -1;
            }
        } else if (MPI_IN_PLACE != sendbuf && NULL != sendbuf &&
                0 < opal_accelerator.check_addr(sendbuf, device, &flags)) {
            /* send buffer is on a device so try to allocate memory there */
            if (OPAL_SUCCESS != opal_accelerator.mem_alloc(*device, &res, size)) {
                /* fall back to the host */
                res = NULL;
                *device = -1;
            }
        }
    }

    if (NULL == res) {
        res = malloc(size);
    }
    return res;
}

/**
 * Like ompi_coll_base_allocate_op_tmpbuf but without checking op-datatype
 * device compatibility.
 */
static inline
void* ompi_coll_base_allocate_tmpbuf(
    const void *sendbuf, const void *recvbuf,
    size_t size, int *device)
{
    return ompi_coll_base_allocate_op_tmpbuf(sendbuf, recvbuf, size, NULL, NULL, device);
}

/**
 * Frees memory allocated through ompi_coll_base_allocate_op_tmpbuf
 * or ompi_coll_base_allocate_tmpbuf.
 */
static inline
void ompi_coll_base_free_tmpbuf(void *tmpbuf, int device) {
    if (-1 == device) {
        free(tmpbuf);
    } else if (NULL != tmpbuf) {
        opal_accelerator.mem_release(device, tmpbuf);
    }
}


END_C_DECLS
#endif /* MCA_COLL_BASE_UTIL_EXPORT_H */
