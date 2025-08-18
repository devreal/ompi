/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2018      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if !defined(OPAL_STDATOMIC_H)
#    define OPAL_STDATOMIC_H

#    include "opal_stdint.h"

#if OPAL_USE_C11_ATOMICS == 0

typedef volatile int opal_atomic_int_t;
typedef volatile long opal_atomic_long_t;

typedef volatile int32_t opal_atomic_int32_t;
typedef volatile uint32_t opal_atomic_uint32_t;
typedef volatile int64_t opal_atomic_int64_t;
typedef volatile uint64_t opal_atomic_uint64_t;

typedef volatile size_t opal_atomic_size_t;
typedef volatile ssize_t opal_atomic_ssize_t;
typedef volatile intptr_t opal_atomic_intptr_t;
typedef volatile uintptr_t opal_atomic_uintptr_t;

typedef opal_atomic_int32_t opal_atomic_lock_t;

enum { OPAL_ATOMIC_LOCK_UNLOCKED = 0,
       OPAL_ATOMIC_LOCK_LOCKED = 1 };

#    define OPAL_ATOMIC_LOCK_INIT OPAL_ATOMIC_LOCK_UNLOCKED

#    define OPAL_ATOMIC_INIT(v) (v)

typedef enum {
    OPAL_ATOMIC_ORDER_RELAXED,
    OPAL_ATOMIC_ORDER_ACQUIRE,
    OPAL_ATOMIC_ORDER_RELEASE,
    OPAL_ATOMIC_ORDER_ACQ_REL,
    OPAL_ATOMIC_ORDER_SEQ_CST
} opal_memory_order_t;

#else /* OPAL_USE_C11_ATOMICS == 0 */

#    include <stdatomic.h>

typedef struct { _Atomic int v; } opal_atomic_int_t;
typedef struct { _Atomic long v; } opal_atomic_long_t;

typedef struct { _Atomic int32_t v; } opal_atomic_int32_t;
typedef struct { _Atomic uint32_t v; } opal_atomic_uint32_t;
typedef struct { _Atomic int64_t v; } opal_atomic_int64_t;
typedef struct { _Atomic uint64_t v; } opal_atomic_uint64_t;

typedef struct { _Atomic size_t v; } opal_atomic_size_t;
typedef struct { _Atomic ssize_t v; } opal_atomic_ssize_t;
typedef struct { _Atomic intptr_t v; } opal_atomic_intptr_t;
typedef struct { _Atomic uintptr_t v; } opal_atomic_uintptr_t;

typedef atomic_flag opal_atomic_lock_t;

#    define OPAL_ATOMIC_LOCK_UNLOCKED false
#    define OPAL_ATOMIC_LOCK_LOCKED   true

#    define OPAL_ATOMIC_LOCK_INIT ATOMIC_FLAG_INIT

#    define OPAL_ATOMIC_INIT(v) { .v = (v) }

typedef memory_order opal_memory_order_t;
#    define OPAL_ATOMIC_ORDER_RELAXED memory_order_relaxed
#    define OPAL_ATOMIC_ORDER_ACQUIRE memory_order_acquire
#    define OPAL_ATOMIC_ORDER_RELEASE memory_order_release
#    define OPAL_ATOMIC_ORDER_ACQ_REL memory_order_acq_rel
#    define OPAL_ATOMIC_ORDER_SEQ_CST memory_order_seq_cst

#    endif /* OPAL_USE_C11_ATOMICS == 0 */

#    if HAVE_OPAL_INT128_T

#        if OPAL_USE_C11_ATOMICS && OPAL_HAVE_C11_CSWAP_INT128

typedef struct { _Atomic opal_int128_t v; } opal_atomic_int128_t;

#        else

typedef struct {
    volatile opal_int128_t v;
} opal_atomic_int128_t __opal_attribute_aligned__(16);

#        endif

#    endif

#endif /* !defined(OPAL_STDATOMIC_H) */
