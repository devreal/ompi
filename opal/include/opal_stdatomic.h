/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2018      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if !defined(OPAL_STDATOMIC_H)
#    define OPAL_STDATOMIC_H

#include "opal_stdint.h"

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

#if HAVE_OPAL_INT128_T

typedef volatile opal_int128_t opal_atomic_int128_t;

#endif

#if OPAL_USE_C11_ATOMICS == 0
typedef opal_atomic_int32_t opal_atomic_lock_t;

enum { OPAL_ATOMIC_LOCK_UNLOCKED = 0,
       OPAL_ATOMIC_LOCK_LOCKED = 1 };

#define OPAL_ATOMIC_LOCK_INIT OPAL_ATOMIC_LOCK_UNLOCKED

#else /* OPAL_USE_C11_ATOMICS == 0 */

#include <stdatomic.h>

typedef atomic_flag opal_atomic_lock_t;

#define OPAL_ATOMIC_LOCK_UNLOCKED false
#define OPAL_ATOMIC_LOCK_LOCKED   true

#define OPAL_ATOMIC_LOCK_INIT ATOMIC_FLAG_INIT

#endif /* OPAL_USE_C11_ATOMICS == 0 */


#endif /* !defined(OPAL_STDATOMIC_H) */
