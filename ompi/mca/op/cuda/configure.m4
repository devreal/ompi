# -*- shell-script -*-
#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates.  All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# MCA_ompi_op_cuda_CONFIG([action-if-can-compile],
#                          [action-if-cant-compile])
# ------------------------------------------------
# Build the CUDA persistent-kernel op component only when the CUDA
# runtime (libcudart + cuda_runtime.h) and nvcc are available.
#
# Requires that OPAL_CHECK_CUDA has already been called (which sets
# $CUDA_SUPPORT, $opal_cuda_incdir, and $with_cuda).
#
# Sets:
#   op_cuda_CPPFLAGS — include path for cuda_runtime.h
#   op_cuda_LDFLAGS  — library search path for libcudart
#   op_cuda_LIBS     — -lcudart
#   NVCC             — path to the nvcc compiler
#   NVCCFLAGS        — default nvcc flags (min arch SM 7.0 for __nanosleep)
#
AC_DEFUN([MCA_ompi_op_cuda_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/op/cuda/Makefile])

    # Ensure the top-level CUDA driver-API check has been performed.
    AC_REQUIRE([OPAL_CHECK_CUDA])

    OPAL_VAR_SCOPE_PUSH([op_cuda_happy op_cuda_save_CPPFLAGS op_cuda_save_LDFLAGS op_cuda_save_LIBS op_cuda_libdir op_cuda_nvcc_path])

    op_cuda_happy=no

    AS_IF([test "x$CUDA_SUPPORT" = "x1"],
      [
        op_cuda_save_CPPFLAGS="$CPPFLAGS"
        op_cuda_save_LDFLAGS="$LDFLAGS"
        op_cuda_save_LIBS="$LIBS"

        CPPFLAGS="-I$opal_cuda_incdir $CPPFLAGS"

        # Verify that the runtime header is present alongside cuda.h.
        AC_CHECK_HEADER([cuda_runtime.h],
          [op_cuda_happy=yes],
          [AC_MSG_WARN([cuda_runtime.h not found; skipping op/cuda component])
           op_cuda_happy=no])

        # Locate libcudart — prefer lib64, fall back to lib.
        AS_IF([test "$op_cuda_happy" = "yes"],
          [op_cuda_libdir=""
           AS_IF([test -d "$with_cuda/lib64"],
                 [op_cuda_libdir="$with_cuda/lib64"],
                 [AS_IF([test -d "$with_cuda/lib"],
                        [op_cuda_libdir="$with_cuda/lib"],
                        [AS_IF([test -d "/usr/local/cuda/lib64"],
                               [op_cuda_libdir="/usr/local/cuda/lib64"])])])
           AS_IF([test -n "$op_cuda_libdir"],
                 [LDFLAGS="-L$op_cuda_libdir $LDFLAGS"])
           AC_CHECK_LIB([cudart], [cudaGetDeviceCount],
             [op_cuda_happy=yes],
             [AC_MSG_WARN([libcudart not found; skipping op/cuda component])
              op_cuda_happy=no])
          ])

        # Locate nvcc.
        AS_IF([test "$op_cuda_happy" = "yes"],
          [op_cuda_nvcc_path="$PATH"
           AS_IF([test -d "$with_cuda/bin"],
                 [op_cuda_nvcc_path="$with_cuda/bin:$PATH"])
           AC_PATH_PROG([NVCC], [nvcc], [not_found], [$op_cuda_nvcc_path])
           AS_IF([test "$NVCC" = "not_found"],
                 [AC_MSG_WARN([nvcc not found; skipping op/cuda component])
                  op_cuda_happy=no])
          ])

        # Populate the output variables.
        AS_IF([test "$op_cuda_happy" = "yes"],
          [op_cuda_CPPFLAGS="-I$opal_cuda_incdir"
           AS_IF([test -n "$op_cuda_libdir"],
                 [op_cuda_LDFLAGS="-L$op_cuda_libdir"],
                 [op_cuda_LDFLAGS=""])
           op_cuda_LIBS="-lcudart"
           # __nanosleep requires SM 7.0 (Volta) or later.
           AS_IF([test "x$NVCCFLAGS" = "x"],
                 [NVCCFLAGS="-arch=sm_70"])
          ])

        CPPFLAGS="$op_cuda_save_CPPFLAGS"
        LDFLAGS="$op_cuda_save_LDFLAGS"
        LIBS="$op_cuda_save_LIBS"
      ])

    AC_SUBST([op_cuda_CPPFLAGS])
    AC_SUBST([op_cuda_LDFLAGS])
    AC_SUBST([op_cuda_LIBS])
    AC_SUBST([NVCC])
    AC_SUBST([NVCCFLAGS])

    OPAL_VAR_SCOPE_POP

    AS_IF([test "$op_cuda_happy" = "yes"],
          [$1],
          [$2])
])dnl
