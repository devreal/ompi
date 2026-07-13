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
# Build the CUDA persistent-kernel op component when the CUDA runtime
# toolkit (cuda_runtime.h, libcudart, nvcc) is available.
#
# Deliberately does NOT require CUDA_SUPPORT=1 (which gates on libcuda.so,
# the GPU driver API library).  The op/cuda component only uses the runtime
# API and can therefore be compiled in build environments that have the CUDA
# toolkit installed but no GPU driver (e.g., CI containers, cross-build nodes).
#
# Requires --with-cuda[=DIR] to locate the toolkit.
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

    # Ensure with_cuda is defined (OPAL_CHECK_CUDA parses --with-cuda).
    AC_REQUIRE([OPAL_CHECK_CUDA])

    OPAL_VAR_SCOPE_PUSH([op_cuda_save_CPPFLAGS op_cuda_save_LDFLAGS op_cuda_save_LIBS op_cuda_libdir op_cuda_nvcc_path op_cuda_incdir])

    op_cuda_happy=no
    op_cuda_incdir=""

    # Only attempt a build when the user asked for CUDA (--with-cuda[=DIR]).
    AS_IF([test "x$with_cuda" != "x" && test "$with_cuda" != "no"],
      [
        # Derive the include directory from $with_cuda, mirroring OPAL_CHECK_CUDA.
        AS_IF([test -f "${with_cuda}/include/cuda_runtime.h"],
              [op_cuda_incdir="${with_cuda}/include"],
              [AS_IF([test -f "${with_cuda}/cuda_runtime.h"],
                     [op_cuda_incdir="${with_cuda}"],
                     [AS_IF([test -f "/usr/local/cuda/include/cuda_runtime.h"],
                            [op_cuda_incdir="/usr/local/cuda/include"])])])

        op_cuda_save_CPPFLAGS="$CPPFLAGS"
        op_cuda_save_LDFLAGS="$LDFLAGS"
        op_cuda_save_LIBS="$LIBS"

        AS_IF([test -n "$op_cuda_incdir"],
              [CPPFLAGS="-I$op_cuda_incdir $CPPFLAGS"])

        # Verify the runtime header is present.
        AC_CHECK_HEADER([cuda_runtime.h],
          [op_cuda_happy=yes],
          [AC_MSG_WARN([cuda_runtime.h not found; skipping op/cuda component])
           op_cuda_happy=no])

        # Locate libcudart — prefer lib64, fall back to lib, then /usr/local/cuda.
        AS_IF([test "$op_cuda_happy" = "yes"],
          [op_cuda_libdir=""
           AS_IF([test "$with_cuda" != "yes"],
                 [AS_IF([test -d "$with_cuda/lib64"],
                        [op_cuda_libdir="$with_cuda/lib64"],
                        [AS_IF([test -d "$with_cuda/lib"],
                               [op_cuda_libdir="$with_cuda/lib"])])])
           AS_IF([test -z "$op_cuda_libdir"],
                 [AS_IF([test -d "/usr/local/cuda/lib64"],
                        [op_cuda_libdir="/usr/local/cuda/lib64"],
                        [AS_IF([test -d "/usr/local/cuda/lib"],
                               [op_cuda_libdir="/usr/local/cuda/lib"])])])
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
           AS_IF([test "$with_cuda" != "yes" && test -d "$with_cuda/bin"],
                 [op_cuda_nvcc_path="$with_cuda/bin:$PATH"])
           AC_PATH_PROG([NVCC], [nvcc], [not_found], [$op_cuda_nvcc_path])
           AS_IF([test "$NVCC" = "not_found"],
                 [AC_MSG_WARN([nvcc not found; skipping op/cuda component])
                  op_cuda_happy=no])
          ])

        # Populate the output variables.
        AS_IF([test "$op_cuda_happy" = "yes"],
          [op_cuda_CPPFLAGS="-I$op_cuda_incdir"
           AS_IF([test -n "$op_cuda_libdir"],
                 [op_cuda_LDFLAGS="-L$op_cuda_libdir"],
                 [op_cuda_LDFLAGS=""])
           op_cuda_LIBS="-lcudart"
           # __nanosleep requires SM 7.0 (Volta) or later.
           AS_IF([test "x$NVCCFLAGS" = "x"],
                 [NVCCFLAGS="-arch=native"])
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
