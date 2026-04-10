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

# MCA_ompi_op_rocm_CONFIG([action-if-can-compile],
#                          [action-if-cant-compile])
# ------------------------------------------------
# Build the ROCm persistent-kernel op component only when the HIP runtime
# (libamdhip64 + hip/hip_runtime.h) and hipcc are available.
#
# Calls OPAL_CHECK_ROCM to locate headers and libraries, then separately
# finds hipcc.  Sets:
#   op_rocm_CPPFLAGS — include/define flags for HIP (includes -D__HIP_PLATFORM_AMD__)
#   op_rocm_LDFLAGS  — library search path for libamdhip64
#   op_rocm_LIBS     — -lamdhip64
#   HIPCC            — path to the hipcc compiler
#   HIPCCFLAGS       — default hipcc flags
#
AC_DEFUN([MCA_ompi_op_rocm_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/op/rocm/Makefile])

    OPAL_VAR_SCOPE_PUSH([op_rocm_happy op_rocm_hipcc_path])

    op_rocm_happy=no

    # OPAL_CHECK_ROCM calls OAC_CHECK_PACKAGE and sets:
    #   op_rocm_CPPFLAGS, op_rocm_LDFLAGS, op_rocm_LIBS
    # It also sets ROCM_SUPPORT=1 on success.
    OPAL_CHECK_ROCM([op_rocm],
        [op_rocm_happy=yes],
        [op_rocm_happy=no])

    # Find hipcc alongside the ROCm installation.
    AS_IF([test "$op_rocm_happy" = "yes"],
      [op_rocm_hipcc_path="$PATH"
       AS_IF([test -n "$with_rocm" && test "$with_rocm" != "no" && test -d "$with_rocm/bin"],
             [op_rocm_hipcc_path="$with_rocm/bin:$PATH"],
             [AS_IF([test -d "/opt/rocm/bin"],
                    [op_rocm_hipcc_path="/opt/rocm/bin:$PATH"])])
       AC_PATH_PROG([HIPCC], [hipcc], [not_found], [$op_rocm_hipcc_path])
       AS_IF([test "$HIPCC" = "not_found"],
             [AC_MSG_WARN([hipcc not found; skipping op/rocm component])
              op_rocm_happy=no])
      ])

    # Default HIPCCFLAGS if not already set by the user.
    AS_IF([test "$op_rocm_happy" = "yes" && test "x$HIPCCFLAGS" = "x"],
          [HIPCCFLAGS="--offload-arch=gfx906"])

    AC_SUBST([op_rocm_CPPFLAGS])
    AC_SUBST([op_rocm_LDFLAGS])
    AC_SUBST([op_rocm_LIBS])
    AC_SUBST([HIPCC])
    AC_SUBST([HIPCCFLAGS])

    OPAL_VAR_SCOPE_POP

    AS_IF([test "$op_rocm_happy" = "yes"],
          [$1],
          [$2])
])dnl
