#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates.  All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# C sources (compiled by the normal C compiler)
sources = \
    op_cuda.h \
    op_cuda_component.c \
    op_cuda_session.c

# The .cu file is compiled separately by nvcc and linked in as a plain
# object.  Libtool does not know how to compile CUDA, so we use a custom
# rule.  The resulting object is appended to LIBADD for both DSO and static
# builds.

EXTRA_DIST  = op_cuda_kernels.cu
CLEANFILES  = op_cuda_kernels.o

# Include paths forwarded to nvcc so it can find ompi_config.h and the
# op/mca headers.
NVCC_INCLUDES = \
    -I$(top_srcdir) \
    -I$(top_builddir) \
    -I$(top_srcdir)/ompi \
    -I$(top_builddir)/ompi

op_cuda_kernels.o: $(srcdir)/op_cuda_kernels.cu \
                   $(srcdir)/op_cuda.h
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) \
	    $(op_cuda_CPPFLAGS) \
	    --compiler-options "$(DEFS)" \
	    -c $< -o $@

AM_CPPFLAGS = $(op_cuda_CPPFLAGS)

# ----------------------------------------------------------------------------
# DSO build
# ----------------------------------------------------------------------------
if MCA_BUILD_ompi_op_cuda_DSO
component_install = mca_op_cuda.la
component_noinst  =
else
component_install =
component_noinst  = libmca_op_cuda.la
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)

mca_op_cuda_la_SOURCES  = $(sources)
mca_op_cuda_la_LDFLAGS  = -module -avoid-version $(op_cuda_LDFLAGS)
mca_op_cuda_la_LIBADD   = $(op_cuda_LIBS) op_cuda_kernels.o
mca_op_cuda_la_CPPFLAGS = $(op_cuda_CPPFLAGS)

# ----------------------------------------------------------------------------
# Static (convenience library) build
# ----------------------------------------------------------------------------
noinst_LTLIBRARIES = $(component_noinst)

libmca_op_cuda_la_SOURCES  = $(sources)
libmca_op_cuda_la_LDFLAGS  = -module -avoid-version $(op_cuda_LDFLAGS)
libmca_op_cuda_la_LIBADD   = $(op_cuda_LIBS) op_cuda_kernels.o
libmca_op_cuda_la_CPPFLAGS = $(op_cuda_CPPFLAGS)
