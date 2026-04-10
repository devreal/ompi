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
    op_rocm.h \
    op_rocm_component.c \
    op_rocm_session.c

# The .cpp file is compiled separately by hipcc and linked in as a plain
# object.  Libtool does not know how to compile HIP C++, so we use a custom
# rule.  The resulting object is appended to LIBADD for both DSO and static
# builds.

EXTRA_DIST  = op_rocm_kernels.cpp
CLEANFILES  = op_rocm_kernels.o

# Include paths forwarded to hipcc so it can find ompi_config.h and the
# op/mca headers.
HIPCC_INCLUDES = \
    -I$(top_srcdir) \
    -I$(top_builddir) \
    -I$(top_srcdir)/ompi \
    -I$(top_builddir)/ompi

op_rocm_kernels.o: $(srcdir)/op_rocm_kernels.cpp \
                   $(srcdir)/op_rocm.h
	$(HIPCC) $(HIPCCFLAGS) $(HIPCC_INCLUDES) \
	    $(op_rocm_CPPFLAGS) \
	    -c $< -o $@

AM_CPPFLAGS = $(op_rocm_CPPFLAGS)

# ----------------------------------------------------------------------------
# DSO build
# ----------------------------------------------------------------------------
if MCA_BUILD_ompi_op_rocm_DSO
component_install = mca_op_rocm.la
component_noinst  =
else
component_install =
component_noinst  = libmca_op_rocm.la
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)

mca_op_rocm_la_SOURCES  = $(sources)
mca_op_rocm_la_LDFLAGS  = -module -avoid-version $(op_rocm_LDFLAGS)
mca_op_rocm_la_LIBADD   = $(op_rocm_LIBS) op_rocm_kernels.o
mca_op_rocm_la_CPPFLAGS = $(op_rocm_CPPFLAGS)

# ----------------------------------------------------------------------------
# Static (convenience library) build
# ----------------------------------------------------------------------------
noinst_LTLIBRARIES = $(component_noinst)

libmca_op_rocm_la_SOURCES  = $(sources)
libmca_op_rocm_la_LDFLAGS  = -module -avoid-version $(op_rocm_LDFLAGS)
libmca_op_rocm_la_LIBADD   = $(op_rocm_LIBS) op_rocm_kernels.o
libmca_op_rocm_la_CPPFLAGS = $(op_rocm_CPPFLAGS)
