#
# Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
    coll_sched.h \
    coll_sched_alloc.c \
    coll_sched_build_allreduce.c \
    coll_sched_build_bcast.c \
    coll_sched_build_reduce.c \
    coll_sched_component.c \
    coll_sched_do_allreduce.c \
    coll_sched_do_bcast.c \
    coll_sched_do_reduce.c \
    coll_sched_exec_pml.c \
    coll_sched_module.c

if MCA_BUILD_ompi_coll_sched_DSO
component_noinst =
component_install = mca_coll_sched.la
else
component_noinst = libmca_coll_sched.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_sched_la_SOURCES  = $(sources)
mca_coll_sched_la_LDFLAGS  = -module -avoid-version
mca_coll_sched_la_LIBADD   = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_sched_la_SOURCES  = $(sources)
libmca_coll_sched_la_LDFLAGS  = -module -avoid-version
