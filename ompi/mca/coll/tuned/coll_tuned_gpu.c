/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      Stony Brook University.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Analytical estimate for coll_tuned_gpu_reduce_threshold.
 *
 * Models two costs for one reduction call on device-resident buffers:
 *
 *   T_host(n)   ~= L_host + n * (2/BW_link + 1/BW_cpu_reduce)
 *   T_device(n) ~= L_ctrl + n * (1/BW_device)
 *
 * where n is the message size in bytes, BW_link is the host<->device link
 * bandwidth (paid twice: device-to-host then host-to-device), BW_cpu_reduce
 * is the assumed host reduction throughput, BW_device is the GPU's own
 * memory bandwidth, L_host is the fixed cost of the two staging memcpys, and
 * L_ctrl is the fixed cost of the persistent kernel's command round trip.
 * Solving T_host(n*) = T_device(n*) gives the crossover used as the
 * threshold:
 *
 *   n* = (L_ctrl - L_host) / (2/BW_link + 1/BW_cpu_reduce - 1/BW_device)
 *
 * BW_link and BW_device come from a per-device query (see
 * ompi_op_gpu_session_query_bandwidth()); BW_cpu_reduce, L_ctrl, and L_host
 * are not derivable from queryable system info and are exposed as separate
 * MCA parameters with conservative defaults instead (see
 * coll_tuned_component.c).
 */

#include "ompi_config.h"

#include <stdbool.h>

#include "ompi/constants.h"
#include "opal/mca/base/mca_base_var.h"
#include "ompi/op/op_gpu_session.h"
#include "coll_tuned.h"

/* Cache is indexed directly by dev_id, which accelerator components assign
 * as small consecutive integers; sized generously above any realistic
 * per-node device count. dev_id outside this range just isn't cached. */
#define GPU_THRESHOLD_CACHE_MAX 64

static double cached_threshold[GPU_THRESHOLD_CACHE_MAX];
static bool   cached_valid[GPU_THRESHOLD_CACHE_MAX] = { false };

/* --------------------------------------------------------------------------
 * compute_estimate
 *
 * Runs the formula from the file comment for one device. Returns the
 * compiled-in default (ompi_coll_tuned_gpu_reduce_threshold, i.e. 0 unless
 * the user changed it) if bandwidths can't be determined, or if the model's
 * inputs don't imply a positive crossover (e.g. the assumed control latency
 * isn't actually higher than the assumed staging latency -- in which case
 * the device path wins even at n=0, so a 0 threshold is exactly right, not
 * just a fallback).
 * -------------------------------------------------------------------------- */
static size_t
compute_estimate(int dev_id)
{
    double device_bw = 0.0, link_bw = 0.0;

    if (OMPI_SUCCESS != ompi_op_gpu_session_query_bandwidth(dev_id, &device_bw, &link_bw)
        || device_bw <= 0.0 || link_bw <= 0.0) {
        return ompi_coll_tuned_gpu_reduce_threshold;
    }

    double host_reduce_bw = (double) ompi_coll_tuned_gpu_host_reduce_bw_mbs * 1.0e6;
    double numer = ((double) ompi_coll_tuned_gpu_ctrl_latency_usec
                     - (double) ompi_coll_tuned_gpu_host_stage_latency_usec) * 1.0e-6;
    if (numer <= 0.0) {
        return 0;
    }

    double denom = (2.0 / link_bw) + (1.0 / host_reduce_bw) - (1.0 / device_bw);
    if (denom <= 0.0) {
        /* Degenerate (e.g. device_bw misreported as tiny); refuse to divide. */
        return 0;
    }

    double n = numer / denom;
    return (n < 0.0) ? 0 : (size_t) n;
}

/* --------------------------------------------------------------------------
 * ompi_coll_tuned_gpu_get_threshold
 * -------------------------------------------------------------------------- */
size_t
ompi_coll_tuned_gpu_get_threshold(int dev_id)
{
    mca_base_var_source_t source = MCA_BASE_VAR_SOURCE_DEFAULT;

    if (ompi_coll_tuned_gpu_reduce_threshold_index >= 0) {
        (void) mca_base_var_get_value(ompi_coll_tuned_gpu_reduce_threshold_index,
                                      NULL, &source, NULL);
    }
    if (MCA_BASE_VAR_SOURCE_DEFAULT != source) {
        /* User explicitly set it (--mca, env, or file) -- always honor
         * that, for every device, and skip estimation entirely. */
        return ompi_coll_tuned_gpu_reduce_threshold;
    }

    if (dev_id < 0 || dev_id >= GPU_THRESHOLD_CACHE_MAX) {
        return compute_estimate(dev_id);
    }

    if (!cached_valid[dev_id]) {
        cached_threshold[dev_id] = (double) compute_estimate(dev_id);
        cached_valid[dev_id] = true;
    }
    return (size_t) cached_threshold[dev_id];
}
