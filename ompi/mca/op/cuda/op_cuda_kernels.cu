/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Persistent reduction kernels for the CUDA op component.
 *
 * OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL (set right below, flip and rebuild to
 * switch) selects between two complete implementations sharing everything
 * except how blocks are coordinated:
 *
 *   1 (default): multi-block, cooperative-groups grid.sync(). Each kernel
 *      runs across however many blocks of 1024 threads the device can run
 *      concurrently (see ompi_op_cuda_compute_coop_grid_size() below), so
 *      large reductions can use the whole device instead of a single SM's
 *      worth of bandwidth. Requires launching through the cooperative-launch
 *      API (cudaLaunchCooperativeKernel) on a device that supports it --
 *      true regardless of block count -- and treats a device lacking that
 *      capability as fatal (see the launcher comment).
 *
 *   0: single block of 1024 threads, __syncthreads() only, plain <<<>>>
 *      launch -- no cooperative-launch dependency, no occupancy query, at
 *      the cost of a ceiling of one SM's worth of memory bandwidth. Useful
 *      if cooperative-launch support turns out to be unavailable, broken,
 *      or to interact badly with other concurrent GPU work in a given
 *      environment (e.g. it has been observed to serialize against
 *      hipMemcpy on some ROCm configurations; see op_rocm_kernels.hip,
 *      which mirrors this same switch).
 *
 * Everything below this point (the poll loop, the atomic status protocol,
 * the barrier placement and why each one is needed, the work-splitting
 * loop, the elected-thread convention) is identical in both modes -- a
 * single block is just the degenerate case of a cooperative grid (gridDim.x
 * == 1, blockIdx.x always 0), so the same macro body produces both.
 *
 * cmd lives in plain device memory (see op_cuda_session.c): the host stages
 * src/dst/count and status=1 into a registered host mirror and pushes it
 * down with an explicit cudaMemcpyAsync, then polls this device slot's
 * status by copying it back until it reads 2.  The kernel here only ever
 * sees ordinary device-memory reads/writes -- it does not know or care that
 * the host side of the handoff goes through a memcpy rather than a shared
 * managed-memory pointer.
 *
 * There is no separate shutdown flag: status < 0 is the shutdown sentinel,
 * posted through the exact same channel as a normal command. status is
 * accessed through cuda::atomic_ref<int32_t, cuda::thread_scope_system>
 * rather than a plain volatile read/write: volatile only orders a single
 * thread's own accesses, it does not guarantee that once this thread
 * observes a value the host wrote, the ordinary (non-atomic) reads of
 * src1/src2/dst/count below see that same host write rather than a stale
 * cached one. The system-scope atomic gives that cross-agent guarantee.
 *
 * Every thread in every block polls status independently (sleeping between
 * polls), then snapshots the observed value into a local *before* the first
 * grid.sync() and branches on that snapshot afterward, not on a fresh
 * re-read of cmd->status. This matters even though every thread saw the
 * same value to get past its own poll loop: after the barrier releases
 * everyone, the elected thread races ahead to (on shutdown) reset status to
 * 0, or (on normal completion, see below) advance it to 2. A fresh re-read
 * by some other, slower thread could observe that already-updated value
 * instead of the one that unblocked its own poll loop, making it take the
 * wrong branch -- e.g. falling through to the compute path on shutdown,
 * where it would then hang forever at a barrier the already-exited elected
 * thread will never reach again. The pre-barrier snapshot is immune to
 * this because nothing else writes status in the window between a thread
 * unblocking its own poll and taking that snapshot.
 *
 * Two more grid.sync() calls bracket the actual reduction: one right after
 * the snapshot (so every block has confirmed the round is real work, not
 * shutdown, before any of them start touching src1/src2/dst -- and, on
 * shutdown, so the elected thread only resets status once every thread has
 * left its poll loop), and one right after the elected thread stores
 * status = 2. That last barrier is necessary, not just tidy: without it, a
 * non-elected block finishing early would loop back to the top of the
 * while(1) and could re-read status (still this round's stale 1, not yet
 * overwritten) before the elected thread's store happens, misreading it as
 * a fresh round and recomputing the same data a second time. Forcing every
 * block to wait for that store closes the window. grid.sync() is a full
 * memory fence across the grid, same as __syncthreads() is within a block,
 * so none of this needs additional fencing beyond the barriers themselves
 * and the status atomic.
 *
 * On shutdown, the elected thread (block 0, thread 0) resets status back to
 * 0 (idle) itself before the kernel returns, so the device slot is already
 * ready for the next launch without the host ever needing to push a reset.
 */

/* See the file comment above:
 * 1 = multi-block cooperative-groups kernel
 * 0 = single-block __syncthreads()-only kernel. */
#ifndef OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL
#define OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL 0
#endif

#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

#include "opal/util/output.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/cuda/op_cuda.h"

#if OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
/* Declares whatever the barrier below needs in scope (a grid_group handle,
 * in this mode) and performs the grid-wide barrier itself. */
#define OMPI_OP_CUDA_GRID_DECL   cg::grid_group grid = cg::this_grid();
#define OMPI_OP_CUDA_GRID_SYNC() grid.sync()
#else
#define OMPI_OP_CUDA_GRID_DECL
#define OMPI_OP_CUDA_GRID_SYNC() __syncthreads()
#endif

/* -------------------------------------------------------------------------
 * PERSISTENT_KERNEL(name, ctype, op_expr)
 *
 * Generates __global__ void ompi_op_cuda_persistent_<name>(...).
 * op_expr must be a statement writing dst[i] from src1[i] and src2[i],
 * e.g. "dst[i] = src1[i] + src2[i]".  src2 may alias dst for in-place ops.
 * ------------------------------------------------------------------------- */
#define PERSISTENT_KERNEL(kname, ctype, op_expr)                               \
__global__ void ompi_op_cuda_persistent_##kname(ompi_op_gpu_cmd_t *cmd)         \
{                                                                               \
    OMPI_OP_CUDA_GRID_DECL                                                      \
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> status_atomic(cmd->status); \
    int64_t grid_stride = (int64_t) blockDim.x * (int64_t) gridDim.x;           \
    bool elected = (0 == blockIdx.x && 0 == threadIdx.x);                      \
    while (1) {                                                                \
        /* Spin-wait for a new command or a shutdown request; sleep 1 µs       \
         * between polls to save power. status == 2 means the previous       \
         * result hasn't been reclaimed by the host yet -- keep waiting in    \
         * that case too, or every thread would immediately redo the last    \
         * reduction on stale operands instead of sleeping. status < 0 is    \
         * the shutdown sentinel. */                                          \
        int32_t status = status_atomic.load(cuda::memory_order_acquire);      \
        while (status != 1 && status >= 0) {                                  \
            __nanosleep(500);                                                \
            status = status_atomic.load(cuda::memory_order_acquire);          \
        }                                                                     \
        /* Every thread in every block has now individually exited its       \
         * poll loop above -- required before anyone acts on the shutdown    \
         * branch below; see the file comment for why. */                     \
        OMPI_OP_CUDA_GRID_SYNC();                                               \
        if (status < 0) {                                                 \
            /* Leave the slot idle (status == 0) for the next launch -- the   \
             * host never needs to push a reset. */                            \
            if (elected) { status_atomic.store(0, cuda::memory_order_relaxed); } \
            break;                                                             \
        }                                                                      \
        const ctype * __restrict__ src1 = (const ctype *) cmd->src1;           \
        const ctype * __restrict__ src2 = (const ctype *) cmd->src2;           \
              ctype * __restrict__ dst  = (      ctype *) cmd->dst;            \
        int64_t n = cmd->count;                                                 \
        for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;      \
             i < n; i += grid_stride) {                                        \
            op_expr;                                                            \
        }                                                                       \
        __threadfence_system();   /* ensure dst writes reach host */            \
        OMPI_OP_CUDA_GRID_SYNC();                                                \
        if (elected) {                                                          \
            status_atomic.store(2, cuda::memory_order_relaxed);          /* signal done */                         \
        }                                                                       \
        /* Wait for the elected thread to store status = 2 before any           \
         * block loops back to the top and re-reads it, or a fast block         \
         * could misread that stale 1 as a new round and redo the reduction.    \
         */                                                                     \
        OMPI_OP_CUDA_GRID_SYNC();                                                \
    }                                                                           \
}

/* =========================================================================
 * Kernel instantiations
 * ========================================================================= */

/* --- MAX --- */
PERSISTENT_KERNEL(max_int8,   int8_t,   dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_uint8,  uint8_t,  dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_int16,  int16_t,  dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_uint16, uint16_t, dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_int32,  int32_t,  dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_uint32, uint32_t, dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_int64,  int64_t,  dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_uint64, uint64_t, dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_float,  float,    dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(max_double, double,   dst[i] = src1[i] > src2[i] ? src1[i] : src2[i])

/* --- MIN --- */
PERSISTENT_KERNEL(min_int8,   int8_t,   dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_uint8,  uint8_t,  dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_int16,  int16_t,  dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_uint16, uint16_t, dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_int32,  int32_t,  dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_uint32, uint32_t, dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_int64,  int64_t,  dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_uint64, uint64_t, dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_float,  float,    dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])
PERSISTENT_KERNEL(min_double, double,   dst[i] = src1[i] < src2[i] ? src1[i] : src2[i])

/* --- SUM --- */
PERSISTENT_KERNEL(sum_int8,   int8_t,   dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_uint8,  uint8_t,  dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_int16,  int16_t,  dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_uint16, uint16_t, dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_int32,  int32_t,  dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_uint32, uint32_t, dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_int64,  int64_t,  dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_uint64, uint64_t, dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_float,  float,    dst[i] = src1[i] + src2[i])
PERSISTENT_KERNEL(sum_double, double,   dst[i] = src1[i] + src2[i])

/* --- PROD --- */
PERSISTENT_KERNEL(prod_int8,   int8_t,   dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_uint8,  uint8_t,  dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_int16,  int16_t,  dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_uint16, uint16_t, dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_int32,  int32_t,  dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_uint32, uint32_t, dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_int64,  int64_t,  dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_uint64, uint64_t, dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_float,  float,    dst[i] = src1[i] * src2[i])
PERSISTENT_KERNEL(prod_double, double,   dst[i] = src1[i] * src2[i])

/* --- BAND (bitwise AND, integer types only) --- */
PERSISTENT_KERNEL(band_int8,   int8_t,   dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_uint8,  uint8_t,  dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_int16,  int16_t,  dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_uint16, uint16_t, dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_int32,  int32_t,  dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_uint32, uint32_t, dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_int64,  int64_t,  dst[i] = src1[i] & src2[i])
PERSISTENT_KERNEL(band_uint64, uint64_t, dst[i] = src1[i] & src2[i])

/* --- BOR (bitwise OR) --- */
PERSISTENT_KERNEL(bor_int8,   int8_t,   dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_uint8,  uint8_t,  dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_int16,  int16_t,  dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_uint16, uint16_t, dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_int32,  int32_t,  dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_uint32, uint32_t, dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_int64,  int64_t,  dst[i] = src1[i] | src2[i])
PERSISTENT_KERNEL(bor_uint64, uint64_t, dst[i] = src1[i] | src2[i])

/* --- BXOR (bitwise XOR) --- */
PERSISTENT_KERNEL(bxor_int8,   int8_t,   dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_uint8,  uint8_t,  dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_int16,  int16_t,  dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_uint16, uint16_t, dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_int32,  int32_t,  dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_uint32, uint32_t, dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_int64,  int64_t,  dst[i] = src1[i] ^ src2[i])
PERSISTENT_KERNEL(bxor_uint64, uint64_t, dst[i] = src1[i] ^ src2[i])

#if OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL
/* =========================================================================
 * ompi_op_cuda_compute_coop_grid_size
 *
 * Number of 1024-thread blocks kernel_fn can run concurrently on dev_id,
 * i.e. cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * SM count -- the
 * most blocks this persistent kernel can ever run without some of them
 * waiting for an SM slot that will never free up (every block spins forever
 * until shutdown, so under-occupancy here is not a transient slowdown, it's
 * a permanent deadlock at launch).
 *
 * grid.sync() requires the kernel to run under the cooperative-launch API
 * on a device that supports it, regardless of block count, so this also
 * checks cudaDevAttrCooperativeLaunch. There is no plausible degraded
 * behavior for this kernel body on a device where either query fails --
 * treated as fatal, same as a launch failure already is in
 * ompi_op_cuda_session_reduce().
 * ========================================================================= */
static int
ompi_op_cuda_compute_coop_grid_size(const void *kernel_fn, int dev_id, int block_size)
{
    int supports_coop = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch,
                                             dev_id);
    if (cudaSuccess != err || 0 == supports_coop) {
        opal_output(0, "op/cuda: device %d does not support cooperative kernel "
                   "launch, required for the multi-block persistent reduction "
                   "kernel", dev_id);
        abort();
    }

    int sm_count = 0;
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    if (cudaSuccess != err || sm_count <= 0) {
        opal_output(0, "op/cuda: failed to query SM count on device %d: %s",
                   dev_id, cudaGetErrorString(err));
        abort();
    }

    int blocks_per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_fn,
                                                        block_size, 0);
    if (cudaSuccess != err || blocks_per_sm <= 0) {
        opal_output(0, "op/cuda: failed to query kernel occupancy on device %d: %s",
                   dev_id, cudaGetErrorString(err));
        abort();
    }

    return blocks_per_sm * sm_count;
}

/* Devices per node this process could plausibly see; only used to size a
 * per-launcher occupancy cache indexed by dev_id -- a dev_id outside this
 * range just isn't cached (recomputed each call, which is correct, only
 * slower). */
#define OMPI_OP_CUDA_MAX_DEVICES 64

/* =========================================================================
 * Host-side launcher wrappers — one per kernel, occupancy-sized grid of
 * 1024-thread blocks, launched cooperatively so the kernel body can use
 * grid.sync() to coordinate across blocks.
 * ========================================================================= */
#define LAUNCHER(kname)                                                        \
static void launch_##kname(ompi_op_gpu_cmd_t *cmd,                            \
                            cudaStream_t       stream,                         \
                            int                dev_id)                        \
{                                                                               \
    static int cached_blocks[OMPI_OP_CUDA_MAX_DEVICES];                       \
    int blocks = (dev_id >= 0 && dev_id < OMPI_OP_CUDA_MAX_DEVICES)            \
                 ? cached_blocks[dev_id] : 0;                                   \
    if (0 == blocks) {                                                          \
        blocks = ompi_op_cuda_compute_coop_grid_size(                          \
            (const void *) ompi_op_cuda_persistent_##kname, dev_id, 1024);     \
        if (dev_id >= 0 && dev_id < OMPI_OP_CUDA_MAX_DEVICES) {                \
            cached_blocks[dev_id] = blocks;                                    \
        }                                                                       \
    }                                                                            \
    void *args[] = { (void *) &cmd };                                            \
    cudaLaunchCooperativeKernel((void *) ompi_op_cuda_persistent_##kname,       \
                                dim3(blocks), dim3(1024), args, 0, stream);     \
}
#else /* !OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL */
/* =========================================================================
 * Host-side launcher wrappers — one per kernel, single block of 1024
 * threads, plain launch. dev_id is unused here (kept so the function
 * pointer type matches the cooperative build's launcher unconditionally --
 * see ompi_op_cuda_launcher_fn_t in op_cuda.h).
 * ========================================================================= */
#define LAUNCHER(kname)                                                        \
static void launch_##kname(ompi_op_gpu_cmd_t *cmd,                            \
                            cudaStream_t       stream,                         \
                            int                dev_id)                        \
{                                                                               \
    (void) dev_id;                                                             \
    ompi_op_cuda_persistent_##kname<<<1, 1024, 0, stream>>>(cmd);              \
}
#endif /* OMPI_OP_CUDA_USE_COOPERATIVE_KERNEL */

LAUNCHER(max_int8)    LAUNCHER(max_uint8)
LAUNCHER(max_int16)   LAUNCHER(max_uint16)
LAUNCHER(max_int32)   LAUNCHER(max_uint32)
LAUNCHER(max_int64)   LAUNCHER(max_uint64)
LAUNCHER(max_float)   LAUNCHER(max_double)

LAUNCHER(min_int8)    LAUNCHER(min_uint8)
LAUNCHER(min_int16)   LAUNCHER(min_uint16)
LAUNCHER(min_int32)   LAUNCHER(min_uint32)
LAUNCHER(min_int64)   LAUNCHER(min_uint64)
LAUNCHER(min_float)   LAUNCHER(min_double)

LAUNCHER(sum_int8)    LAUNCHER(sum_uint8)
LAUNCHER(sum_int16)   LAUNCHER(sum_uint16)
LAUNCHER(sum_int32)   LAUNCHER(sum_uint32)
LAUNCHER(sum_int64)   LAUNCHER(sum_uint64)
LAUNCHER(sum_float)   LAUNCHER(sum_double)

LAUNCHER(prod_int8)   LAUNCHER(prod_uint8)
LAUNCHER(prod_int16)  LAUNCHER(prod_uint16)
LAUNCHER(prod_int32)  LAUNCHER(prod_uint32)
LAUNCHER(prod_int64)  LAUNCHER(prod_uint64)
LAUNCHER(prod_float)  LAUNCHER(prod_double)

LAUNCHER(band_int8)   LAUNCHER(band_uint8)
LAUNCHER(band_int16)  LAUNCHER(band_uint16)
LAUNCHER(band_int32)  LAUNCHER(band_uint32)
LAUNCHER(band_int64)  LAUNCHER(band_uint64)

LAUNCHER(bor_int8)    LAUNCHER(bor_uint8)
LAUNCHER(bor_int16)   LAUNCHER(bor_uint16)
LAUNCHER(bor_int32)   LAUNCHER(bor_uint32)
LAUNCHER(bor_int64)   LAUNCHER(bor_uint64)

LAUNCHER(bxor_int8)   LAUNCHER(bxor_uint8)
LAUNCHER(bxor_int16)  LAUNCHER(bxor_uint16)
LAUNCHER(bxor_int32)  LAUNCHER(bxor_uint32)
LAUNCHER(bxor_int64)  LAUNCHER(bxor_uint64)

/* =========================================================================
 * 2D launcher table [op_index][type_index]
 *
 * Indexed by OMPI_OP_BASE_FORTRAN_* (rows) × OMPI_OP_BASE_TYPE_* (columns).
 * Zero/NULL entries mean "not supported on GPU" → host fallback.
 *
 * Zero-initialized here; filled by ompi_op_cuda_kernel_fns_init() called
 * from cuda_component_open().  The init function uses plain assignment
 * instead of designated initializers to stay compatible with NVCC's C++
 * frontend, which does not support non-trivial designated initializers.
 * ========================================================================= */
ompi_op_cuda_launcher_fn_t
ompi_op_cuda_kernel_fns[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];

void
ompi_op_cuda_kernel_fns_init(void)
{
#define SET(op, type, fn) \
    ompi_op_cuda_kernel_fns[OMPI_OP_BASE_FORTRAN_##op][OMPI_OP_BASE_TYPE_##type] = (fn)

    SET(MAX, INT8_T,   launch_max_int8);
    SET(MAX, UINT8_T,  launch_max_uint8);
    SET(MAX, INT16_T,  launch_max_int16);
    SET(MAX, UINT16_T, launch_max_uint16);
    SET(MAX, INT32_T,  launch_max_int32);
    SET(MAX, UINT32_T, launch_max_uint32);
    SET(MAX, INT64_T,  launch_max_int64);
    SET(MAX, UINT64_T, launch_max_uint64);
    SET(MAX, FLOAT,    launch_max_float);
    SET(MAX, DOUBLE,   launch_max_double);

    SET(MIN, INT8_T,   launch_min_int8);
    SET(MIN, UINT8_T,  launch_min_uint8);
    SET(MIN, INT16_T,  launch_min_int16);
    SET(MIN, UINT16_T, launch_min_uint16);
    SET(MIN, INT32_T,  launch_min_int32);
    SET(MIN, UINT32_T, launch_min_uint32);
    SET(MIN, INT64_T,  launch_min_int64);
    SET(MIN, UINT64_T, launch_min_uint64);
    SET(MIN, FLOAT,    launch_min_float);
    SET(MIN, DOUBLE,   launch_min_double);

    SET(SUM, INT8_T,   launch_sum_int8);
    SET(SUM, UINT8_T,  launch_sum_uint8);
    SET(SUM, INT16_T,  launch_sum_int16);
    SET(SUM, UINT16_T, launch_sum_uint16);
    SET(SUM, INT32_T,  launch_sum_int32);
    SET(SUM, UINT32_T, launch_sum_uint32);
    SET(SUM, INT64_T,  launch_sum_int64);
    SET(SUM, UINT64_T, launch_sum_uint64);
    SET(SUM, FLOAT,    launch_sum_float);
    SET(SUM, DOUBLE,   launch_sum_double);

    SET(PROD, INT8_T,   launch_prod_int8);
    SET(PROD, UINT8_T,  launch_prod_uint8);
    SET(PROD, INT16_T,  launch_prod_int16);
    SET(PROD, UINT16_T, launch_prod_uint16);
    SET(PROD, INT32_T,  launch_prod_int32);
    SET(PROD, UINT32_T, launch_prod_uint32);
    SET(PROD, INT64_T,  launch_prod_int64);
    SET(PROD, UINT64_T, launch_prod_uint64);
    SET(PROD, FLOAT,    launch_prod_float);
    SET(PROD, DOUBLE,   launch_prod_double);

    SET(BAND, INT8_T,   launch_band_int8);
    SET(BAND, UINT8_T,  launch_band_uint8);
    SET(BAND, INT16_T,  launch_band_int16);
    SET(BAND, UINT16_T, launch_band_uint16);
    SET(BAND, INT32_T,  launch_band_int32);
    SET(BAND, UINT32_T, launch_band_uint32);
    SET(BAND, INT64_T,  launch_band_int64);
    SET(BAND, UINT64_T, launch_band_uint64);

    SET(BOR, INT8_T,   launch_bor_int8);
    SET(BOR, UINT8_T,  launch_bor_uint8);
    SET(BOR, INT16_T,  launch_bor_int16);
    SET(BOR, UINT16_T, launch_bor_uint16);
    SET(BOR, INT32_T,  launch_bor_int32);
    SET(BOR, UINT32_T, launch_bor_uint32);
    SET(BOR, INT64_T,  launch_bor_int64);
    SET(BOR, UINT64_T, launch_bor_uint64);

    SET(BXOR, INT8_T,   launch_bxor_int8);
    SET(BXOR, UINT8_T,  launch_bxor_uint8);
    SET(BXOR, INT16_T,  launch_bxor_int16);
    SET(BXOR, UINT16_T, launch_bxor_uint16);
    SET(BXOR, INT32_T,  launch_bxor_int32);
    SET(BXOR, UINT32_T, launch_bxor_uint32);
    SET(BXOR, INT64_T,  launch_bxor_int64);
    SET(BXOR, UINT64_T, launch_bxor_uint64);

    /* LAND, LOR, LXOR, MAXLOC, MINLOC, REPLACE, NO_OP: NULL → host path */
#undef SET
}
