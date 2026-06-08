# coll/sched — Schedule IR and Execution Model

## Overview

`coll/sched` implements MPI collective operations using a two-phase design:
**schedule building** (rank-topology encoded at module-enable time, independent of
message size or datatype) and **execution** (applies the schedule to actual buffers
at call time via a pluggable executor).  The same schedule object is cached and
reused across all invocations of the same collective on the same communicator.

---

## The Schedule IR (`ompi_coll_sched_t`)

A schedule is a flat array of **steps**, each containing an array of **ops**.

```
ompi_coll_sched_t
  └── steps[0..num_steps-1]   ompi_coll_sched_step_t
        ├── ops[0..num_ops-1]  ompi_coll_sched_op_t
        └── barrier            bool
```

### Ops

Five op types exist:

| Type | Fields | Meaning |
|------|--------|---------|
| `SEND` | `peer`, `buf`, `comm_slot` | Non-blocking point-to-point send |
| `RECV` | `peer`, `buf`, `comm_slot` | Non-blocking point-to-point recv |
| `REDUCE` | `src`, `dst` | Local 2-buffer reduction: `dst = op(src, dst)` |
| `REDUCE3` | `src1`, `src2`, `dst` | Local 3-buffer reduction: `dst = op(src1, src2)` |
| `COPY` | `src`, `dst` | Local buffer copy |

`REDUCE3` is used in step 0 of any recv-then-reduce sequence to initialise `rbuf`
without requiring a pre-copy of `sbuf` into `rbuf`.  Subsequent steps use `REDUCE`.

`comm_slot` indexes into the `comms[]` array passed at execute time, enabling
multi-communicator (hierarchical) schedules without encoding actual communicator
pointers into the IR.

### Buffer references (`ompi_coll_sched_bufref_t`)

Every buffer operand is a `{buf_id, slot, num_slots}` triple:

- `buf_id`: `BUF_SEND` (user sbuf), `BUF_RECV` (user rbuf), or `BUF_TEMP(n)`.
- `slot` / `num_slots`: selects one equal partition of the buffer.

At execute time the executor resolves a bufref to a pointer and element count:

```
unit   = ceil(count / num_slots)
offset = slot * unit              (elements)
ptr    = base + offset * extent
n_elems = min(unit, count - offset)
```

`num_slots=1, slot=0` selects the whole buffer.  `num_slots=n` with varying `slot`
divides the buffer into per-rank chunks (used by ring allreduce and alltoall).

**Alltoall special case**: alltoall schedules use `num_slots=n` over an
`n*sendcount` total element count so each slot resolves to exactly `sendcount`
elements.  The dispatch must pass `n * sendcount` as `count` to the executor.

### Temp buffers

Up to `OMPI_COLL_SCHED_MAX_TEMP_BUFS` (4) scratch buffers may be registered via
`ompi_coll_sched_add_temp_buf(sched, full_size)`:

- `full_size=true`: allocate `count` elements (full message — used for
  recursive-doubling, reduce/bcast whole-buffer exchange).
- `full_size=false`: allocate `ceil(count/n)` elements (one chunk — used for
  ring allreduce receive scratch).

The executor allocates the temp bufs at execute time (after `count` is known) and
frees them when done.  The schedule IR stores only the sizing hint.

### Steps and barriers

Within a step, all network ops (SEND/RECV) are posted non-blocking first; local
ops (REDUCE/COPY) run after all network ops in that step have completed.
`barrier=true` forces the step to fully complete before the next step begins.
`barrier=false` allows the executor to chain directly into the next step.

Most algorithms use `barrier=false` because the data dependency is already encoded
in the step ordering (a recv in step k+1 cannot fire before step k's recv
completes).  `barrier=true` is used where local ops in step k write data that a
send in step k itself (same step) must read — e.g. bcast recv steps where the
same rank then sends forwarded data.

### What is NOT in the schedule

- Actual communicator pointers (only `comm_slot` indices).
- `count`, `datatype`, `op` — resolved at execute time.
- Peer rank translation for virtual-rank algorithms (baked in at build time via
  the rank and n parameters).

---

## Schedule Builders

Builders take `(rank, n)` — and `root` for rooted collectives — and return a
heap-allocated `ompi_coll_sched_t`.  They encode the algorithm topology for this
specific rank.  Builders never fail silently; they return NULL on OOM.

| Builder | Algorithm | Steps | Temp bufs |
|---------|-----------|-------|-----------|
| `build_allreduce_ring` | Reduce-scatter + allgather | 2(n-1) | 1 chunk |
| `build_allreduce_recursivedoubling` | Recursive doubling (power-of-2 only) | log₂n | 1 full |
| `build_allreduce_nonoverlapping` | Binomial reduce + binomial bcast | O(2 log n) | 1 full (if recvs) |
| `build_reduce_binomial` | Binomial tree reduce | O(log n) | 1 full (non-root receivers) |
| `build_bcast_binomial` | Binomial tree bcast | O(log n) | 0 |
| `build_bcast_chain` | Linear pipeline | ≤ 2 | 0 |
| `build_alltoall_linear` | Post all-to-all simultaneously | 1 | 0 |
| `build_alltoall_pairwise` | One exchange per step | n | 0 |

### Algorithm selection at dispatch time

| Collective | Condition | Algorithm |
|------------|-----------|-----------|
| Allreduce | n is power-of-2 | Recursive doubling |
| Allreduce | n is not power-of-2 | Nonoverlapping (binomial reduce + bcast) |
| Reduce | any | Binomial |
| Bcast | n == 2 | Chain (same step count, simpler) |
| Bcast | n > 2 | Binomial |
| Alltoall | n ≤ 6 | Pairwise (lower overhead for small n) |
| Alltoall | n > 6 | Linear |

---

## Schedule Cache

Each `mca_coll_sched_module_t` holds a per-communicator linear cache of up to 16
entries keyed by `(colltype, variant, root, param)`.  Schedules are built once and
reused.  On overflow the oldest entry is evicted (FIFO).

```c
ompi_coll_sched_cache_get(m, colltype, variant, root, param);
ompi_coll_sched_cache_put(m, colltype, variant, root, param, sched);
```

Schedules are freed in `mca_coll_sched_module_disable`.

---

## Executor Interface

```c
typedef struct ompi_coll_sched_exec_t {
    can_execute(exec, sched, comms, dtype, op) → bool
    execute(exec, sched, comms, sbuf, rbuf, count, dtype, op, base_tag) → int
    iexecute(exec, sched, comms, sbuf, rbuf, count, dtype, op, base_tag, *request) → int
    free(exec)
} ompi_coll_sched_exec_t;
```

`iexecute` is NULL for executors that do not support non-blocking operation.
`ompi_coll_sched_select_exec` returns the first executor whose `can_execute`
returns true; `ompi_coll_sched_select_iexec` additionally requires `iexecute != NULL`.

Two executors are registered, in priority order:

1. **CB executor** (`coll_sched_exec_cb.c`) — callback/continuation-based;
   supports both blocking (`execute`) and non-blocking (`iexecute`).
2. **PML executor** (`coll_sched_exec_pml.c`) — BSP poll-based; blocking only.

---

## CB Executor — Execution Model

### Blocking path (`cb_execute`)

1. Allocate `ompi_coll_sched_exec_cb_ctx_t` on the heap.
2. Allocate temp buffers.
3. Create an `ompi_coll_base_nbc_request_t` as internal completion signal (`comp_req`).
4. Call `start_step(ctx)` for step 0.
5. Block on `ompi_request_wait(&comp_req, ...)`.
6. Free temp buffers and ctx, return `ctx->rc`.

### Non-blocking path (`cb_iexecute`)

Same as blocking through step 4, then:

- Store ctx in `cr->req_complete_cb_data` so `cb_nbc_request_free` can free it.
- Set `cr->super.req_free = cb_nbc_request_free`.
- Return `&cr->super` as the user-visible `*request`.
- **Do not wait** — return immediately.

**Lifetime safety**: the dispatch calls `iexecute` with a stack-allocated
`comms[1]`.  The CB executor copies `comms[]` into `ctx->comms_buf[4]` and sets
`ctx->comms = ctx->comms_buf` before posting any network ops.

### Step execution (`start_step` / `advance`)

```
start_step(ctx):
    count net ops in this step
    if 0: execute_local_ops(); advance(); return
    ctx->pending = net_count   ← set BEFORE posting (avoids race)
    for each SEND/RECV op:
        MCA_PML_CALL(isend/irecv(..., &req))
        ompi_request_set_callback(req, net_cb, ctx)

net_cb(req):
    req->req_free(&req)        ← release PML request immediately
    if atomic_sub(ctx->pending, 1) != 0: return   ← not last
    execute_local_ops(ctx)     ← last: run local ops
    advance(ctx)               ← then chain to next step

advance(ctx):
    ctx->step++
    if ctx->step < num_steps: start_step(ctx)
    else: ompi_request_complete(ctx->comp_req, 1)
```

The atomic `pending` counter ensures exactly one callback runs local ops and
advances the state machine, even when completions arrive concurrently from a
progress thread.

### Resource cleanup

**Blocking**: ctx and temp bufs owned by `cb_execute`; freed after `ompi_request_wait`.
`cb_request_free` just finalises and releases the internal NBC request.

**Non-blocking**: ctx and temp bufs owned by the user-visible request.
`cb_nbc_request_free` frees:
1. Temp bufs in `ctx->temp_raw[]`.
2. The ctx itself.
3. Any extra allocations attached by the dispatch via
   `ompi_coll_base_append_array_to_release` (e.g. the non-root reduce
   scratch buffer `work_buf_raw`).

---

## PML Executor — Execution Model

BSP loop: for each step, post all SEND/RECV ops non-blocking, call
`ompi_request_wait_all`, then execute local ops.  No callbacks.
`iexecute` is NULL — not supported.

---

## Dispatch Functions

Each collective has two dispatch files: `coll_sched_do_<coll>.c` (blocking) and
`coll_sched_do_i<coll>.c` (non-blocking).  The dispatch pattern is:

```
1. Look up schedule in cache; build and cache if missing.
2. Select executor (select_exec or select_iexec).
3. If no executor: fall back to c_coll (lower-priority module).
4. Allocate any extra per-invocation buffers (e.g. non-root reduce scratch).
5. Call exec->execute() or exec->iexecute().
6. For NBC: attach extra buffers to request via append_array_to_release.
```

NBC dispatches also call `ompi_coll_base_nbc_reserve_tags(comm, 1)` to obtain a
unique per-invocation tag, preventing concurrent NBC collectives from matching
each other's messages.

---

## File Map

| File | Role |
|------|------|
| `coll_sched.h` | All types, constants, and function declarations |
| `coll_sched_alloc.c` | Schedule alloc/free, step/op builders, cache, executor selection |
| `coll_sched_exec_cb.c` | Callback executor (blocking + NBC) |
| `coll_sched_exec_pml.c` | BSP PML executor (blocking only) |
| `coll_sched_module.c` | MCA module lifecycle; executor + topology setup |
| `coll_sched_build_allreduce.c` | Ring, recursive-doubling, nonoverlapping builders |
| `coll_sched_build_reduce.c` | Binomial reduce builder |
| `coll_sched_build_bcast.c` | Binomial bcast + chain builders |
| `coll_sched_build_alltoall.c` | Linear + pairwise alltoall builders |
| `coll_sched_do_allreduce.c` | Blocking allreduce dispatch |
| `coll_sched_do_iallreduce.c` | NBC allreduce dispatch |
| `coll_sched_do_reduce.c` | Blocking reduce dispatch |
| `coll_sched_do_ireduce.c` | NBC reduce dispatch |
| `coll_sched_do_bcast.c` | Blocking bcast dispatch |
| `coll_sched_do_ibcast.c` | NBC bcast dispatch |
| `coll_sched_do_alltoall.c` | Blocking + NBC alltoall dispatch |
