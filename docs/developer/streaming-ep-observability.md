# Streaming EP workload observability

Streaming logs report target group progress and per-rank work separately. The
counter `[002/248]` means the second of 248 target groups is executing. Groups
can contain different kinds of work, so that counter is not a percentage of
elapsed time, a layer count, or a calibration sample count. Long groups display
their first and last targets and the number of targets in the group.

## Phase boundaries and reporting cadence

Each group reports calibration, modifier update, and activation propagation as
separate phases. Each phase emits a table at entry, every 30 seconds while it
is running, and at exit. Failed phases emit their partial counters with a
`failed` status. The counters start from zero for every phase; calibration and
propagation do not share a token total.

The reporting thread reads host-side counters only. It does not synchronize a
device, create device events, copy device tensors to the CPU, or impose a
barrier between owners. The existing executor dependencies and lifecycle
boundaries continue to determine when work can advance.

## Reading forward tables

| Field | Meaning |
| --- | --- |
| `rank` | Position in the declared EP device list. |
| `device` | Logical execution device inside the process. |
| `visible` | Device identifier selected by the visibility environment, when available. For example, logical `cuda:0` can refer to visible device `4`. |
| `routed_tokens` | Sum of input rows submitted to experts on this owner in this phase. A token routed to several experts contributes once to each expert. This is not the number of original prompt tokens. |
| `share` | This owner's share of the phase's submitted routed tokens. |
| `experts/calls` | Distinct experts reached and expert invocations submitted after coalescing. One invocation may combine several calibration boundaries. |
| `dispatches` | Owner batches submitted. A dispatch can contain several expert invocations. |
| `avg/max_tokens` | Mean and maximum routed input tokens in an owner dispatch. |
| `capacity` | Admitted maximum tokens per dispatch for this owner. Owners can have different capacities. |
| `peak/cap` | Maximum dispatched tokens divided by this owner's capacity. This measures token budget usage, not memory or compute utilization. |
| `queue/pending` | Ready dispatches waiting in the scheduler and dispatches whose host futures have not been joined. |
| `host_s` | Wall time inside this owner's submission function, including host waits and copies that block that function. |
| `host_state` | Host worker state: waiting, queued, running, returned, or failed. |

`token_max/mean` compares the largest submitted owner token total to the mean
across all EP ranks, including idle ranks. It describes token imbalance.
Different expert shapes, attention work on the primary device, and different
modifier costs can produce different runtimes at the same token count.

The phase header also reports the number of boundary batches whose host forward
has returned. It does not imply that every device kernel has completed.
`primary` names the current primary-thread activity so a long attention call
can be distinguished from waiting for expert submission futures.

The host timing categories are:

- `load`: boundary metadata reads and preparation of execution inputs.
- `fx`: dependency traversal, route operations, primary-only calls, and their
  host execution or submission time.
- `pack`: expert grouping, concatenation, and preparation before submission.
- `wait`: time waiting for at least one outstanding owner host future.
- `boundary`: publishing outputs, boundary copies, and associated bookkeeping.

These are CPU wall times. Owner work overlaps the primary thread and other
owners, so times should not be added across ranks to estimate elapsed time.
A returned host future can still have device work in flight; its output is
consumed under the executor's existing stream event dependencies.

## Reading update and memory tables

Modifier update reports modules returned versus modules assigned to each owner,
host wall time, and host worker state. It uses the generic owner scheduler and
does not depend on GPTQ internals. A recipe using the serial modifier contract
reports that mode explicitly and attributes the lifecycle call to the primary
host worker. Modules are resident modules passed to the update contract; they
are not a claim that every module was quantized by a modifier.

Admission logs report rank, device, resident weight GiB, estimated modifier peak
GiB, runtime bytes per token, admitted token capacity, required GiB, and usable
GiB. At group completion each rank reports its peak allocated device memory in
GiB. The memory values use the existing admission and allocator measurements.

## Profiling and interpretation

Low `peak/cap` can reveal underfilled dispatches. A high `token_max/mean` can
reveal an imbalanced placement or route distribution. Neither field proves that
device compute is saturated. Long `fx` or `pack` host time can motivate a CPU
profile, while long owner host time can include device waits, transfers, or
modifier work. Use operator profiles and, when supported by the accelerator
runtime, device timelines to identify those costs.

The logs intentionally avoid claiming device busy percentages from host
measurements. Profiling captures, checkpoint locations, run identifiers, and
experimental measurements belong in private output directories outside version
control.
