# Streaming PTQ

Streaming PTQ quantizes one sequential model target at a time. It preserves the
Modifier lifecycle and output format of `oneshot()` without requiring the full
uncompressed model to remain resident on the execution device.

This document defines the implemented architecture for the pretrained streaming
pipeline. The design has one execution device, one execution-ready CPU weight cache
for the next target, CPU-resident activation boundaries, and an optional bounded
asynchronous writer. The existing `pipeline_devices` API and model-specific
deferred-expert loading path have been removed.

To validate source or quantized checkpoints with cached prefill, decode and
generation, use the separate
[streaming reference inference entry point](streaming_inference.md).
It reuses native Transformers inference and Accelerate weight offload.

## Design goals

The pretrained streaming path must satisfy these invariants:

1. Only the current sequential target is resident on the execution device.
2. All weights in the current target remain resident for its complete
   calibration, Modifier update, propagation, compression, and save lifecycle.
3. In normal mode, complete activation boundaries remain in CPU memory. Recovery
   mode may persist them on disk, but stages only one batch through CPU. The
   execution device never owns a complete boundary.
4. Each calibration and propagation forward transfers exactly one boundary
   batch from CPU to the execution device. The number of stored samples must not
   increase execution-device residency.
5. At most one next target is materialized in CPU memory.
6. Checkpoint decoding and casting finish before a prepared target enters the
   execution-device transfer boundary.
7. Model mutation and forward execution remain on the main thread.
8. Background workers own immutable prepared CPU tensors or independent output
   snapshots, never live model parameters.
9. Default execution does not write intermediate calibration artifacts.
10. `checkpoint_progress=True` is an explicit recovery mode, not the normal
   execution path.
11. A memory reservation failure is reported before another target is loaded;
   the pipeline does not silently fall back to per-module checkpoint reads.

The design deliberately separates four concerns:

| Concern | Owner | Device contract |
| --- | --- | --- |
| Checkpoint interpretation | `CheckpointWeightSource` and `WeightMaterializer` | Produces execution-ready CPU tensors |
| Target execution | `SubgraphWeightSession` and the streaming pipeline | Uses one explicit execution device |
| Activation boundaries | `BoundaryActivationStore` | Owns complete normal-mode boundaries on CPU and serves one batch at a time |
| Output persistence | Direct or recovery writer | Owns independent CPU snapshots |

The subgraph working set includes checkpoint-backed modules called directly and
modules obtained through FX `get_attr` nodes and passed to another module (for
example, an embedding passed to an MTP layer). These dependencies are loaded for
the same lifecycle as the target and restored to meta afterward.

Calibration batches preserve their native Python numeric metadata, including image
dimensions, frame counts, and floating-point coefficients. These values and tensor
lengths may vary between samples while their types and container structure stay
fixed. Boolean flags, string modes, and optional input presence must remain
consistent. If tracing specializes a branch using numeric metadata, that input's
numeric values must also remain fixed; the streaming adapter rejects a mismatched
batch before executing its traced prefix. Input fingerprints still include the
actual metadata values, so changing calibration data invalidates cached results.

Next-target prefetch estimates the device working set from the current activation
boundary. Before installing the next target, the pipeline recomputes its requirement
from the actual input boundary and modifier state. If the requirement grew, it
reserves the additional capacity while retaining the prefetch reservation. A
capacity failure releases the prepared weights and reservations before target
execution.

## Class model

The following class diagram is the normative implementation contract.
Deferred-module classes and multi-device scheduling are intentionally absent.

```mermaid
classDiagram
    direction TB

    class run_subgraph_streaming_pipeline {
        <<function>>
        +execute()
    }

    class TracedBoundaryAdapter {
        +targets
        +target_subgraphs
        +calibration_boundaries()
    }

    class SubgraphPrefetcher {
        -session
        -pending
        +submit(plan, dtype, reservation, label)
        +take() PreparedSubgraphWeights
        +close(suppress_errors)
    }

    class SubgraphWeightSession {
        +working_set(subgraph)
        +plan(subgraph) SubgraphLoadPlan
        +plan_modules(names) SubgraphLoadPlan
        +estimate_preparation(plan, dtype) int
        +prepare(plan, dtype, reservation) PreparedSubgraphWeights
        +loaded_plan(plan, device, dtype) LoadedSubgraph
        +installed(prepared) LoadedSubgraph
    }

    class TargetWeightLoader {
        +plan(target_name) TargetLoadPlan
        +materialize_cpu(plan, dtype) PreparedTargetWeights
        +installed(prepared)
    }

    class PreparedSubgraphWeights {
        +plan
        +targets
        +device
        +nbytes
        +reservation
        +move_to(device)
        +close()
    }

    class PreparedTargetWeights {
        +plan
        +device
        +parameter_values
        +buffer_values
        +nbytes
        +close()
    }

    class LoadedSubgraph {
        +module_names
        +modules
        +state_tensors()
    }

    class HostMemoryBudget {
        +effective_available_bytes() int
        +reserve(owner, required_bytes) HostMemoryReservation
    }

    class HostMemoryReservation {
        +owner
        +reserved_bytes
        +committed_bytes
        +commit(bytes)
        +uncommit(bytes)
        +release_reserved(bytes)
        +release_committed(bytes)
        +close()
    }

    class BoundaryActivationStore {
        <<abstract>>
        +put(boundary, batch, value)
        +get(boundary, batch, device)
        +commit_propagated_batch(source, destination, batch, value)
        +batch_indices(boundary)
        +delete(boundary)
        #delete_batch(boundary, batch)
    }

    class InMemoryBoundaryActivationStore
    class DiskBoundaryActivationStore

    class CheckpointWeightSource {
        <<protocol>>
        +tensor_names()
        +metadata(name)
        +load_tensors_cpu(names)
    }

    class WeightMaterializer {
        <<abstract>>
        +dependencies(name, metadata)
        +logical_shape(name, metadata)
        +estimate_workspace_bytes(name, metadata, dtype) int
        +materialize_cpu(name, tensors, dtype)
    }

    class KimiK3WeightMaterializer

    class DirectSafetensorsWriter {
        +write_shard(shard_id, tensors)
    }

    class AsyncDirectSafetensorsWriter {
        +estimate_snapshot_bytes(tensors) int
        +write_shard(shard_id, tensors, reservation)
        +poll()
        +close()
    }

    class StreamingCheckpointWriter {
        +transaction(transaction_id)
        +committed_boundary(transaction_id)
    }

    run_subgraph_streaming_pipeline --> TracedBoundaryAdapter : executes trace plan
    run_subgraph_streaming_pipeline --> SubgraphPrefetcher : owns
    run_subgraph_streaming_pipeline --> BoundaryActivationStore : owns
    run_subgraph_streaming_pipeline --> HostMemoryBudget : checks before loading
    run_subgraph_streaming_pipeline --> DirectSafetensorsWriter : normal output
    run_subgraph_streaming_pipeline --> AsyncDirectSafetensorsWriter : optional wrapper
    run_subgraph_streaming_pipeline --> StreamingCheckpointWriter : recovery mode

    TracedBoundaryAdapter *-- SubgraphWeightSession
    SubgraphPrefetcher --> SubgraphWeightSession : prepares on CPU
    SubgraphPrefetcher ..> HostMemoryReservation : takes ownership
    SubgraphWeightSession *-- TargetWeightLoader
    SubgraphWeightSession --> PreparedSubgraphWeights : creates
    SubgraphWeightSession --> LoadedSubgraph : installs
    PreparedSubgraphWeights *-- PreparedTargetWeights
    PreparedSubgraphWeights *-- HostMemoryReservation : owns until released
    TargetWeightLoader --> CheckpointWeightSource
    TargetWeightLoader --> WeightMaterializer

    BoundaryActivationStore <|-- InMemoryBoundaryActivationStore
    BoundaryActivationStore <|-- DiskBoundaryActivationStore
    WeightMaterializer <|-- KimiK3WeightMaterializer
    AsyncDirectSafetensorsWriter o-- DirectSafetensorsWriter
    AsyncDirectSafetensorsWriter *-- HostMemoryReservation : pending snapshot owns
    HostMemoryBudget --> HostMemoryReservation : creates
```

`KimiK3WeightMaterializer` is shown as the representative model-specific
implementation. Cast and DeepSeek-V4 materializers implement the same CPU-only
`WeightMaterializer` contract.

The class contracts fix the following ownership decisions:

1. `SubgraphPrefetcher` always calls `prepare` for CPU and owns at most one
   future. CUDA mode may attach a bounded transfer reservation to that future.
2. The pipeline reserves host memory before `submit`. Ownership of that
   reservation moves to the prefetch task and then to
   `PreparedSubgraphWeights`; every failure path closes it.
3. `PreparedSubgraphWeights.move_to(device)` is the synchronous
   pipeline-facing transfer method. CUDA prefetch may use
   `move_to_async(device)` followed by `wait_for_transfer()` before install.
   Both paths replace one CPU tensor at a time and release source ownership
   after the copy is safely queued or completed. A partially moved object never
   escapes; any failure closes the complete aggregate after its tensors are
   released.
4. `SubgraphWeightSession.installed()` accepts only a completely prepared,
   single-device aggregate. Installation and restoration to meta remain on the
   main thread.
5. `BoundaryActivationStore.commit_propagated_batch()` snapshots the destination
   CPU batch before deleting the matching source batch. The pipeline does not
   call `delete_batch` directly.
6. `CheckpointWeightSource` and `WeightMaterializer` expose CPU-only load and
   materialization methods. Weight H2D is owned by
   `PreparedSubgraphWeights.move_to()` or its bounded asynchronous variant.
7. Before creating an output snapshot, the pipeline reserves its estimated CPU
   bytes. Ownership moves to the asynchronous writer and is released after the
   direct writer commits or fails. The writer never owns live model parameters
   or an activation boundary.
8. A reservation distinguishes unallocated `reserved_bytes` from allocated
   `committed_bytes`. Allocation commits bytes; freeing actual storage releases
   committed bytes; reusable workspace may be uncommitted back to its claim.
   State transitions and `close()` are thread-safe, and `close()` is idempotent
   because output reservations may be released by the writer thread.

## Execution lifecycle

For target N, the steady-state schedule is:

| Resource | Work |
| --- | --- |
| Execution device | Keep target N resident and execute one activation batch at a time |
| CPU boundary store | Retain boundary N and build boundary N + 1 batch by batch |
| CPU prefetch worker | Read and materialize target N + 1 |
| CPU writer worker | Optionally publish the independent snapshot of target N - 1 |

The target transition is ordered:

1. Finish all execution and snapshot ownership transfers for target N.
2. Restore target N to meta tensors and release its execution-device storage.
3. Wait for the prepared state for target N + 1; when a background H2D was
   admitted, wait for its completion event.
4. If the target is still CPU-prepared, move the prepared tensors to the
   execution device, releasing each CPU source tensor as ownership transfers.
5. Atomically install the complete prepared target into the meta model.
6. Start CPU materialization of target N + 2.
7. Execute target N + 1.

The first target is prepared synchronously because there is no preceding
computation to overlap. Every later target uses the same CPU prepare and device
transfer path; model-specific materializers do not bypass this lifecycle.

When the execution device is CUDA, the prepared next target can also enqueue its
CPU-to-device copies on a private CUDA stream. The transfer is admitted as a
second, bounded device working set and is waited on immediately before atomic
installation. If the device or pinned host budget cannot reserve that working
set, the pipeline keeps the target CPU-prepared and uses the synchronous
transfer path. When capacity allows, the steady-state schedule is:

```text
target N execution || target N+1 CPU preparation and H2D || target N-1 write
```

The device-transfer stage is capacity-gated. It may temporarily hold the
current and next target on the execution device, so devices that cannot fit both
working sets retain the one-target memory-minimal behavior. There is still one
execution device and no alternating model placement.

Set `LLMCOMPRESSOR_STREAMING_ASYNC_H2D=0` to disable background device transfers
and always wait for the CPU-to-device copy at the target boundary. The switch is
useful for memory-constrained hosts, for debugging, and for A/B measurements.

### Measured pipeline behavior

Measured on node16 (`llm-quant-base`), one 96 GiB GPU, with Qwen3-VL-32B-Instruct
RTN W8A16, 8 synthetic multimodal calibration samples, `max_seq_length=512`, 64
targets, and asynchronous checkpoint saving enabled:

| Configuration | Wall time | `load weights` total | Async transfer enqueue | Transfer wait | GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LLMCOMPRESSOR_STREAMING_ASYNC_H2D=0` | 154.7 s | 17.05 s | 0.00 s | 0.00 s | 2.87 GiB |
| `LLMCOMPRESSOR_STREAMING_ASYNC_H2D=1` | 141.0 s | 0.19 s | 8.04 s (overlapped) | 0.00 s | 3.84 GiB |

The asynchronous path transferred 57.22 GiB across 63 targets with no deferred
transfers, reduced wall time by 8.9%, and produced a byte-identical checkpoint.
The extra device peak of about 1 GiB is the bounded overlap of the current and
next target working sets.

## System sequence

The normal-mode steady-state sequence below shows the three independent work
lanes with `async_save=True`. Target N executes on the single execution device
while target N + 1 is decoded into CPU memory and the already-owned snapshot of
target N - 1 may be written. Model mutation, Modifier callbacks, and forward
execution stay on the main thread. The prepare lane is absent for the final
target, and the writer lane is absent for the first target or when asynchronous
saving is disabled.

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant Main as Pipeline main thread
    participant Adapter as TracedBoundaryAdapter
    participant Budget as HostMemoryBudget
    participant WeightRes as Weight HostMemoryReservation
    participant OutputRes as Output HostMemoryReservation
    participant Boundary as BoundaryActivationStore CPU
    participant Prefetch as SubgraphPrefetcher CPU worker
    participant Session as SubgraphWeightSession
    participant Source as CheckpointWeightSource
    participant Materializer as WeightMaterializer
    participant Prepared as PreparedSubgraphWeights CPU
    participant Device as Execution device
    participant Modifier as Modifier session
    participant Writer as AsyncDirectSafetensorsWriter
    participant Direct as Direct writer CPU worker

    Caller->>Main: streaming_oneshot()
    Main->>Adapter: trace targets and boundary contract
    Main->>Boundary: put initial boundary batches
    Main->>Modifier: initialize recipe and hooks
    Main->>Session: plan(target 0)
    Session-->>Main: SubgraphLoadPlan
    Main->>Session: estimate_preparation(plan, dtype)
    Session-->>Main: required host bytes
    Main->>Budget: reserve(target 0, required bytes)
    Budget-->>WeightRes: create target reservation
    WeightRes-->>Main: transfer ownership
    Main->>Session: prepare(plan, dtype, reservation)

    loop physical-order checkpoint groups
        Session->>Source: load source tensors on CPU
        Source-->>Session: native CPU tensors
        Session->>WeightRes: commit source-group bytes
        Session->>Materializer: decode and cast on CPU
        Materializer-->>Session: execution-ready CPU tensors
        Session->>WeightRes: commit prepared-tensor bytes
        Session->>Session: release raw source group
        Session->>WeightRes: uncommit(reusable group bytes)
    end

    Session->>WeightRes: release_reserved(unused workspace claim)
    Session-->>Main: PreparedSubgraphWeights target 0

    loop target N
        Main->>Prepared: move_to(execution device)
        loop each prepared tensor
            Prepared->>Device: H2D one tensor
            Prepared->>Prepared: release CPU source tensor
            Prepared->>WeightRes: release_committed(tensor bytes)
        end
        Prepared-->>Main: complete single-device aggregate
        Main->>Session: installed(prepared N)
        Session-->>Main: LoadedSubgraph N
        Note over Device: Complete target N plus one activation batch

        opt target N + 1 exists
            Main->>Session: plan(target N + 1)
            Session-->>Main: next SubgraphLoadPlan
            Main->>Session: estimate_preparation(next plan, dtype)
            Session-->>Main: next required host bytes
            Main->>Budget: reserve(target N + 1)
            Budget-->>WeightRes: create next target reservation
            WeightRes-->>Main: transfer ownership
            Main->>Prefetch: submit(next plan, dtype, reservation)
        end

        par Prepare target N + 1 on CPU
            Prefetch->>Session: prepare(next plan, dtype, reservation)
            loop physical-order checkpoint groups
                Session->>Source: load source group
                Source-->>Session: native CPU tensors
                Session->>WeightRes: commit source-group bytes
                Session->>Materializer: decode next group
                Materializer-->>Session: prepared CPU tensors
                Session->>WeightRes: commit prepared-tensor bytes
                Session->>Session: release raw source group
                Session->>WeightRes: uncommit(reusable group bytes)
            end
            Session->>WeightRes: release_reserved(unused workspace claim)
            Session-->>Prefetch: prepared target N + 1
            Note over Prefetch,Materializer: No model mutation
        and Execute target N on device
            loop calibration batch k
                Main->>Boundary: get(N, k, execution device)
                Boundary->>Device: H2D one boundary batch
                Boundary-->>Main: device-owned batch k
                Main->>Device: subgraph.forward(batch k)
                Device->>Modifier: hooks update statistics
                Device-->>Main: calibration output discarded
                Main->>Device: release batch and temporaries
            end

            Main->>Modifier: sequential_epoch_end(loaded modules)
            Modifier-->>Main: target weights updated

            loop propagation batch k
                Main->>Boundary: get(N, k, execution device)
                Boundary->>Device: H2D one boundary batch
                Boundary-->>Main: device-owned batch k
                Main->>Device: forward with hooks and QDQ disabled
                Device-->>Main: next boundary value
                Main->>Boundary: commit_propagated_batch(N, N + 1, k, value)
                Device-->>Boundary: D2H destination snapshot
                Boundary->>Boundary: delete source batch N[k]
                Boundary-->>Main: destination batch committed
                Main->>Device: release batch and temporaries
            end

            Main->>Device: compress resident target N
            Main->>Writer: estimate_snapshot_bytes(target N)
            Writer-->>Main: compressed snapshot bytes
            Main->>Budget: reserve(output shard N)
            Budget-->>OutputRes: create output reservation
            OutputRes-->>Main: transfer ownership
            Main->>Writer: write_shard(target N, output reservation)
            Device-->>Writer: synchronous D2H compressed weights
            Writer->>OutputRes: commit(snapshot bytes)
            Writer-->>Main: CPU-owned snapshot queued
        and Save target N - 1
            Writer->>Direct: encode, fsync, and publish pending shard
            Direct-->>Writer: shard committed
            Writer->>OutputRes: release_committed(snapshot bytes)
            Writer->>OutputRes: close reservation N - 1
        end

        Main->>Session: close installed context
        Session->>Device: restore target N to meta and release storage

        opt target N + 1 exists
            Main->>Prefetch: take()
            Prefetch-->>Main: PreparedSubgraphWeights target N + 1
        end
    end

    Main->>Writer: close and drain
    Writer->>Direct: finish final pending shard
    Main->>Modifier: finalize
    Main-->>Caller: finalized output directory
```

The execution lane intentionally performs two H2D passes over the same CPU
input boundary: calibration first, then post-Modifier propagation. The second
pass cannot reuse calibration outputs in the generic design because the Modifier
may have changed the resident weights.

Recovery mode preserves the same main-thread and execution-device sequence. It
substitutes `DiskBoundaryActivationStore` for the in-memory store and
`StreamingCheckpointWriter` transactions for the direct writers; the
asynchronous writer lane is disabled. Restored boundary batches still pass
through CPU one at a time.

## Relationship to `oneshot()`

The sequential `oneshot()` pipeline iterates calibration batches within each
subgraph and keeps onloaded parameters resident for the complete subgraph by
using `disable_offloading()`. Streaming PTQ must preserve that semantic boundary:
weights cannot be unloaded between calibration batches or before propagation.

The equivalent streaming order is:

```text
load target to execution device
  -> for each calibration batch: CPU-to-device, forward, release device batch
  -> sequential_epoch_end / Modifier update
  -> for each propagation batch: CPU-to-device, forward, snapshot next CPU batch
  -> compression
  -> compressed-weight snapshot to CPU
  -> unload target
```

This batch-major order is intentional. Expert-major execution is not required
when a complete MoE target fits on the execution device because every expert is
already resident. A future lower-memory execution strategy may use expert-major
scheduling, but it must be a separate, explicitly validated execution policy.

## Public API

The normal entry point accepts a local Transformers checkpoint, calibration
data, a standard recipe, one execution device, and output locations:

```python
from llmcompressor.streaming import streaming_oneshot

streaming_oneshot(
    model="/path/to/model",
    dataset=calibration_dataset,
    recipe=recipe,
    device="cuda:0",
    output_dir="/path/to/output",
    work_dir="/path/to/work",
    checkpoint_progress=False,
    async_save=True,
    pack_to_int8=True,
    finalize_only=False,
)
```

`pipeline_devices` is removed rather than deprecated. It described a bounded
two-GPU weight-prefetch implementation, not pipeline parallel execution, and it
would conflict with a future expert-placement API. Callers provide exactly one
`device` for target execution.

`async_save=True` remains independent of weight prefetch. It transfers completed
output tensors to independent CPU ownership synchronously, then performs
safetensors encoding and durable publication on one bounded background worker.
At most one output snapshot may be pending.

The advanced boundary-mode API remains available for adapters that provide an
explicit meta-model factory, target order, calibration boundaries, and exact
schemes. It follows the same single-device execution and CPU-prefetch contract.

`finalize_only=True` is a pretrained normal-mode recovery operation. It traces
the same target plan and rebuilds the same run identity, but does not initialize
modifiers or execute any target. It validates every completed shard under
`work_dir/publish` against that run identity, writes any still-missing static
model tensors, and then publishes the checkpoint.

### Shared calibration data helpers

Streaming examples use the public helpers in `llmcompressor.datasets` to load
calibration sources and build image/text conversations:

```python
from llmcompressor.datasets import (
    build_calibration_messages,
    load_calibration_datasets,
)

dataset, text_examples = load_calibration_datasets(
    sources=["lmms-lab/flickr30k"],
    split="test",
    text_sources=["HuggingFaceH4/ultrachat_200k", "/data/custom.jsonl"],
    text_split="train_sft",
    samples=32,
    text_samples=16,
    seed=42,
)

messages = build_calibration_messages(
    dataset[0],
    text_examples[0] if text_examples else None,
    image_format="image",
)
```

`load_calibration_source(source, split, samples)` preserves the original source
schema. Local JSON/JSONL files use the JSON builder's `train` split; Hub datasets
and local dataset directories use the requested split.

`load_calibration_dataset(sources, split, samples, seed=42)` divides the total
sample budget across sources, assigning any remainder to the first sources.
Each source is sliced before shuffling, matching the examples' existing sampling
order. Text sources with `messages` or `text` are normalized to one chat schema,
so differing metadata and JSON/Arrow message representations can be combined.
Image and pretokenized sources keep their original features; concatenated sources
must have compatible columns and feature types. A `preprocess=` callback can
override normalization with a model-specific encoder, as DeepSeek V4 does.
Pretokenized sources bypass this callback.

`load_calibration_datasets(...)` applies this loader independently to primary
and supplementary text sources, allowing different splits and sample budgets.
Supplementary text is returned separately for cyclic pairing with image rows;
it does not add primary calibration batches. Setting `text_samples=0` skips
loading supplementary text.

`build_calibration_messages(...)` handles image/caption and text-only records.
The `image_format` argument selects the processor's expected content block:

| Format | Image block | Example consumer |
| --- | --- | --- |
| `"image"` | `{"type": "image", "image": original_image}` | Kimi K3 |
| `"url"` | `{"type": "image", "url": png_data_url}` | GLM-5.3 |
| `"image_url"` | `{"type": "image_url", "image_url": {"url": png_data_url}}` | DeepSeek V4.1 |
| `"image_url_flat"` | `{"type": "image", "image_url": png_data_url}` | Kimi K2.5/K2.6 |

`format_calibration_text`, `normalize_text_content`, and `image_to_data_url`
are also public for custom adapters. PNG conversion accepts PIL images and
datasets image records containing bytes or paths. Model-specific tokenization,
image resizing, patch extraction, sequence truncation, and collation remain in
the respective processor or example.

The GLM-5.3, Kimi K2.5/K2.6 WNa8, and Qwen3.5 MoE iMatrix RTN examples in
`examples/multimodal_vision` use the same helpers for image/text calibration.
`load_calibration_datasets(..., text_preprocess=...)` allows a model-specific
policy for supplementary text. Qwen uses it to trim whitespace and map
unsupported or missing chat roles to `user` before cyclic pairing.

## Tracing and boundaries

The pretrained entry point builds a meta model and traces ordered sequential
targets. Runtime prefix dependencies hidden behind autowrapped calls are
discovered from a real sample batch; model classes must not declare private,
model-specific prefix dependency lists.

A boundary contains the forward values needed by the next target, such as
hidden states, masks, positions, and model-specific residual state. It does not
contain weights, quantization parameters, or durable output data.

With `checkpoint_progress=False`, adjacent boundaries are held in CPU memory and
deduplicated where possible. With `checkpoint_progress=True`, completed target
transactions and their next boundaries are persisted for recovery; each disk
batch is loaded through CPU immediately before it is moved to the execution
device. Neither mode places a complete boundary on the execution device.

### Activation ownership and transfer

In normal mode, the boundary store owns the complete input dataset for target N
on CPU. A boundary is batch-oriented: one stored batch may contain several
tensors such as hidden states, masks, positions, and residual state, but all of
those tensors remain on CPU between forwards. Recovery mode applies the same
batch contract after loading one persisted batch through CPU.

Calibration and propagation are separate passes because a Modifier update may
change the target weights. Both passes fetch one complete boundary batch and
move that batch to the execution device immediately before `subgraph.forward`:

```text
calibration:
  fetch CPU boundary batch N[k]
  -> H2D one batch
  -> forward with hooks enabled
  -> discard forward output and release the device batch

propagation:
  fetch CPU boundary batch N[k]
  -> H2D one batch
  -> forward with hooks disabled
  -> snapshot boundary batch N+1[k] to CPU
  -> delete consumed CPU boundary batch N[k]
  -> release the device input and output
```

The propagation snapshot must complete successfully before its input batch is
deleted. This preserves exception safety and ensures that the two adjacent
boundaries roll forward rather than accumulating as two complete datasets. Once
the final input batch is consumed, boundary N has no remaining storage.

Cross-device `.to(device)` already creates independent storage. Boundary
transfer helpers must not add an unconditional `.clone()` after a cross-device
copy. A clone is required only when returning an isolated value on the same
device as the store. Aliasing and equal-tensor deduplication within the stored
CPU value must remain correct.

The current implementation uses synchronous, single-batch H2D. Optional H2D
overlap may be added later with one bounded pinned staging batch, a dedicated
copy stream, and an event that the execution stream waits on. It must be an
explicit execution option because it allows two activation batches to be live
on the execution device. Pinning the complete activation boundary is not
allowed.

Propagation runs after Modifier update with hooks and fake quantization disabled,
matching the existing Sequential Pipeline. This preserves algorithms that mutate
full-precision weights during their update. A model-specific shortcut may not
reuse calibration outputs unless its supported recipe contract proves that the
update cannot change propagated values.

## CPU prepared-target cache

The CPU cache owns one `PreparedSubgraphWeights` instance for the next target.
Prepared means:

- all source tensors and declared dependencies were read;
- checkpoint-specific decoding is complete;
- tensors have the selected floating computation dtype;
- shapes and source mappings were validated;
- tensors are contiguous and ready for execution-device transfer;
- the shared meta model has not been mutated.

The cache does not retain raw checkpoint groups in addition to decoded tensors.
Each raw group is released as soon as its final prepared tensor is produced.
Requests are sorted by source shard and physical `storage_index`, and shard
handles are reused for the complete preparation task.

Before either synchronous preparation or prefetch, the pipeline calls
`SubgraphWeightSession.estimate_preparation(plan, dtype)`. The estimate combines
logical output shapes, checkpoint dependency metadata, and each materializer's
declared workspace. The returned peak is reserved before any source tensor is
read.

The prefetch worker has depth one. It receives an immutable `SubgraphLoadPlan`
and calls `SubgraphWeightSession.prepare(plan, dtype, reservation)`. Preparation
is CPU-only by contract. After preparation, CUDA mode may enqueue a bounded
pinned H2D transfer, but the worker never installs parameters, invokes Modifier
callbacks, or runs model code.

`SubgraphPrefetcher` is a prepared-target prefetcher with a CPU materialization
stage and an optional CUDA transfer stage. Its public
operations are:

```text
submit(plan, dtype, reservation, label) -> take ownership and start one task
take()                                  -> transfer prepared-state ownership
close()                                 -> drain or release the pending result
```

`CheckpointWeightSource.load_tensors_cpu()` and
`WeightMaterializer.materialize_cpu()` make the CPU preparation boundary
explicit all the way down the call stack. CUDA transfer uses a separately
reserved pinned staging buffer and never installs parameters from the worker.

## CPU-to-device ownership transfer

`PreparedTargetWeights` is a CPU-prepared leaf owned by
`PreparedSubgraphWeights`, which is the aggregate ownership boundary. The
aggregate exposes the only pipeline-facing move operation and transfers all of
its target tensors without mutating the model:

```text
prepared_subgraph_cpu.move_to(execution_device)
```

The pipelined CUDA path may instead use:

```text
prepared_subgraph_cpu.move_to_async(execution_device)
    -> wait_for_transfer()
    -> installed(prepared_subgraph)
```

The move mutates the aggregate in place. It must process one tensor at a time,
replace the CPU entry with the execution-device tensor, remove the corresponding
CPU reference immediately, and shrink its host-memory reservation. On failure
it closes the reservation and releases both moved and unconsumed tensors. Only a
completely moved, single-device aggregate may enter
`SubgraphWeightSession.installed()`.

When CPU is the execution device, `move_to("cpu")` is a no-op and the aggregate
retains its host-memory reservation until the installed context closes. This
keeps CPU-only execution under the same ownership accounting.

Installation remains atomic from the model's perspective: either every planned
parameter is installed and the target can execute, or the target remains meta.
Missing checkpoint state may allocate auxiliary tensors, but parameters with a
materialized source must never receive duplicate placeholder storage.

After parameters and persistent buffers are installed, their model registrations
become the sole owners of the materialized storage. The prepared-target maps must
drop their tensor references immediately. Otherwise replacing a BF16 parameter
during compression leaves the original storage alive until target unload, causing
the complete BF16 and compressed target to overlap on the execution device.

Pinned staging is bounded by the next target's materialized bytes and is
released after the CUDA transfer event completes. Pinning an entire model or
more than one pending target is not allowed.

## Materializer contract

A `WeightMaterializer` converts checkpoint-native tensors and dependencies into
the selected execution dtype. It must:

- declare all source dependencies before loading;
- expose the logical shape expected by the model;
- produce one independent floating CPU tensor;
- estimate any additional CPU workspace not represented by source or output
  tensor metadata before source reads begin;
- release raw dependency tensors after each materialized result;
- provide stable configuration data for artifact compatibility;
- map logical model tensor names to their serialized checkpoint names before
  every writer snapshot when a runtime requires a model-native key layout;
- avoid retaining global tensor or checkpoint-header caches.

The generic safetensors source uses the native safetensors loader for supported
dtypes and returns CPU tensors only. It must not manually parse safetensors
headers or data offsets. For a native `F8_E8M0` tensor, the source may expose its
storage bytes with `tensor.view(torch.uint8)` when a materializer requires UE8M0
exponent bits.

Format interpretation belongs to the materializer. Kimi K3 and DeepSeek-V4 use
the vLLM-compatible UE8M0 conversion:

```python
(scale_bytes.to(torch.int32) << 23).view(torch.float32)
```

This conversion is not equivalent to blindly casting every native E8M0 value to
FP32 at endpoint encodings. The conversion remains model-format logic; the
manual `_read_e8m0` file reader and its `lru_cache` do not.

DeepSeek-V4 calibration uses `model.*` logical names internally. Raw output mode
maps them back to the SGLang/vLLM checkpoint contract before writing each shard:
it removes the leading `model.`, maps `lm_head` to `head`, and maps
`weight_scale` to `scale`. Non-raw output mode keeps the internal logical names.
Both layouts preserve the source BF16 checkpoint's `architectures` and
`model_type`. Output-name mapping is part of the materializer fingerprint, so
the two layouts cannot share resume artifacts.

## MoE targets

If one decoded MoE target fits on the execution device, all routed experts are
materialized and remain resident for the full target lifecycle. This is the
default policy and matches the residency behavior of sequential `oneshot()`.

For Kimi K3, the materializer reads each expert's `weight_packed` and
`weight_scale`, decodes it to BF16 in the CPU prefetch worker, and transfers the
complete decoder target to the execution device before calibration. It must not
install per-expert forward wrappers or load experts from the checkpoint during a
forward pass.

At sequence length 2048 with top-16 routing, nearly every one of Kimi K3's 896
experts can be active. Treating `moe_calibrate_all_experts=False` as a memory or
I/O optimization is therefore unsafe; it only changes the routing semantics for
experts with no selected tokens.

### Future expert parallelism

Expert parallelism is a future execution-placement feature, not a prefetch
feature. It will distribute one resident MoE target across a device group and
add token scatter, expert execution, output gather, and observer reduction.

No expert-parallel CLI is exposed until that execution path exists. The current
implementation preserves a clean boundary between prepared CPU state and target
placement so a future placement object can replace the single-device move. It
must not retain `pipeline_devices` as a placeholder for expert parallelism.

## Modifier lifecycle

Streaming PTQ reuses normal Modifier hooks and `sequential_epoch_end`; it does
not maintain separate implementations of iMatrix, RTN, GPTQ, SmoothQuant, or AWQ.
Legacy SparseGPT and Wanda pruning were removed with the upstream sparsity APIs.

The session resolves quantization schemes once on the meta model so transforms
can inspect them. Weight and activation observers and quantization calibration
hooks are deferred until the current working set is installed. Streaming calls
the shared `QuantizationMixin.start_layerwise_calibration()` used by upstream
layerwise decompression, including scoped fused observers and scale dtype setup.
GPTQ keeps its own Hessian hooks and update implementation.

After all modifiers finish `sequential_epoch_end`, the shared
`end_layerwise_calibration()` releases the current group's calibration hooks and
observers. Failure cleanup runs before restoring the working set to meta.
Compression continues through compressed-tensors' `compress_module()` with the
streaming scale dtype preparation and checkpoint writer.

A modifier is compatible only if every module it reads or changes is present in
the current working set. Cross-target mappings require an explicit working-set
declaration. GPTQ must retain its Hessian and perform error propagation while the
weight is resident; it cannot be reduced to precomputed qparams plus generic
RTN.

The first E5 adapters reuse this lifecycle with explicitly limited recipes:

| Adapter | Accepted recipe and scope | Boundary propagation |
| --- | --- | --- |
| QAD | `[QuantizationModifier, QADModifier]`, weight-only quantization, CPU teacher/input/snapshot cache, at least two calibration batches | Recompute using the trained, fake-quantized weights |
| AutoSmooth | `[AutoSmoothModifier, QuantizationModifier]`, weight-only quantization, mappings whose smooth layer, balance layers and replay parent all belong to one target | Recompute after smoothing and weight calibration |
| AutoRound | Standalone `AutoRoundModifier`, symmetric INT4/INT8 weight-only Linear layers with both dimensions divisible by 32; `device_ids=None`, `enable_torch_compile=False` | Reuse the full-precision calibration output |

For example, an existing weight-only RTN recipe can add QAD:

```python
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import streaming_oneshot

streaming_oneshot(
    model="/path/to/local/checkpoint",
    dataset=calibration_dataloader,
    dataset_fingerprint="my-calibration-v1",
    recipe=[
        QuantizationModifier(
            targets="Linear", scheme="W4A16", ignore=["lm_head"]
        ),
        QADModifier(num_epochs=1, offload_device="cpu"),
    ],
    output_dir="/path/to/output",
    work_dir="/path/to/work",
    device="cuda:0",
    seed=42,
    checkpoint_progress=True,
)
```

QAD's `seed` controls its train/validation split and epoch shuffling. The
streaming `seed` also controls target-local algorithm RNG. A failed, uncommitted
target restarts from its original source weights; optimizer steps are not
persisted. Committed targets restore their output and activation frontier and
are marked complete without training again.

AutoSmooth resolves mappings on the meta model and checks their mutation scope
before loading target weights. Parameters shared outside that target are rejected
by all three adapters. Search replays use unquantized parent forwards, followed by
the ordinary RTN observer update. Arbitrary combinations with GPTQ, AWQ, or other
adapters are not admitted by these initial recipe contracts.

AutoRound runs independent block optimization: its private `q_input` is cleared
at each target, including after recovery. This explicitly differs from its
ordinary pipeline's cross-block `q_input` propagation. The adapter supplies the
pipeline's device and dtype, disables implicit AMP, fingerprints the installed
`auto-round` version, and requires each requested layer to return nonzero,
finite weight scales. AutoRound supplies updated weights/qparams; compression,
sharding, publication and recovery remain owned by the streaming pipeline.
External layer skipping is rejected rather than silently publishing an
unquantized result. Temporary parameter dtype/device changes and process-wide
determinism settings made by AutoRound are restored before returning to the
pipeline, including on failure.

SpinQuant and QuIP still fail preflight with an explicit transform-adapter
requirement. Their next stage needs per-target offline fusion, separate ownership
of embeddings/head/norms and tied weights, reproducible transform identity, and
online transform metadata that survives checkpoint reload. See the
[E5 implementation and validation notes](../developer/streaming-e5-validation.md).
The first adapters supply working-set estimates for admission; these are not
strict bounds for arbitrary model activations, attention workspaces, extra
forward kwargs, or external optimizer implementations.

## Output and recovery

The normal path uses `checkpoint_progress=False`:

- activation boundaries remain in CPU memory and are consumed batch by batch;
- each target is written once as a final safetensors shard;
- immutable run metadata is replaced at startup because normal mode never resumes
  calibration state from `work_dir/artifacts`;
- finalization adds the index, config, recipe, and auxiliary files;
- `work_dir/publish` is renamed to `output_dir` on the same filesystem;
- after successful publication, framework-owned `artifacts`, `boundaries`,
  `publish`, and `staging` entries are removed. The same successful-publication
  cleanup applies to checkpoint-progress mode and the low-level boundary API.
  The work root is removed only when empty, so unrelated caller files are
  preserved.

`checkpoint_progress=True` persists target transactions and boundary snapshots
for crash recovery. It has higher disk and I/O cost and must remain opt-in. CPU
next-target prefetch remains enabled because preparing immutable weights does not
change the durable transaction boundary. Asynchronous direct saving remains
disabled in recovery mode because recovery uses the transaction writer instead.

Recovery-mode manifests are strict resume records: a changed source, recipe,
dataset, target plan, or materializer is rejected. Normal-mode manifests are
replaceable publication metadata and are written before target execution. They
must not be regenerated after modifier finalization because modifier lifecycle
state is mutable and is not part of the immutable run identity.

If normal-mode quantization completed but final publication failed,
`finalize_only=True` preserves `work_dir/publish`, replaces stale normal-mode
metadata, verifies every target shard's stored run fingerprint, supplies static
tensors if the original run failed before that phase, and retries final
publication without requantizing. The caller must use the same model, dataset
fingerprint, recipe, target dtype, materializer, and packing option as the
completed run.

Any failure before or during publication retains the work directory for
diagnosis and `finalize_only` recovery. A successful `finalize_only` retry uses
the same cleanup rule as a successful normal run. Recovery mode with
`checkpoint_progress=True` retains durable progress until publication succeeds,
then removes the temporary artifacts and the empty work directory.

`pack_to_int8=True` packs W4A8 `int-quantized` weights before every writer path.
No unpacked INT4 checkpoint may be written as an intermediate normal-run output.

Do not load `work_dir/publish` while quantization is running. Only an output
directory containing `FINALIZED` is a published checkpoint.

## Memory model

Define:

| Symbol | Meaning |
| --- | --- |
| `T` | Largest decoded sequential target |
| `P` | Prepared CPU state for the next target |
| `C_in` | Unconsumed CPU input batches for the current target |
| `C_out` | Produced CPU output batches for the next target |
| `B` | Current execution-device activation batch, including forward outputs |
| `H` | Current execution-device Modifier statistics |
| `M_w` | Bounded CPU checkpoint-read and weight-materialization temporaries |
| `M_a` | One bounded CPU activation-copy temporary |
| `Q` | One independent compressed output snapshot |
| `R_d` | Execution framework, allocator, and device safety reserve |
| `R_h` | Host framework, allocator, and host safety reserve |

The steady-state lower bounds are:

```text
execution-device memory ~= T + B + H + runtime temporaries + R_d
host memory             ~= P + C_in + C_out + M_w + M_a + optional Q + R_h
```

The number of calibration batches changes `C_in` and `C_out`, not `B`. With
single-batch execution, increasing the sample count must not linearly increase
execution-device memory. During propagation, a successful output snapshot is
followed by deletion of the matching input batch, so `C_in` decreases as
`C_out` grows. For the brief commit window, the matching input and output
batches are both live; all other batches belong to exactly one adjacent
boundary.

If optional activation prefetch is implemented, the execution-device term
becomes approximately `T + 2 * B + H + runtime temporaries + R_d`. It must
never prefetch more than one batch.

Recovery mode replaces complete in-memory `C_in` and `C_out` with durable disk
boundaries and one CPU staging batch. It does not change the execution-device
bound because a complete boundary is never restored to the execution device.

During target transfer, CPU ownership is released tensor by tensor. The design
must not retain a complete CPU P after a complete device T exists.

Before submitting next-target prefetch, a host-memory reservation checks both
operating-system available memory and any active cgroup limit. That reservation
includes the next prepared target and its bounded materialization workspace.
Live activation batches and an asynchronous writer snapshot hold separate
reservations in the same shared budget. Each component therefore fails before
its corresponding allocation begins; insufficient capacity never disables
prefetch or switches to lazy checkpoint reads.

`HostMemoryBudget.effective_available_bytes()` uses the smaller of operating-
system available memory and cgroup remaining capacity, then subtracts `R_h` and
only the `reserved_bytes` of active reservations. It must not subtract
`committed_bytes`, because those allocations have already reduced the observed
OS and cgroup availability.

A reservation is accounting ownership, not an eager allocation. Weight
preparation initially claims `P + M_w`. Source groups and prepared outputs move
from reserved to committed accounting as they are allocated; released group
workspace is uncommitted for reuse by the next group. After decoding, the unused
workspace claim is released and the prepared aggregate retains committed `P`.
Its H2D move releases committed bytes tensor by tensor as CPU ownership ends.
The asynchronous writer separately claims `Q`, commits it during the synchronous
snapshot, and releases it after the pending write commits or fails. All
exceptions close the reservation held by the component that received ownership.

For GPTQ, add approximately `4 * in_features * in_features` bytes for each
active FP32 Hessian. iMatrix uses per-input-channel sums and counts and is much
smaller.

## Implementation map

| Area | Implemented responsibility |
| --- | --- |
| `streaming/pipeline.py` | Single-device scheduling, synchronous first-target preparation, depth-one CPU prefetch, batch-wise activation transfer, and bounded output ownership |
| `streaming/activations/store.py` | CPU boundary ownership, per-batch host reservations, and commit-before-consume propagation |
| `streaming/loading/host_memory.py` | OS/cgroup capacity checks and thread-safe reserved/committed accounting |
| `streaming/loading/prefetch.py` | One immutable CPU preparation task with explicit reservation ownership |
| `streaming/loading/session.py` | Preparation estimates, aggregate ownership, exception-safe `move_to()`, installation, and meta restoration |
| `streaming/loading/target.py` | Source-name planning, CPU-only target materialization, shape validation, and model installation |
| `streaming/materialization/` | CPU decoding, physical-order group reads, workspace estimates, and format-specific transforms |
| `streaming/checkpoint/weight_source.py` | Native CPU safetensors reads without manual header or offset parsing |
| `streaming/checkpoint/writer.py` | Direct output and one bounded asynchronous CPU snapshot with independent reservation ownership |

`async_save` is an output-I/O optimization and does not change target placement.
The removed multi-device and deferred-expert APIs have no compatibility aliases.
Production accelerator acceptance remains required for each new large-model
adapter even when the common CPU tests pass.

## Test and acceptance requirements

Fast tests must verify:

- a prepared CPU subgraph does not mutate the meta model;
- checkpoint sources and materializers return CPU tensors and cannot perform
  direct H2D;
- `PreparedSubgraphWeights.move_to()` is the only weight H2D boundary;
- aggregate transfer releases CPU tensors incrementally and a partial failure
  closes all moved and unconsumed storage;
- only one prefetch result can be pending;
- target N is unloaded before target N + 1 is installed;
- no target executes on a device other than the configured `device`;
- every complete in-memory boundary is CPU-resident;
- calibration and propagation transfer only one batch to the execution device;
- a failed propagated snapshot retains its input batch;
- a successful propagated snapshot deletes only its matching input batch;
- cross-device boundary transfer does not perform a redundant clone;
- increasing the calibration-batch count does not increase live
  execution-device boundary storage;
- source tensors are read once in physical storage order;
- raw materialization groups are released before the next group;
- a sourced parameter never receives duplicate placeholder storage;
- native U8 and `F8_E8M0` scale storage produce the expected UE8M0 bytes;
- writer and prefetch failures propagate and release owned storage;
- host-memory reservation failures occur before source reads;
- committed memory is not subtracted from OS availability a second time;
- reservation commit, uncommit, release, and close transitions reject
  underflow, are thread-safe, and close idempotently;
- prepared-state and writer ownership transfers release reservations on every
  success and failure path;
- checkpoint-progress recovery and normal final publication remain valid.

MoE tests must include a 896-expert, small-dimension fixture so expert count,
planning, naming, and target discovery match production cardinality without
requiring production memory. Kimi K3 remote-code tests must cover KDA and MLA
layers, attention-residual state, shared experts, multimodal prefix execution,
index-derived INT4 targets, W4 packing, and output reload.

Accelerator acceptance requires a production-shape MoE layer and then two
consecutive layers. Record:

- CPU prepared-target bytes and peak RSS;
- live `C_in`, `C_out`, and peak CPU boundary bytes;
- current and peak execution-device allocation;
- current and peak execution-device activation bytes;
- source read and materialization time;
- CPU-to-device transfer time;
- per-batch activation H2D and propagated-boundary D2H time;
- calibration, propagation, compression, and save time;
- finite iMatrix statistics and quantization scales;
- complete release of target N before target N + 1 execution;
- no complete activation boundary present on the execution device.

A tiny end-to-end pass is a functional check, not evidence that production
memory and scheduling behavior are correct.
