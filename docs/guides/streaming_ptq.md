# Streaming PTQ

Streaming PTQ is an experimental, out-of-core alternative to `oneshot()` for
models whose full uncompressed weights do not fit in RAM or accelerator memory.
It does not change the existing `oneshot()` path. The normal pretrained entry
point currently supports local Transformers checkpoints with dense `nn.Linear`
quantization targets. DeepSeek-V4 raw FP8+FP4 checkpoints are supported when a
`DeepSeekV4WeightMaterializer` is supplied; the materializer decodes each
requested target to BF16 on demand.

The workflow has three durable stages:

1. `collect_calibration_statistics()` loads one sequential target at a time,
   stores GPTQ/iMatrix sufficient statistics, and releases its weights.
2. `quantize_streaming()` rereads source shards, restores those statistics,
   compresses quantized Linear weights, and atomically writes staging shards.
3. `finalize_streaming_checkpoint()` validates every staging shard, creates the
   safetensors index and compressed-tensors config, copies auxiliary files, and
   atomically publishes the output directory.

`streaming_oneshot()` runs all three stages. Completed statistics and staging
shards are reused when their fingerprints match. The public Python API accepts
a local checkpoint, calibration data, and a normal iMatrix + RTN/GPTQ recipe.
It derives decoder targets, exact module schemes, and layer-zero boundaries
without loading the full model. See
`examples/streaming_oneshot/qwen3_0_6b_imatrix_rtn.py`.

The advanced boundary-mode API remains available for model adapters that need
custom target ordering or non-standard forward inputs. It requires a meta-model
factory, ordered targets, precomputed boundary batches, and exact module schemes.

The CLI exposes the same boundaries:

```bash
llmcompressor.streaming collect --help
llmcompressor.streaming quantize --help
llmcompressor.streaming finalize --help
llmcompressor.streaming run --help
```

The CLI model factory uses `package.module:callable` syntax. Recipe, schemes,
and calibration inputs are local files: JSON for recipe/schemes and a trusted
`torch.save` list or tuple for calibration batches.

Current constraints are deliberate: reference calibration semantics only;
dense `nn.Linear` targets; GPTQ and iMatrix; local safetensors checkpoints;
single-device execution. AWQ, SmoothQuant, AutoSmooth, AutoRound, pruning, and
arbitrary fused checkpoint layouts are rejected or outside this initial
implementation. MoE models are supported only when their expert projections are
represented as ordinary Linear modules and their target execution can be traced;
DeepSeek-V4 uses its dedicated materializer for the raw FP8+FP4 layout. Custom
source formats should supply a `WeightMaterializer` that converts the requested
weight and its declared dependencies to the selected floating computation dtype.

Do not load a staging directory as a model. It intentionally has no index or
config. Only an output directory containing `FINALIZED` is a published result.

## Peak memory and disk requirements

Streaming reduces the model-weight residency of calibration, but it does not
make every operation independent of checkpoint shard size. Plan capacity for
the largest of the following quantities:

| Symbol | Meaning | Where it matters |
| --- | --- | --- |
| `T` | Decoded parameters and persistent buffers in the largest sequential target | Stage 1 CPU/RAM or accelerator memory |
| `A` | One calibration boundary batch, including tensors such as hidden states, masks, and position information | Stage 1 CPU/RAM or accelerator memory |
| `W` | One source weight plus its dependencies and decoded floating-point view | Stage 2 CPU/RAM or accelerator memory |
| `Q` | The current quantized module and its output tensors | Stage 2 CPU/RAM or accelerator memory |
| `S` | The largest source safetensors shard | Disk capacity and sequential I/O, not a RAM/VRAM reservation |
| `H` | Calibration statistics for the current target | Stage 1 CPU/RAM; small for iMatrix, potentially large for GPTQ |

Practical lower bounds are therefore:

```text
Stage 1 RAM/VRAM:  T + A + runtime temporary tensors + H
Stage 2 CPU:      RAM ~= W + Q + temporary tensors; VRAM ~= 0
Stage 2 CUDA:     RAM ~= small I/O buffers; VRAM ~= W + Q + temporary tensors
```

These are lower bounds, not allocator-independent guarantees. PyTorch,
safetensors, quantization kernels, and Python bookkeeping add headroom. Use at
least 20-30% additional capacity, and more for GPTQ or models with large custom
buffers. Stage 2 reads one source tensor and its materializer dependencies,
quantizes it if applicable, writes a temporary tensor file, and releases it
before reading the next one. It then assembles the final safetensors shard by
copying those files in bounded chunks. A large `S` increases runtime and disk
pressure, but is not loaded into RAM or VRAM as a whole.

Before a run, inspect both the total checkpoint size and the largest shard:

```bash
du -sh /path/to/model
du -h /path/to/model/*.safetensors | sort -h | tail -1
```

For a BF16 Qwen3-0.6B checkpoint of about 1.4 GiB, Stage 2 does not reserve the
full checkpoint or shard. Size RAM/VRAM from the largest sequential target,
largest Linear weight plus its quantized output, and one boundary batch. The
exact peak still depends on sequence length, batch size, and allocator
fragmentation. For DeepSeek-V4 raw FP8+FP4, the materializer also creates a
transient BF16 view of each decoded target, so the largest target and shard
deserve extra headroom.

GPTQ changes the estimate materially. Its Hessian is quadratic in a Linear's
input width: a float32 Hessian alone costs approximately
`4 * in_features * in_features` bytes per active module, in addition to `T`,
`A`, and the model's other temporary tensors. iMatrix stores only per-input-
channel sums and counts and is usually much smaller. For very wide projections,
prefer iMatrix+RTN unless the available RAM has been sized for the GPTQ
Hessians.

Disk capacity must cover all of the following at the same time:

```text
source checkpoint + calibration boundary artifacts + statistics
                 + staging shards + finalized output checkpoint
```

The source checkpoint is never modified. Stage 1 stores boundary batches and
statistics under `work_dir`; Stage 2 stores complete output shards under its
staging subdirectory; Stage 3 copies those shards to `output_dir`. Keep the
work and output directories on a filesystem with sufficient free space, and do
not place either directory inside the source checkpoint. A resume run reuses
completed artifacts only when the source, recipe, dataset, materializer, and
calibration settings match.

To reduce peak usage, start with `batch_size=1`, reduce `max_seq_length`, and use
iMatrix+RTN instead of GPTQ. Batch size and sequence length reduce `A`; fewer
calibration samples primarily reduce runtime and boundary-artifact disk usage,
not target-weight residency. Setting `device="cpu"` avoids accelerator memory
pressure but moves the Stage 1 and Stage 2 working sets into system RAM.

The three-stage logger reports `rss_peak_mib`, `gpu_peak_mib`, and the current
artifact directory size after each stage. These are process high-water/resource
figures rather than a reservation guarantee; compare them with the formulas
above and leave allocator headroom. The `rss_peak_mib` unit is platform
dependent, so treat it as a diagnostic and use the operating system's memory
monitor for final capacity planning.
