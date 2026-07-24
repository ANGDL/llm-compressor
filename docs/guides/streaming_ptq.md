# Streaming PTQ

Streaming PTQ is an experimental, out-of-core alternative to `oneshot()` for
models whose full uncompressed weights do not fit in RAM or accelerator memory.
It does not change the existing `oneshot()` path. The normal pretrained entry
point currently supports local Transformers checkpoints with dense `nn.Linear`
quantization targets. DeepSeek-V4 raw FP8+FP4 checkpoints are supported when a
`DeepSeekV4WeightMaterializer` is supplied; the materializer decodes each
requested target to BF16 on demand.

The normal pretrained workflow is a per-subgraph closed loop:

1. Load only the current traced subgraph working set.
2. Run calibration through the existing Modifier hooks.
3. Invoke `sequential_epoch_end`, then propagate the quantized output to the
   next subgraph with hooks disabled.
4. Compress and write the subgraph tensors directly as a final safetensors
   shard under `work_dir/publish`.
5. Release weights and statistics, then repeat for the next subgraph.

This preserves the ordering and error-propagation behavior of the existing
Sequential Pipeline instead of maintaining separate GPTQ/iMatrix algorithms. A
normal run keeps adjacent activation boundaries in CPU memory and writes each
completed subgraph once as a final shard. Finalization adds the index, config,
and auxiliary files, then renames `work_dir/publish` to `output_dir`. Set
`checkpoint_progress=True` to persist every completed subgraph and
its next boundary for crash recovery; this mode trades additional I/O and disk
space for resumability. The public Python API accepts a local checkpoint,
calibration data, and a normal iMatrix + RTN/GPTQ recipe. It derives decoder
targets, exact module schemes, and layer-zero boundaries without loading the full
model. See
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

The currently verified compatibility set is local safetensors checkpoints,
single-device execution, and dense `nn.Linear` targets using iMatrix+RTN, GPTQ,
SmoothQuant, AWQ, SparseGPT, or Wanda. The normal pretrained path uses
quantized propagation, matching the default Sequential Pipeline behavior. Tiny
Qwen3 tests produce tensor-identical checkpoints for iMatrix+RTN and GPTQ.
SmoothQuant, SparseGPT, and Wanda produce numerically close checkpoints and
reloaded logits. AWQ produces the same checkpoint schema and close reloaded
logits; tiny floating-point differences can select a different grid-search
point, so packed tensors are not required to be identical.

This does not imply automatic support for every Sequential Pipeline modifier. A
modifier is safe only when all modules it reads or changes fit in the current
traced subgraph working set. Cross-subgraph mappings need an explicit working-set
declaration before they can be streamed without changing algorithm semantics.
AutoSmooth has not yet been verified. AutoRound is currently rejected because
it owns both quantization and checkpoint compression; preserving its output
format requires a dedicated streaming output adapter. MoE execution is supported
only when its target graph and working set can be traced; DeepSeek-V4 uses its
dedicated materializer for raw FP8+FP4 layout. Custom source formats should
supply a `WeightMaterializer` that converts each requested weight and declared
dependencies to the selected floating computation dtype.

Do not load `work_dir/publish` while quantization is running. Its shards already
contain final tensor bytes, but the directory has no complete index or config
until finalization. Only an output directory containing `FINALIZED` is a
published result.

## Peak memory and disk requirements

Streaming reduces the model-weight residency of calibration, but it does not
make every operation independent of checkpoint shard size. Plan capacity for
the largest of the following quantities:

| Symbol | Meaning | Where it matters |
| --- | --- | --- |
| `T` | Decoded parameters and persistent buffers in the largest sequential target | Current subgraph CPU/RAM or accelerator memory |
| `A` | One calibration boundary batch, including tensors such as hidden states, masks, and position information | Boundary store and current forward |
| `Q` | Current subgraph compressed output tensors | Final shard writing |
| `S` | The largest source safetensors shard | Disk capacity and sequential I/O, not a RAM/VRAM reservation |
| `H` | Calibration statistics for the current target | Current subgraph; small for iMatrix, potentially large for GPTQ |

Practical lower bounds are therefore:

```text
CPU execution:   RAM ~= T + A + H + Q + runtime temporary tensors
CUDA execution: VRAM ~= T + A + H + runtime temporary tensors
                RAM  ~= boundary storage + safetensors I/O buffers + Q
```

These are lower bounds, not allocator-independent guarantees. PyTorch,
safetensors, quantization kernels, and Python bookkeeping add headroom. Use at
least 20-30% additional capacity, and more for GPTQ or models with large custom
buffers. The writer consumes one output tensor at a time and assembles the final
safetensors shard directly, without raw transaction payloads or later shard
assembly. A large `S`
increases runtime and disk pressure, but is not loaded into RAM or VRAM as a
whole.

Before a run, inspect both the total checkpoint size and the largest shard:

```bash
du -sh /path/to/model
du -h /path/to/model/*.safetensors | sort -h | tail -1
```

For the local BF16 Qwen3-0.6B checkpoint of about 1.4 GiB, a CPU iMatrix+RTN
smoke run with one length-8 sample measured an 847 MiB process RSS high-water
mark. Importing the same framework stack without running quantization measured
about 435 MiB, so the observed incremental peak was about 412 MiB. This confirms
that the full 1.4 GiB source weight file was not resident. These numbers are one
macOS run, not a capacity guarantee; sequence length, batch size, allocator,
PyTorch version, and model implementation change the result. For DeepSeek-V4
raw FP8+FP4, the materializer also creates a transient BF16 view of each decoded
target, so the largest target deserves extra headroom.

The same resource assertion can be repeated in an isolated process for any local
safetensors checkpoint. The test requires an explicit path because it performs a
real end-to-end quantization and is not part of the fast unit-test suite:

```bash
LLMCOMPRESSOR_STREAMING_RESOURCE_CHECKPOINT=/path/to/model \
  .venv/bin/python -m pytest -q -m integration \
  tests/llmcompressor/streaming/test_resource_regression.py
```

The regression subtracts the framework-import RSS baseline and requires the
incremental peak to remain below the source checkpoint size. It also requires a
finalized output checkpoint.

GPTQ changes the estimate materially. Its Hessian is quadratic in a Linear's
input width: a float32 Hessian alone costs approximately
`4 * in_features * in_features` bytes per active module, in addition to `T`,
`A`, and the model's other temporary tensors. iMatrix stores only per-input-
channel sums and counts and is usually much smaller. For very wide projections,
prefer iMatrix+RTN unless the available RAM has been sized for the GPTQ
Hessians.

Disk capacity must cover all of the following at the same time:

```text
source checkpoint + in-progress final output shards + boundary artifacts
```

The source checkpoint is never modified. The default path writes final shards
under `work_dir/publish`; final publication performs a directory rename rather
than copying or reassembling weight bytes. Therefore `work_dir` and
`output_dir` must be on the same filesystem. With `checkpoint_progress=True`,
the recovery path also stores committed transactions and adjacent boundaries.
Pass `overwrite_output=True` to atomically replace an existing output directory;
without it, a completed output is reused and any other non-empty output is
rejected. Do not place work or output inside the source checkpoint.

To reduce peak usage, start with `batch_size=1`, reduce `max_seq_length`, and use
iMatrix+RTN instead of GPTQ. Batch size and sequence length reduce `A`; fewer
calibration samples primarily reduce runtime and boundary-artifact disk usage,
not target-weight residency. Setting `device="cpu"` avoids accelerator memory
pressure but moves the subgraph working set into system RAM.

The advanced three-stage boundary API still reports `rss_peak_mib`,
`gpu_peak_mib`, and artifact size after each stage. These are process high-water
figures rather than a reservation guarantee. The `rss_peak_mib` unit is
platform dependent, so use the operating system's memory monitor for final
capacity planning.
