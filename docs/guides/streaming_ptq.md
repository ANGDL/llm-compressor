# Streaming PTQ

Streaming PTQ is an experimental, out-of-core alternative to `oneshot()` for
dense models whose full uncompressed weights do not fit in RAM or accelerator
memory. It does not change the existing `oneshot()` path.

The workflow has three durable stages:

1. `collect_calibration_statistics()` loads one sequential target at a time,
   stores GPTQ/iMatrix sufficient statistics, and releases its weights.
2. `quantize_streaming()` rereads source shards, restores those statistics,
   compresses quantized Linear weights, and atomically writes staging shards.
3. `finalize_streaming_checkpoint()` validates every staging shard, creates the
   safetensors index and compressed-tensors config, copies auxiliary files, and
   atomically publishes the output directory.

`streaming_oneshot()` runs all three stages. Completed statistics and staging
shards are reused when their fingerprints match. The public Python API requires
a model factory that constructs the checkpoint architecture without loading its
weights, ordered target names, calibration batches, and exact module-to-scheme
mappings. See the function docstrings for the complete arguments.

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
dense Linear targets; GPTQ and iMatrix; local safetensors checkpoints. AWQ,
SmoothQuant, AutoSmooth, AutoRound, pruning, MoE/fused checkpoint layouts, and
multi-GPU execution are rejected or outside this initial implementation. Custom
source formats should supply a `WeightMaterializer` that converts the requested
weight and its declared dependencies to the selected floating computation dtype.

Do not load a staging directory as a model. It intentionally has no index or
config. Only an output directory containing `FINALIZED` is a published result.
