# Streaming reference inference

`load_streaming_model()` runs a local checkpoint through its native Transformers
forward and generation implementation. Parameters start on meta. Accelerate loads
the current module's weights from a lazy checkpoint Mapping and releases them
after the call. The Mapping retains metadata and counters, never decoded weights.

This path is intended to check model adaptation and quantized checkpoint
serialization with limited accelerator memory. Each decode token rereads the
required weights, so it is not a serving engine.

The pipeline is generic across architectures: it owns checkpoint access, weight
offload, memory budgets and validation entry points. Model adapters own attention,
precision boundaries and request cache semantics, following their official
implementations. Architecture-specific computation belongs in the model adapter;
an algebraically equivalent replacement is insufficient when it changes
low-precision rounding. Shared model helpers are appropriate where the official
models use the same algorithm.

## Native prefill, decode and generation

```python
import torch
from llmcompressor.streaming import load_streaming_model

with load_streaming_model(
    "/path/to/local/checkpoint",
    device="cuda:0",
    dtype=torch.bfloat16,
) as inference:
    prompt = torch.tensor([[1, 3, 5, 7]])
    output = inference(prompt, use_cache=True)
    output = inference(
        torch.tensor([[9]]),
        past_key_values=output.past_key_values,
        use_cache=True,
    )
    sequences = inference.generate(
        prompt, use_cache=True, do_sample=False, max_new_tokens=16,
    )
    report = inference.report()
```

Inputs are moved to the execution device by the thin wrapper. `inference.model`
exposes the native model for inspection and diagnostic hooks. Run computation
through `inference(...)`, `generate(...)`, or `call("native_method", ...)` to retain
inference mode and failure cleanup.

Standard Llama/Qwen3 models use their original Cache implementations. On a suitable
accelerator, Transformers KV offload can be requested with
`generate(cache_implementation="offloaded", ...)`. Weight offload and KV offload
are separate. The CUDA KV transfer path requires GPU validation.

If a forward fails after modifying a supplied cache, the wrapper marks that cache
invalid and releases installed weights. Discard the cache and prefill a fresh
request. Calls on one inference instance are serialized. Closing the context
removes hooks without loading the checkpoint into CPU memory.

## Quantized weights

The initial implementation supports compressed-tensors checkpoints with:

| Weight scheme | Storage |
| --- | --- |
| Symmetric channel/tensor INT8 | `int-quantized`, signed int8 |
| Symmetric channel/tensor INT4 | `int-quantized`, unpacked signed int8 |
| Symmetric channel/tensor INT4 | `int-quantized`, adjacent signed nibbles in int8 |
| Symmetric channel/tensor INT4/INT8 | `pack-quantized`, standard int32 packing |

Each module resolves its own scheme, allowing mixed W8A8/W4A8. Integer weights
are decoded on CPU **when the current tensor is requested**, then cast to the
model's execution dtype, BF16 by default. No full BF16 state dict, whole-model CT
decompression hook or new quantized matrix multiplication kernel is installed.
Group/asymmetric weight quantization, static activation schemes, runtime
transforms and unknown formats currently fail explicitly.

Dynamic token INT8 input/output activation Q/DQ remains enabled. The actual
bound model forward is preserved, including custom accumulation behavior.
Already decoded weights are not quantized a second time. This is numerical
reference execution; it does not claim native W8A8/W4A8 kernel performance.

Token Q/DQ normalizes a flattened MoE input `[rows, features]` to
`[1, rows, features]` before calling the CT primitive. This keeps each row's
scale independent of the other routed tokens. The installed CT TOKEN helper
otherwise uses whole-tensor min/max for a two-dimensional input. This correction
is confined to inference and the resident validation loader.

For a weight-only diagnostic, pass `activation_qdq=False`. The report labels this
`dequantized_weights`; the default is `quantized_reference`.

## Working sets and budgets

The default Accelerate hooks work at module granularity. For a dense model, pass
`preload_module_classes=["LlamaDecoderLayer"]` to load one complete decoder at
a time. For a large MoE, retain leaf hooks or group only an `Expert`; preloading
an entire MoE block may load all experts.

A parent module's direct control parameters can remain live while its child runs.
The header-only working-set estimate includes such ancestor weights.
`max_device_weight_bytes` rejects an oversized weight group before payload reads.
It excludes activations, derived buffers, KV cache, and operator workspaces;
leave room for these when selecting a weight budget.

`max_preparation_bytes` limits the estimated CPU preparation footprint of one
tensor, including packed inputs, decode workspace and the execution value.
The existing `HostMemoryBudget` also checks system availability before reading.
There is no weight prefetch or decoded CPU cache in this initial implementation.

Nonpersistent derived buffers are initially kept on CPU, moved for their owning
module's call, and returned to CPU on success or failure. This does not replace a
model's request-cache contract. Models with hidden parameter access, fused
experts or mutable buffer aliases need their own compatibility validation and
possibly a minimum safe `preload_module_classes` group.

An adapter may declare immutable `_streaming_cpu_buffers` and transfer only
required slices inside its forward. V4.1 uses this for its shared RoPE table,
avoiding full-context table transfers or a separate CPU copy for every layer.

## Raw DeepSeek V4.1

Use the existing source materializer and native registration:

```python
from transformers import AutoTokenizer
from llmcompressor.streaming import (
    DeepSeekV41WeightMaterializer,
    load_streaming_model,
)

checkpoint = "/path/to/DeepSeek-V4.1-Flash"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
with load_streaming_model(
    checkpoint,
    device="cuda:0",
    materializer=DeepSeekV41WeightMaterializer(),
    model_kwargs={"tokenizer": tokenizer, "engram_checkpoint": checkpoint},
) as inference:
    tokens = tokenizer("Hello", return_tensors="pt")["input_ids"]
    sequences = inference.generate(tokens, use_cache=True, max_new_tokens=8)
```

The same bridge handles this project's WNa8/W8A8/W4A8 raw-layout outputs, including
their `.scale` names and signed nibble INT4 packing. Engram tables retain bounded
row lookup rather than becoming full device parameters.

The model declares the official FP32 compressor projections (ratio > 1), LM and
Markov heads, confidence projection and mHC controls explicitly. The loader
respects those declarations while ordinary computation defaults to BF16.
The PyTorch sparse attention path follows the official 64-KV online softmax,
with BF16 weights before each value GEMM and FP32 accumulation/rescaling.
Fully masked padding rows retain zero output with finite calibration gradients.
This follows the model's numerical contract without introducing model-specific
rules into the inference pipeline.

Native V4.1 now has request-owned CPU state for window history, source-owned
compressed KV/index K, unfinished compression groups and Engram history.
It supports unpadded batch=1 cached prefill, single-token decode and append-only
prefill chunks. `DeepseekV41Cache` is returned through `past_key_values`; HF's
generation loop remains responsible for token selection and stopping.
Cached backbone inference skips the teacher-forced DSpark calibration branch.
Beam search, speculative cache rollback and padded cached batches are rejected.
Image features can be prepared with `inference.call("encode_image", ...)`;
generation with prepared image inputs requires matching `input_ids` for Engram.

Sparse score ties now choose smaller positions consistently, so adding future
masked columns cannot change historical selections. This is a deliberate tie
rule for causal cache consistency; arbitrary tied selections in a target kernel
can differ. Stable reference sorting also has a performance cost.

The older **native V4** adapter's buffer/cache integration is not included in
this implementation. Availability of a registered model class alone is not a
support claim. Its pending cache work must not be inferred from V4.1 support.

## Fixed-token validation and CLI

```python
from llmcompressor.streaming import compare_cached_to_recompute

with load_streaming_model("/path/to/checkpoint") as inference:
    comparison = compare_cached_to_recompute(
        inference,
        torch.tensor([[1, 3, 5]]),
        torch.tensor([[7, 9, 11]]),
    )
    assert comparison["passed"], comparison
```

This compares the same checkpoint's cached path with full-prefix recomputation
on identical tokens, reporting max/RMS error, relative L2, top-1 agreement and
cache length. `passed` requires the requested elementwise tolerances, finite
logits and correct cache lengths. It does not independently prove the checkpoint
or model math.
This is a same-checkpoint consistency check, not quantization error against
source weights. Compare each PTQ checkpoint to the same source execution to
rank quantization quality; cache/recompute differences need not rank by bit width.
For V4.1 comparisons, disable `inference.model.config.calibrate_dspark` so the
full-prefix reference also runs only the backbone.

```shell
python examples/streaming_inference/generate.py /path/to/checkpoint \
    --device cuda:0 --prompt "Hello" --max-new-tokens 8

python examples/streaming_inference/generate.py /path/to/checkpoint \
    --input-ids 1,3,5 --continuation-ids 7,9,11 --report /tmp/inference.json
```

Add `--layout deepseek-v41` for raw V4.1 or the corresponding PTQ outputs.
`--rtol` and `--atol` explicitly control the comparison threshold.

## Full-topology tiny validation

```shell
python tools/validate_tiny_dsv41_inference.py \
    .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny
```

This reads existing artifacts without retraining or requantizing. It compares
streamed and independently loaded resident weights for source FP8/FP4,
WNa8, W8A8 and W4A8. The fixture has 40 text, 32 vision and 3 DSpark blocks.
The CPU run found exact streamed/resident agreement for prefill, three decode
steps, greedy generation, vision features and teacher-forced DSpark taps.
Hook initialization read no weight payloads and completed calls left parameters
on meta.

Cache/recompute comparisons are recorded separately. On this fixture, the
source and WNa8 BF16 runs meet the default tolerances. W8A8 and W4A8 contain
strict elementwise failures, with maximum absolute logit differences about
`0.0139` and `0.0306`, respectively. Top-1 agrees at every tested step;
relative L2 remains below `0.0023`. The report includes an additional BF16
epsilon-based precision budget, but it does not replace the strict result.
FP32 parameters reduce the differences, yet the W4A8 reference still has a
strict failure of about `0.00330`. Native attention retains low-precision
intermediate Q/DQ even with FP32 parameters. A numerical report is not an
assertion of exact cache/recompute equivalence.

An uncached-only W4A8 probe also changed historical logits by `0.000654`
when extending six tokens to seven. In layer 29, an A8 input difference of
`1.98e-5` crossed a rounding threshold and produced a Q/DQ difference of
`0.01577`. This demonstrates numerical sensitivity outside cache management;
the precise source of the initial floating-point perturbation was not isolated.
The validator records this uncached prefix check as an additional diagnostic.

A follow-up W8A8 investigation isolated the first BF16 divergence to the FFN
mHC merge in zero-based layer 21. Identical FFN output and residual were combined
with slightly different control coefficients: FP32 values `13.46874809` and
`13.46875000` rounded to BF16 `13.4375` and `13.5`. A diagnostic-only fixed-row
mHC projection reduced all three decode logit differences to at most `9.54e-7`.
Default inference math is unchanged. The paired source comparison on eight toy
texts gave MSE `0.001823` for W8A8 and `0.003409` for WNa8, so W8A8 was more
accurate against that common source. See the
[precision investigation](../reports/dsv41_streaming_precision_2026-10-17.md)
and `tools/diagnose_tiny_dsv41_precision.py` for reproduction and scope.

The validator keeps numerical failures visible and exits unsuccessfully when
either the BF16 or FP32 strict cache comparison fails. Its
`streaming_parity_passed` field records the independently checked
streamed/resident parity. Inspect both fields in `streaming-inference.json`.
GPU transfer and original kernel conformance require separate validation.

CPU integration tests cover native Llama/Qwen3, integer layouts, mixed precision,
activation Q/DQ, tied weights, budget rejection, failure cleanup and V4.1 cache
boundaries. GPU transfer, peak memory, production kernels and full-size weights
require validation on a suitable machine.
