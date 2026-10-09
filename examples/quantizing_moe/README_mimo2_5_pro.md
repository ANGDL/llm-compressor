# MiMo-V2.5-Pro iMatrix + RTN Pipeline Notes

This note documents a `weight_scale` NaN issue observed while running
[`mimo2_5_pro_w4a8.py`](mimo2_5_pro_w4a8.py) with `imatrix_mse` and RTN. The
BF16 expert weights were finite, but NaNs appeared in expert `weight_scale`
tensors. The number of NaNs also changed as the calibration dataset size
changed.

The working configuration is:

```python
recipe = [
    IMatrixGatherer(ignore=ignores),
    QuantizationModifier(config_groups=config_groups, ignore=ignores),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="independent",
    ...,
)
```

Do not use expert-level `MiMoV2MLP` targets with the sequential pipeline for
this checkpoint's remote model implementation.

## Symptoms and Controls

The investigation used the following controls:

1. Verify every BF16 expert projection with `torch.isfinite(module.weight)`.
2. Compare W4 and W8 recipes instead of assuming they use the same pipeline.
3. Inspect NaNs by module name; they were concentrated in
   `mlp.experts.<index>.*_proj`.
4. Trace a reduced MiMo model with the repository's real
   `trace_sequential_plan` path.
5. Run the W4 recipe with `pipeline="independent"` while keeping the model,
   dataset, quantization schemes, observer, and modifier unchanged.

The last control removed the NaNs. This isolates the failure to sequential
calibration coverage rather than INT4 arithmetic or the BF16 checkpoint.

## Why `sequential_targets` Did Not Protect the Experts

`sequential_targets` has two separate stages:

1. Module matching finds modules whose class or name matches the configured
   patterns.
2. FX tracing must emit a `call_module` node for each matched module before it
   can become a sequential subgraph boundary.

For MiMo, the first stage succeeds. `MiMoV2MLP` matches every expert MLP. The
second stage fails for those experts.

The checkpoint's `MiMoV2MoE.forward` calls a helper with dynamic sparse routing
and reshapes its result using a starred shape:

```python
orig_shape = hidden_states.shape
topk_indices, topk_weights = self.gate(hidden_states)
hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
```

The `moe` helper loops over all experts but invokes an expert only when routing
selects at least one token:

```python
for expert_idx, expert in enumerate(self.experts):
    token_indices, weight_indices = torch.where(expert_mask[expert_idx])
    if token_indices.numel() > 0:
        expert_output = expert(hidden_states[token_indices])
```

The sequential tracer's auto-wrapper converts operations that FX cannot safely
trace, including starred argument unpacking and tensor-dependent control flow,
into wrapped function calls. For this MiMo forward, the resulting FX graph has
a `call_function` node for the wrapped MoE computation. It does not have
`call_module` nodes for `mlp.experts.<index>`.

A reduced-model trace demonstrated the distinction directly:

```text
matched MiMoV2MLP modules:
  model.layers.0.mlp.experts.0
  model.layers.0.mlp.experts.1
  model.layers.0.mlp.experts.2
  model.layers.0.mlp.experts.3

traced sequential targets:
  model.layers.0.self_attn
```

The target selector therefore worked, but the selected expert modules were not
present in the traced execution graph.

## How Missing FX Nodes Become NaN Scales

The sequential pipeline updates quantization parameters at each subgraph
boundary:

```text
FX call_module nodes
  -> subgraph.submodules(model)
  -> QuantizationModifier.on_sequential_epoch_end(modules)
  -> observe(modules, "weight")
  -> update_qparams(modules, "weight")
```

Expert modules hidden inside the wrapped function are absent from
`subgraph.submodules(model)`. Their weight observers are therefore not run at a
sequential boundary and their `weight_scale` parameters are not updated.

`compressed-tensors` initially allocates static scale holders with
`torch.empty`. An expert scale that is never updated contains unspecified
memory. It may contain NaNs, and changing calibration workload or allocation
order can change the observed NaN count. This is different from the iMatrix
observer computing a NaN scale from finite weights.

Missing iMatrix data alone is not sufficient to produce this failure. When
`update_qparams` is actually called, `imatrix_mse` validates its importance
statistics and falls back to uniform MSE when the count is zero or values are
non-finite. With finite weights, that fallback produces finite qparams.

## Why `pipeline="independent"` Works

The independent pipeline runs each modifier in its own calibration epoch.

1. `IMatrixGatherer` uses its default `attach_by_initialize=True` behavior and
   gathers importance statistics during its pass.
2. `QuantizationModifier` runs in a separate pass.
3. The basic calibration pass used by each modifier sends
   `list(model.modules())` to the boundary callback.
4. Every quantized expert projection is observed and receives updated weight
   qparams, including experts with no usable iMatrix data.

This is also why the working W8 example did not reproduce the W4 failure: it
used the default independent pipeline, while the failing W4 example explicitly
selected `pipeline="sequential"` and deferred iMatrix attachment for that
pipeline.

## MoE Calibration Boundary

`moe_calibrate_all_experts=True` does not automatically cover every custom
remote-code MoE implementation. LLM Compressor's existing all-expert paths
require either:

- a recognized/linearized expert implementation using `LinearExperts2D`; or
- a registered `MoECalibrationModule` adapter for the custom MoE class.

`MiMoV2MoE` satisfies neither condition in the current implementation. The
independent pipeline fixes unwritten scales, but experts that receive no routed
tokens may still use uniform-MSE fallback instead of importance-weighted MSE.
A dedicated `MiMoV2MoE` calibration adapter is the longer-term option if every
expert must see every calibration token.

## Practical Checks

For custom MoE models, use these acceptance checks before saving:

```python
for name, module in model.named_modules():
    if ".mlp.experts." not in name or not hasattr(module, "weight_scale"):
        continue
    assert torch.isfinite(module.weight).all(), name
    assert torch.isfinite(module.weight_scale).all(), name
```

When using the sequential pipeline, treat either of these warnings as a
calibration coverage failure until proven otherwise:

```text
Expected ... subgraphs, but only traced ...
Sequential tracing left ... matched target modules outside all subgraphs ...
```

Choose one of the following:

- use `pipeline="independent"` when memory allows;
- choose a trace-visible parent such as the decoder layer as the sequential
  target, then verify all descendant qparams are updated; or
- add a model-specific MoE calibration/tracing adapter.
