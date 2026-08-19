# Sequential Onloading #

## Introduction ##

LLM Compressor is capable of compressing models much larger than the amount of memory available as VRAM. This is achieved through a technique called **sequential onloading** whereby only a fraction of the model weights are moved to GPU memory for calibration while the rest of the weights remain offloaded to CPU or disk. When performing calibration, the entire dataset is offloaded to CPU, then onloaded one batch at a time to reduce peak activations memory usage.

![sequential_onloading](../../assets/sequential_onloading.jpg)

If basic calibration/inference is represented with the following pseudo code...
```python
for i in range(len(activations)):
    for layer in model.layers:
        activations[i] = layer(activations[i])
```

Then sequential onloading is the technique by which the order of the two for loops is swapped.
```python
for layer in model.layers:
    for i in range(len(activations)):
        activations[i] = layer(activations[i])
```

## Implementation ##

Before a model can be sequentially onloaded, it must first be broken up into disjoint parts which can be individually onloaded. This is achieved through the [torch.fx.Tracer](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md#tracing) module, which allows a model to be represented as a graph of operations (nodes) and data inputs (edges). Once the model has been traced into a valid graph representation, the graph is cut (partitioned) into disjoint subgraphs, each of which is onloaded individually as a layer. This implementation can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/pipelines/sequential/helpers.py).

![sequential_onloading](../../assets/model_graph.jpg)
*This image depicts some of the operations performed when executing the Llama3.2-Vision model*

![sequential_onloading](../../assets/sequential_decoder_layers.jpg)
*This image depicts the sequential text decoder layers of the Llama3.2-Vision model. Each of the individual decoder layers is onloaded separately*

## Sequential Targets and Usage ##
You can opt into sequential onloading by calling `oneshot` with the
`pipeline="sequential"` argument. The default pipeline is `independent`. If the
sequential pipeline proves to be problematic, use `pipeline="independent"` so
each modifier receives its own calibration pass, or `pipeline="basic"` to share
one non-sequential pass across modifiers. Non-sequential pipelines only work
performantly when the model fits into the available memory.

`pipeline="independent"` isolates modifiers; it does not guarantee that every
modifier avoids sequential tracing. Calibration-dependent modifiers can still
infer the sequential implementation internally, so they require a valid,
trace-visible `sequential_targets` value. Use a parent module such as a decoder
layer when nested calls are hidden by model control flow.

If you are compressing a model using a GPU with a small amount of memory, you may need to change your sequential targets. Sequential targets control how many weights to onload to the GPU at a time. By default, the sequential targets are decoder layers which may include large MoE layers. In these cases, setting the `sequential_targets="Linear"` argument in `oneshot` will result in lower VRAM usage, but a longer runtime.

Sequential target matching does not guarantee that a target becomes an FX
subgraph boundary. A custom model may call matched modules inside
tensor-dependent control flow or another operation that the tracer must wrap as
a function. Wrapped calls do not expose their inner modules as `call_module`
nodes, so those modules will not receive sequential boundary updates. Do not
ignore warnings that matched targets were omitted. Use a trace-visible parent
target or switch to `pipeline="independent"`, then verify that every quantized
module has finite qparams before saving.

![sequential_onloading](../../assets/seq_targets.jpg)

## More information ##

For more information, see the [RedHat AI blog post](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments#generalizing_to_multimodal_and_moe_architectures) or the [LLM Compressor Office Hours Recording](https://www.youtube.com/watch?v=GrhuqQDmBk8).
