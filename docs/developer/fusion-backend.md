# Fusion KDA Backend

`llm-compressor` exposes the generic backend name `fusion` without depending
on a deployment-specific kernel library. A private deployment package may
install `llmcompressor_fusion_backend` with a `get_provider()` function that
returns an object implementing `FusionKDAProvider` from
`llmcompressor.utils.fusion_backend`.

The provider receives the Transformers KDA tensor contract and returns
`(output, final_state)`. It owns extension loading, device checks, ABI details,
and any vendor-specific implementation.

Select a backend with `LLMCOMPRESSOR_KDA_BACKEND`:

- `auto`: use the default implementation when Triton is allowed; otherwise
  try the optional fusion provider and then use the PyTorch implementation.
- `default`: use the model's default dispatch.
- `fusion`: require the private provider and fail if it is unavailable.
- `fla`: require the FLA Triton implementation.
- `torch`: use the Transformers PyTorch implementation.
- `fla_reference`: use FLA's readable reference implementation for diagnosis.

The legacy value `triton` is accepted as an alias for `fla`. The open-source
repository intentionally contains no deployment library names or vendor ABI
symbols.

## Known Precision Risk

The optional `fusion` provider is not part of the FLA-reference parity claim.
The validated reference path compares FLA's naive recurrent/chunk
implementations with Transformers native KDA. Historical vendor-fusion tests
reported larger differences against the native reference, including a
final-state error around `1e-2`; the new private provider still requires a
same-input output/state comparison before it can be treated as numerically
validated. Keep fusion out of reference-precision conclusions until that test
is complete.
