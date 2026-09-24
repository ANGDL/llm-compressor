# Model Quality CI

This directory is the isolated control plane for production model compression
and quality jobs. It does not change library entrypoints or existing test
pipelines. The design document is
`docs/developer/model-quality-ci-plan.md`.

中文部署与逐步操作说明：[使用指南](../../docs/developer/model-quality-ci-usage.md)。

## Current delivery

The first delivery implements configuration validation, stable fingerprints,
P0-P3 ordering, GPU-hour admission, `DEFERRED_BUDGET`, Buildkite dynamic
pipeline rendering, preflight, a safe argv quantization runner, checkpoint
validation, real vLLM loading/generation, optional evaluator commands,
JSON/Markdown reports, and a guarded `bcecmd` publish stage.

The checked-in example model is disabled so a checkout cannot accidentally run
an expensive model job. Its source path, CI entrypoint, vLLM Python, resource
estimate, and BCE target must be reviewed before enabling it.

## Local dry run

```bash
python3 -m ci.model_quality.plan \
  --config ci/model_quality/config/models.yaml \
  --git-sha local \
  --max-gpu-hours 8 \
  --plan-output /tmp/model-quality-plan.json \
  --pipeline-output /tmp/model-quality-pipeline.yml
```

Enable and edit a model only after its source path, command, GPU-hour estimate,
and required capabilities have been verified on the target host.
Large model definitions can live in separate files and be added through the
root manifest's `includes` list. The checked-in
`config/models/deepseek_v4_wna8.example.yaml` shows every argument supported by
the current DeepSeek streaming entrypoint, including multiple dataset values and
BooleanOptionalAction flags.

Commands are argv arrays: one list element is one exact process argument. Paths
use controlled placeholders such as `{source_path}`, `{output_dir}`, and
`{work_dir}`. Omit a boolean flag to use the script default, add `--flag` for
true, and add `--no-flag` for false. Multi-value `nargs="+"` arguments place
all values after the flag and before the next option. Model-specific invalid
combinations belong in `forbidden_argument_pairs`; DeepSeek recovery mode, for
example, forbids `--checkpoint-progress` with `--async-save`.

The resolver replaces only CI-owned placeholder tokens. JSON objects, regular
expressions, and unknown braces are preserved verbatim, so arguments such as
`{"name":"exact_match"}` and `a{2,4}` do not need escaping. Quantization,
evaluation, result paths, and BCE upload commands all use this resolver.
`workflow.revision` must pin the quantization entrypoint commit or image digest;
it is part of artifact identity even though the ambient checkout SHA is not.

## Run modes

- `quantize`: quantize, validate, run vLLM smoke, and report.
- `quantize_and_eval`: add the configured evaluator command.
- `eval_only`: reuse `--run-id`, validate the existing artifact, smoke it, and evaluate.
- `upload_only`: reuse `--run-id` and publish its already-passing report.

`eval_only` and `upload_only` require an explicit run ID. Evaluation is skipped
in normal `quantize` mode. When evaluation is explicitly requested but no
evaluator command is configured, the job fails with `CONFIG_ERROR`. Upload is
not scheduled unless it is enabled, except for an explicit `upload_only` request.

`ci.model_quality.evaluators.lm_eval_pair` is the first causal-LM adapter. It
runs the base and compressed models in separate subprocesses, persists selected
lm-eval metrics, and feeds the generic comparator. The comparator handles both
higher-is-better task scores and lower-is-better values such as perplexity;
quality gates always use unrounded values.
Each selected job carries separate artifact and evaluation fingerprints plus a
unique attempt ID. Stage records and attempt-specific logs are filtered by these
identities. A changed evaluator configuration invalidates only evaluation
results; changing the checkout Git SHA alone no longer invalidates an existing
compressed artifact. Evaluators must always produce a configured structured
`result_file`; exit status zero without one is a configuration error.
`evaluation.runtime_revision` and `runtime_smoke.runtime_revision` must pin the
corresponding image digest or lockfile/runtime revision.

Raw evaluator output is attempt-local. Any pre-existing raw result is removed
before the evaluator starts, so a successful command that writes nothing cannot
relabel stale data. Normalized results are cached only by matching artifact and
evaluation fingerprints. Reports are built under the attempt directory and
atomically promoted to the shared `reports/` view together with
`current-attempt.json`; publish verifies that promoted view before upload.
Failed preflight attempts never create the immutable input manifest. Successful
preflight binds the request identity to observed source checkpoint provenance;
later stages load that finalized artifact identity from the manifest.

## Buildkite host setup

The target host needs a Buildkite queue named `model-quality-gpu`, `python3`
with this project installed, a separate vLLM Python selected by
`VLLM_PYTHON_ENV` or `runtime_smoke.python`, local model/dataset storage, and
optionally `bcecmd`. The global concurrency group keeps the current delivery
single-build/single-model on the eight-GPU host.

Buildkite must set `MODEL_QUALITY_RUNS_ROOT` to an absolute directory shared by
all steps on that host, for example `/data/model-quality/runs`. Relative paths
are rejected in Buildkite. Input manifests are immutable: reusing a run ID with
a different planner fingerprint fails before quantization.
The bootstrap also persists `execution-plan.json` under that shared run root so
the aggregate report retains GPU-hour budget and deferred jobs.
The persisted plan contains immutable compression inputs and scheduling
decisions, but strips upload commands, evaluator commands, runtime Python paths,
and business-only policy fields. Secrets are never allowed in the manifest.

Keep `upload.enabled: false` until the installed `bcecmd` version is checked.
When enabling it, provide explicit argv arrays for both `commands` and
`verify_commands`, plus `success_commands` for the final `SUCCESS.json`; only
`bcecmd` is accepted. Supported placeholders include
`{output_dir}`, `{reports_dir}`, `{remote_run_prefix}`, `{model_id}`, and
`{run_id}`. `upload.allowlist` limits publication to explicit run-relative paths
such as `model` and `reports`; the publisher generates an SHA256 upload manifest
before calling `bcecmd`, excludes its own manifest and commit marker from that
frozen set, runs remote verification, creates `SUCCESS.json` only after verify,
and uploads it last. Upload commands may use only `{output_dir}` and
`{reports_dir}` as local roots; run, source, and work directories are rejected.
Secrets must come from the Buildkite environment, never from YAML.

## Implementation status

Implemented now:

- manifest validation, fingerprints, P0-P3 ordering, and GPU-hour admission;
- dynamic pipelines for quantize, quantize-and-evaluate, evaluate-only, and upload-only;
- safe argv execution, timeout/termination forwarding, atomic state files, and logs;
- checkpoint index/shard validation, finite scans, and SHA256 manifests;
- real causal-LM vLLM load/generation smoke;
- generic higher/lower metric gates and paired lm-eval subprocesses;
- failure-tolerant reports, aggregate reports, and guarded bcecmd command adapters.

Target-host validation still required:

- replace the disabled example paths with the first two real model definitions;
- verify the Buildkite queue, GPU visibility, shared storage, and vLLM environment;
- pin the lm-eval task/metric names and thresholds on the target runtime;
- verify the installed bcecmd syntax before supplying upload/verify argv arrays;
- run an actual quantization, vLLM smoke, evaluation, and remote upload.
