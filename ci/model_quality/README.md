# Model Quality CI

This directory is the isolated control plane for production model compression
and quality jobs. It does not change library entrypoints or existing test
pipelines. The design documents are
`docs/developer/model-quality-ci-plan.md` and
`docs/developer/model-quality-ci-web-design.md`.

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

## Web control plane

The web layer reads the same run directory used by the CI stages, so facts stay
in the run JSON and the API never becomes a second source of truth. It is a
WSGI application with no third-party framework and no shell execution inside a
request:

```bash
MODEL_QUALITY_RUNS_ROOT=/data/model-quality/runs \
python -m ci.model_quality.web --host 127.0.0.1 --port 8000 \
  --config ci/model_quality/config/models.yaml
```

Open `http://127.0.0.1:8000/` for the browser UI. The server hosts the static
HTML/CSS/JavaScript and same-origin API together; a developer working remotely
can keep the server bound to loopback and forward it locally with
`ssh -L 8000:127.0.0.1:8000 <host>`. The Runs, Run Detail, Plan Preview, log,
capacity, trend, and schedule views are available without a frontend build
tool. The Identity dialog is a development aid; production must have a trusted
OIDC/reverse proxy inject actor and role headers rather than accepting browser
claims directly.

### Remote container lifecycle

Use the checked-in manager instead of manually coordinating a container name,
SSH control socket, service process, and local port forward:

```bash
scripts/model-quality-web-remote init
$EDITOR .model-quality/web-remote.env
scripts/model-quality-web-remote doctor
scripts/model-quality-web-remote open
```

`open` is convergent: it recreates a missing SSH control socket, verifies the
configured container, auto-discovers Python, starts or repairs the Web service,
repairs the local tunnel, performs `/api/health`, and opens the browser. Run it
again after a laptop restart or a dropped SSH connection. When a container,
host, repository path, runs root, port, or service revision changes, edit only
`.model-quality/web-remote.env` and run `restart`:

```bash
scripts/model-quality-web-remote restart
scripts/model-quality-web-remote status
scripts/model-quality-web-remote logs
```

By default `down` removes only the local tunnel, keeping the remote dashboard
available to other clients. Use `MQ_STOP_SERVICE=1
scripts/model-quality-web-remote down` to stop the managed container process as
well. The service remains bound to container loopback, so it is never exposed
directly on the node. An optional systemd unit template is available at
`ci/model_quality/web/deploy/model-quality-web.service` for hosts where the
service must survive all client sessions and host reboots.

Read-only surface (Phase 1): `GET /api/health`, `GET /api/whoami`,
`GET /api/models`, `GET /api/capabilities`, paginated `GET /api/runs`, run and
model detail (attempt history, stage records, fingerprints, artifacts, publish,
backup), stage-filtered logs, `GET /api/runs/{run_id}/artifacts/{id}/access`
for short-lived download metadata, `GET /api/audit`, `GET /api/trends/quality`,
`GET /api/trends/cost`, `GET /api/ops/schedules`, and
`GET /api/ops/notifications`. `GET /api/runs/{run_id}/events` returns an
ordered SSE snapshot; pass `after=<sequence>` when reconnecting to request only
newer events. Malformed optional files are reported as missing so one partial
run does not hide the others.

Controlled execution (Phase 2): `POST /api/plans/preview`, `POST /api/runs`,
`POST /api/runs/{run_id}/retry`, `POST /api/runs/{run_id}/evaluate`, and
`POST /api/runs/{run_id}/cancel`. Every write carries an `idempotency_key`,
records an audit entry, and only ever stores an allowlisted
`ci.model_quality.stage` argv. Quantization commands are imported through
`shell lexer → allowlist → structured argv`; shell operators, substitutions,
unknown flags, and out-of-policy paths are rejected. Inference configuration
accepts a prestarted container name and script only, never a server, port, URL,
or shell string.

The queue worker is the only component that spawns processes:

```bash
MODEL_QUALITY_RUNS_ROOT=/data/model-quality/runs \
python -m ci.model_quality.web.worker --config ci/model_quality/config/models.yaml
```

It drains queued jobs in creation order, which is the run-mode stage order
(preflight/quantize/validate → runtime-smoke → evaluate → report → publish).
One GPU-hour reservation covers a model/attempt; it stays held until every lane
is terminal, moves `RESERVED → QUEUED → RELEASED`, and is released immediately
when a stage fails and cancels its siblings. A retryable job retry re-admits its
GPU-hours before it is queued, so retrying never runs for free and never
resurrects a released reservation. Failures are classified as
`RESOURCE_TIMEOUT`, `RESOURCE_UNAVAILABLE`, `SCRIPT_FAILED`, or
`QUALITY_GATE_FAILED` so retryable infrastructure problems are never confused
with a rejected artifact.

Publish and backup (Phase 3): `POST /api/runs/{run_id}/publish/preview`,
`POST /api/runs/{run_id}/publish` (requires an explicit `confirm: true`, plus a
second approver when `MODEL_QUALITY_PUBLISH_DUAL_APPROVAL=1`),
`POST /api/runs/{run_id}/backup/preview`, `POST /api/runs/{run_id}/backup`,
`POST /api/backups/{backup_id}/restore/preview`, and
`POST /api/backups/{backup_id}/restore`. Preview shows the allowlist, file
count, total bytes, checksums, remote target, and risks. Backup manifests are
immutable; restore always creates a new run id, keeps the source backup id, and
never overwrites an existing run or marks an artifact as current.

Operations (Phase 4): cron schedules (`GET/POST /api/ops/schedules`,
`PATCH/DELETE /api/ops/schedules/{id}`) start one run per occurrence and replay
the recorded run when the same occurrence is processed again, a webhook
notification service that stays disabled unless
`MODEL_QUALITY_NOTIFY_ALLOW_NETWORK=1`, quality and GPU-hour cost trends, and a
conservative queue capacity forecast inside `GET /api/capabilities` (drain
estimate derived from GPU-hours actually released in the recent window,
reported as `confidence: none` when there is no history). A cron unit or
systemd timer drives the schedules with a single tick:

```bash
MODEL_QUALITY_RUNS_ROOT=/data/model-quality/runs \
MODEL_QUALITY_CONFIG=ci/model_quality/config/models.yaml \
python -m ci.model_quality.web.scheduler
```

The tick prints one JSON result per due schedule (`STARTED`, `REPLAYED`, or
`FAILED`) so the timer log records exactly what happened.

Roles come from trusted headers (`X-Model-Quality-Actor`,
`X-Model-Quality-Roles`) or a bearer token (`MODEL_QUALITY_WEB_TOKEN`):
`viewer` (read), `operator` (plan/start/retry/cancel/evaluate), `publisher`
(publish/backup), `admin` (everything, including restore). Related environment
variables: `MODEL_QUALITY_GPU_HOUR_CAPACITY`,
`MODEL_QUALITY_INFERENCE_CONTAINERS`, `MODEL_QUALITY_INFERENCE_SCRIPT_ROOTS`,
`MODEL_QUALITY_OUTPUT_ROOTS`, `MODEL_QUALITY_PUBLISH_DUAL_APPROVAL`,
`MODEL_QUALITY_WEB_EXECUTOR`, `MODEL_QUALITY_BACKUP_*`, and
`MODEL_QUALITY_WEB_STAGE_CLI`.

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
- the read-only WSGI observer, plan preview and controlled start/retry/evaluate/cancel;
- the allowlisted queue worker with per-model GPU-hour reservations and failure classes;
- publish preview/confirm with optional dual approval, plus immutable backups and restores;
- cron schedules, notification log, quality/cost trends, and the capacity forecast.

Target-host validation still required:

- replace the disabled example paths with the first two real model definitions;
- verify the Buildkite queue, GPU visibility, shared storage, and vLLM environment;
- pin the lm-eval task/metric names and thresholds on the target runtime;
- verify the installed bcecmd syntax before supplying upload/verify argv arrays;
- run an actual quantization, vLLM smoke, evaluation, and remote upload.
