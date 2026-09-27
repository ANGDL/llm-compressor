"use strict";

const STAGES = ["preflight", "quantize", "validate", "runtime-smoke", "evaluate", "report", "publish"];
const state = { models: [], runs: [], run: null, jobs: [], plan: null, eventTimer: null };

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const text = (value, fallback = "—") => value === undefined || value === null || value === "" ? fallback : String(value);
const escapeHtml = (value) => text(value, "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const statusClass = (value) => text(value, "pending").toLowerCase().replaceAll(" ", "_");
const badge = (value) => `<span class="status ${statusClass(value)}">${escapeHtml(text(value, "PENDING"))}</span>`;
const short = (value, size = 12) => value ? `${String(value).slice(0, size)}${String(value).length > size ? "…" : ""}` : "—";
const formatDate = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
const formatNumber = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits).replace(/\.0$/, "") : "—";
const key = () => `${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`;

function identity() {
  return {
    actor: localStorage.getItem("mq.actor") || "",
    role: localStorage.getItem("mq.role") || "viewer",
    token: localStorage.getItem("mq.token") || "",
  };
}

function requestHeaders(write = false) {
  const current = identity();
  const headers = { Accept: "application/json" };
  if (write) headers["Content-Type"] = "application/json";
  if (current.actor) {
    headers["X-Model-Quality-Actor"] = current.actor;
    headers["X-Model-Quality-Roles"] = current.role;
  }
  if (current.token) headers.Authorization = `Bearer ${current.token}`;
  return headers;
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...requestHeaders(options.method && options.method !== "GET"), ...(options.headers || {}) } });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(payload.error || payload || `Request failed (${response.status})`);
  return payload;
}

function notify(message, error = false) {
  const element = $("#notice");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.remove("hidden");
  clearTimeout(notify.timeout);
  notify.timeout = setTimeout(() => element.classList.add("hidden"), 6000);
}

function setPlanFeedback(message = "", error = false) {
  const element = $("#plan-feedback");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.toggle("hidden", !message);
}

function setPreviewPending(pending) {
  const button = $("#preview-plan");
  button.disabled = pending;
  button.textContent = pending ? "Previewing…" : "Preview plan";
  button.setAttribute("aria-busy", String(pending));
}

function metric(label, value, detail = "") {
  return `<article class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}</article>`;
}

function setConnection(ok, label) {
  const element = $("#connection");
  element.textContent = label;
  element.className = `connection ${ok ? "" : "failed"}`;
}

async function bootstrap() {
  try {
    const [health, models] = await Promise.all([api("/api/health"), api("/api/models")]);
    state.models = models.items || [];
    setConnection(true, `Connected · ${health.principal.actor}`);
    populateModels();
    updatePlanMode();
    route();
  } catch (error) {
    setConnection(false, "API unavailable");
    notify(error.message, true);
  }
}

function populateModels() {
  const options = state.models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.id)}${model.enabled ? "" : " (disabled)"}</option>`).join("");
  $("#run-filters [name=model]").insertAdjacentHTML("beforeend", options);
  $("#plan-form [name=models]").insertAdjacentHTML("beforeend", options);
}

function populateRunIds() {
  const select = $("#plan-form [name=run_id]");
  if (!select) return;
  select.innerHTML = '<option value="">Select an existing run</option>' + state.runs.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${escapeHtml(run.status)}</option>`).join("");
}

const MODE_GUIDANCE = {
  quantize: ["Quantize new artifact", "Model, budget, inference container, and smoke script are required. Quality evaluation is skipped."],
  quantize_and_eval: ["Quantize and evaluate", "Model, budget, inference smoke, evaluation tool, and baseline are required."],
  eval_only: ["Evaluate an existing artifact", "Existing run, budget, inference smoke, evaluation tool, and baseline are required. Quantization fields are hidden."],
  upload_only: ["Publish an existing artifact", "Existing run and inference smoke are required. Evaluation and quantization fields are hidden."],
};

function updatePlanMode() {
  const form = $("#plan-form");
  const mode = form.elements.run_mode.value;
  $$('[data-modes]', form).forEach((element) => {
    const visible = element.dataset.modes.split(" ").includes(mode);
    element.classList.toggle("hidden", !visible);
    $$('input, select, textarea', element).forEach((field) => { field.disabled = !visible; });
  });
  const [title, detail] = MODE_GUIDANCE[mode];
  $("#mode-guidance").innerHTML = `<strong>${escapeHtml(title)}</strong> · ${escapeHtml(detail)}`;
  setPlanFeedback();
  clearPlanFieldErrors();
  updateBaselineFields();
}

function updateBaselineFields() {
  const form = $("#plan-form");
  const reference = form.elements.baseline_source?.value === "reference";
  $$('[data-baseline="reference"]', form).forEach((element) => {
    const visible = !element.closest("fieldset").classList.contains("hidden") && reference;
    element.classList.toggle("hidden", !visible);
    $$('input, select, textarea', element).forEach((field) => { field.disabled = !visible; });
  });
}

function validatePlanForm(form) {
  const mode = form.get("run_mode");
  const missing = [];
  if (["eval_only", "upload_only"].includes(mode) && !form.get("run_id")) missing.push(["run_id", "existing run ID"]);
  if (!form.get("container_name")) missing.push(["container_name", "inference container"]);
  if (!form.get("script")) missing.push(["script", "inference script"]);
  if (["quantize_and_eval", "eval_only"].includes(mode) && form.get("baseline_source") === "reference") {
    if (!form.get("reference_id")) missing.push(["reference_id", "reference ID"]);
    if (!form.get("reference_values")) missing.push(["reference_values", "reference metrics JSON"]);
  }
  if (missing.length) {
    const error = new Error(`Required fields missing: ${missing.map((item) => item[1]).join(", ")}`);
    error.fields = missing.map((item) => item[0]);
    throw error;
  }
}

function clearPlanFieldErrors() {
  $$("[aria-invalid='true']", $("#plan-form")).forEach((field) => field.removeAttribute("aria-invalid"));
}

function showPlanError(error) {
  clearPlanFieldErrors();
  setPlanFeedback(error.message, true);
  const fields = error.fields || [];
  fields.forEach((name) => $(`#plan-form [name="${name}"]`)?.setAttribute("aria-invalid", "true"));
  const firstField = fields.length ? $(`#plan-form [name="${fields[0]}"]`) : null;
  (firstField || $("#plan-feedback")).scrollIntoView({ behavior: "smooth", block: "center" });
  firstField?.focus({ preventScroll: true });
}

function route() {
  const hash = location.hash.slice(1) || "runs";
  closeEvents();
  $$(".view").forEach((view) => view.classList.add("hidden"));
  $$('[data-nav]').forEach((nav) => nav.classList.toggle("active", hash.startsWith(nav.dataset.nav)));
  if (hash.startsWith("run/")) {
    $("#run-view").classList.remove("hidden");
    loadRun(decodeURIComponent(hash.slice(4)));
  } else if (hash === "plan") {
    $("#plan-view").classList.remove("hidden");
    if (!state.runs.length) loadRuns({ updateTable: false });
  } else if (hash === "operations") {
    $("#operations-view").classList.remove("hidden");
    loadOperations();
  } else {
    $("#runs-view").classList.remove("hidden");
    loadRuns();
  }
  $("#app").focus({ preventScroll: true });
}

async function loadRuns(options = {}) {
  const updateTable = options.updateTable !== false;
  const form = new FormData($("#run-filters"));
  const query = new URLSearchParams();
  for (const [name, value] of form.entries()) if (value) query.set(name, value);
  if (updateTable) $("#run-rows").innerHTML = '<tr><td colspan="7" class="empty">Loading runs…</td></tr>';
  try {
    const result = await api(`/api/runs?${query}`);
    state.runs = result.items || [];
    populateRunIds();
    if (!updateTable) return;
    const totals = { total: result.total, running: 0, failed: 0, ready: 0 };
    for (const run of state.runs) {
      if (run.status === "RUNNING") totals.running += 1;
      if (["FAIL", "PUBLISH_FAILED", "RESOURCE_TIMEOUT"].includes(run.status)) totals.failed += 1;
      if (["READY", "SUCCESS"].includes(run.publish)) totals.ready += 1;
    }
    $("#run-summary").innerHTML = metric("Runs", totals.total) + metric("Running", totals.running) + metric("Needs attention", totals.failed) + metric("Published / ready", totals.ready);
    $("#run-rows").innerHTML = state.runs.length ? state.runs.map((run) => {
      const budget = run.budget || {};
      const modelCounts = run.models || {};
      return `<tr><td>${badge(run.status)}</td><td><a class="run-link" href="#run/${encodeURIComponent(run.run_id)}">${escapeHtml(run.run_id)}</a><div class="muted mono">${escapeHtml(short(run.git_sha))}</div></td><td>${escapeHtml(text(run.run_mode))}</td><td>${escapeHtml(run.model_count)} <span class="muted">(${modelCounts.PASS || 0} pass, ${modelCounts.FAIL || 0} fail)</span></td><td>${escapeHtml(formatNumber(budget.selected_gpu_hours))} / ${escapeHtml(formatNumber(budget.max_gpu_hours))} GPU-h</td><td>${badge(run.publish)}</td><td>${escapeHtml(formatDate(run.updated_at || run.created_at))}</td></tr>`;
    }).join("") : '<tr><td colspan="7" class="empty">No runs match these filters.</td></tr>';
  } catch (error) {
    $("#run-rows").innerHTML = `<tr><td colspan="7" class="empty">${escapeHtml(error.message)}</td></tr>`;
    notify(error.message, true);
  }
}

async function loadRun(runId) {
  $("#run-title").textContent = runId;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.run = run;
    state.jobs = [];
    renderRun();
    subscribe(runId);
    const jobs = await api(`/api/runs/${encodeURIComponent(runId)}/jobs`);
    state.jobs = jobs.items || [];
    renderJobs();
  } catch (error) {
    notify(error.message, true);
  }
}

function renderRun() {
  const run = state.run;
  $("#run-title").innerHTML = `${badge(run.status)} <span class="mono">${escapeHtml(run.run_id)}</span>`;
  $("#run-subtitle").textContent = `${text(run.run_mode)} · git ${short(run.git_sha)} · ${formatDate(run.updated_at || run.created_at)}`;
  const budget = run.budget || {};
  $("#run-cards").innerHTML = metric("Status", text(run.status)) + metric("Attempt", short(run.attempt_id, 22)) + metric("GPU-hour budget", `${formatNumber(budget.selected_gpu_hours)} / ${formatNumber(budget.max_gpu_hours)}`, `${formatNumber(budget.remaining_gpu_hours)} remaining`) + metric("Publish", text(run.publish));
  const permissions = new Set(identity().role === "admin" ? ["retry", "evaluate", "cancel", "publish", "backup"] : identity().role === "publisher" ? ["retry", "evaluate", "cancel", "publish", "backup"] : identity().role === "operator" ? ["retry", "evaluate", "cancel"] : []);
  const actions = [];
  if (permissions.has("retry")) actions.push('<button class="button" data-action="retry">Retry failed</button>');
  if (permissions.has("evaluate")) actions.push('<button class="button" data-action="evaluate">Evaluate</button>');
  if (permissions.has("backup")) actions.push('<button class="button" data-action="backup-preview">Backup preview</button>');
  if (permissions.has("publish")) actions.push('<button class="button" data-action="publish-preview">Publish preview</button>');
  if (permissions.has("cancel")) actions.push('<button class="button danger" data-action="cancel">Cancel run</button>');
  $("#run-actions").innerHTML = actions.join("");
  renderFailures();
  renderTimeline(run.models || []);
  renderJobs();
  renderModels();
  configureLogs();
}

function failureMessage(value) {
  if (!value || typeof value !== "object") return "No failure detail was recorded. Open the stage log for more information.";
  if (value.message) return String(value.message);
  if (Array.isArray(value.failures) && value.failures.length) return value.failures.join("; ");
  if (value.reason) return String(value.reason);
  if (value.reason_code) return String(value.reason_code).replaceAll("_", " " ).toLowerCase();
  return "No failure detail was recorded. Open the stage log for more information.";
}

function renderFailures() {
  const failures = [];
  for (const failure of (state.run.failure_history || [])) {
    failures.push({
      model: failure.model_id || state.run.models?.[0]?.model_id || "model",
      stage: failure.stage || "unknown",
      reasonCode: failure.reason_code || failure.failure_class || failure.status,
      message: failure.message || (failure.status === "CANCELED" ? "Job was canceled." : failureMessage(failure)),
      attemptId: failure.attempt_id,
      current: Boolean(failure.current),
      status: failure.status,
    });
  }
  for (const model of state.run.models || []) {
    for (const stage of STAGES) {
      const detail = model.stage_details?.[stage];
      if (!["FAIL", "FAILED", "RESOURCE_TIMEOUT"].includes(String(detail?.status || "").toUpperCase())) continue;
      if (!failures.some((item) => item.model === model.model_id && item.stage === stage && item.attemptId === detail.attempt_id)) failures.push({ model: model.model_id, stage, reasonCode: detail.reason_code, message: failureMessage(detail), attemptId: detail.attempt_id, current: true });
    }
  }
  for (const job of state.jobs) {
    if (!failureStatus(job.status) && job.failure_class !== "CANCELED") continue;
    const failedStage = job.details?.failed_stage || (job.stages || [])[0] || job.kind;
    const model = (state.run.models || []).find((item) => item.model_id === job.model_id);
    const latestStage = model?.stage_details?.[failedStage];
    if (latestStage?.attempt_id === job.attempt_id && !failureStatus(latestStage.status)) continue;
    if (job.failure_class === "CANCELED" && !String(job.details?.cancel_reason || "").startsWith("failed:")) continue;
    if (failures.some((item) => item.model === job.model_id && item.stage === failedStage && item.attemptId === job.attempt_id)) continue;
    failures.push({ model: job.model_id, stage: failedStage, reasonCode: job.failure_class, message: job.details?.message || job.details?.cancel_reason || failureMessage(job.details), attemptId: job.attempt_id, current: false });
  }
  const hasCurrentFailure = failures.some((failure) => failure.current);
  $("#failure-eyebrow").textContent = hasCurrentFailure ? "Needs attention" : "Earlier attempts";
  $("#failure-heading").textContent = hasCurrentFailure ? "Current failure reasons" : "Failure history";
  $("#run-failures").classList.toggle("hidden", !failures.length);
  $("#failure-list").innerHTML = failures.map((failure) => `<article class="failure-item ${failure.current ? "current" : "historical"}"><div><strong>${escapeHtml(failure.stage)} failed</strong> ${failure.reasonCode ? `<span class="reason-code">${escapeHtml(failure.reasonCode)}</span>` : ""} <span class="failure-age">${failure.current ? "Current" : "Historical"}</span></div><p>${escapeHtml(failure.message)}</p><small>${escapeHtml(failure.model)} · attempt ${escapeHtml(short(failure.attemptId, 24))}</small><button class="ghost failure-log" type="button" data-failure-model="${escapeHtml(failure.model)}" data-failure-stage="${escapeHtml(failure.stage)}" data-failure-attempt="${escapeHtml(failure.attemptId || "")}">Open stage log</button></article>`).join("");
}

function failureStatus(value) {
  return ["FAIL", "FAILED", "RESOURCE_TIMEOUT"].includes(String(value || "").toUpperCase());
}

function renderTimeline(models) {
  const statuses = {};
  for (const stage of STAGES) {
    const values = models.map((model) => model.stages?.[stage] || "PENDING");
    statuses[stage] = values.some((value) => ["FAIL", "FAILED"].includes(value)) ? "FAIL" : values.some((value) => ["RUNNING", "QUEUED"].includes(value)) ? "RUNNING" : values.some((value) => value === "WARN") ? "WARN" : values.length && values.every((value) => ["PASS", "SUCCESS", "SUCCEEDED", "SKIPPED"].includes(value)) ? "PASS" : "PENDING";
  }
  $("#timeline").innerHTML = STAGES.map((stage) => `<article class="stage ${statusClass(statuses[stage])}"><span class="stage-mark">${statuses[stage] === "PASS" ? "✓" : statuses[stage] === "FAIL" ? "!" : "·"}</span><strong>${escapeHtml(stage)}</strong><small>${escapeHtml(statuses[stage])}</small></article>`).join("");
}

function renderJobs() {
  $("#job-lanes").innerHTML = state.jobs.length ? state.jobs.map((job) => `<article class="lane ${failureStatus(job.status) ? "failed" : ""}"><h3>${escapeHtml(job.kind)} ${badge(job.status)}</h3><dl><dt>Queue</dt><dd>${escapeHtml(text(job.queue))}</dd><dt>GPU</dt><dd>${escapeHtml(text(job.gpu_count, "0"))} · ${escapeHtml(formatNumber(job.gpu_hours))} h</dd><dt>Attempt</dt><dd class="mono">${escapeHtml(short(job.attempt_id, 18))}</dd><dt>Stages</dt><dd>${escapeHtml((job.stages || []).join(" → "))}</dd><dt>Exit</dt><dd>${escapeHtml(text(job.exit_code))}</dd>${failureStatus(job.status) ? `<dt>Reason</dt><dd class="job-failure">${escapeHtml(job.failure_class || job.details?.reason_code || "FAILED")} · ${escapeHtml(job.details?.message || "Open the stage log for details.")}</dd>` : ""}</dl></article>`).join("") : '<p class="empty">No execution jobs recorded.</p>';
  renderFailures();
}

function openFailureLog(button) {
  $("#log-model").value = button.dataset.failureModel;
  $("#log-stage").value = button.dataset.failureStage;
  updateAttempts();
  if (button.dataset.failureAttempt) $("#log-attempt").value = button.dataset.failureAttempt;
  loadLog();
  $("#log-output").scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderModels() {
  $("#model-list").innerHTML = (state.run.models || []).map((model) => {
    const attempts = (model.attempts || []).map((attempt) => `${short(attempt.attempt_id, 18)} · ${attempt.status}`).join(" | ");
    const report = model.report || {};
    return `<article class="model-card"><div class="model-head"><div><h3>${escapeHtml(model.model_id)}</h3><span class="muted">${escapeHtml(attempts || "No attempt history")}</span></div>${badge(model.status)}</div><div class="fingerprints"><span>Artifact <code>${escapeHtml(model.artifact_fingerprint || "not finalized")}</code></span><span>Evaluation <code>${escapeHtml(model.evaluation_fingerprint || "not finalized")}</code></span><span>Quality report <code>${escapeHtml(report.status || "not available")}</code></span></div></article>`;
  }).join("");
}

function configureLogs() {
  const models = state.run.models || [];
  const selectedModel = $("#log-model").value;
  const selectedStage = $("#log-stage").value;
  $("#log-model").innerHTML = models.map((model) => `<option>${escapeHtml(model.model_id)}</option>`).join("");
  $("#log-stage").innerHTML = STAGES.map((stage) => `<option>${stage}</option>`).join("");
  if (models.some((model) => model.model_id === selectedModel)) $("#log-model").value = selectedModel;
  if (STAGES.includes(selectedStage)) $("#log-stage").value = selectedStage;
  updateAttempts();
}

function updateAttempts() {
  const model = (state.run?.models || []).find((item) => item.model_id === $("#log-model").value);
  const stage = $("#log-stage").value;
  const allAttempts = model?.attempts || [];
  const attempts = allAttempts.filter((attempt) => {
    const stages = attempt.stages || {};
    return Object.prototype.hasOwnProperty.call(stages, stage);
  });
  const selectedAttempt = $("#log-attempt").value;
  $("#log-attempt").innerHTML = attempts.length ? attempts.map((attempt) => `<option>${escapeHtml(attempt.attempt_id)}</option>`).join("") : '<option value="">No recorded attempt</option>';
  if (attempts.some((attempt) => attempt.attempt_id === selectedAttempt)) $("#log-attempt").value = selectedAttempt;
}

async function loadLog() {
  const runId = encodeURIComponent(state.run.run_id);
  const model = encodeURIComponent($("#log-model").value);
  const query = new URLSearchParams({ stage: $("#log-stage").value, attempt_id: $("#log-attempt").value, limit: "1000" });
  $("#log-output").textContent = "Loading log…";
  try {
    const result = await api(`/api/runs/${runId}/models/${model}/logs?${query}`);
    $("#log-output").textContent = (result.items || []).map((item) => item.text).join("\n") || "No log lines found.";
  } catch (error) {
    $("#log-output").textContent = error.message;
  }
}

function subscribe(runId) {
  if (state.eventTimer) return;
  $("#live-state").textContent = "Live · 5s";
  state.eventTimer = window.setInterval(async () => {
    if (!location.hash.startsWith("#run/")) return;
    try {
      const [run, jobs] = await Promise.all([
        api(`/api/runs/${encodeURIComponent(runId)}`),
        api(`/api/runs/${encodeURIComponent(runId)}/jobs`),
      ]);
      state.run = run;
      state.jobs = jobs.items || [];
      renderRun();
    } catch (error) {
      $("#live-state").textContent = "Snapshot";
    }
  }, 5000);
}

function closeEvents() {
  if (state.eventTimer) window.clearInterval(state.eventTimer);
  state.eventTimer = null;
}

async function previewPlan(event) {
  event.preventDefault();
  setPlanFeedback();
  clearPlanFieldErrors();
  const form = new FormData(event.currentTarget);
  try { validatePlanForm(form); } catch (error) { showPlanError(error); return; }
  const request = { run_mode: form.get("run_mode"), max_gpu_hours: Number(form.get("max_gpu_hours")) };
  if (form.get("models")) request.models = form.get("models");
  if (form.get("run_id")) request.run_id = form.get("run_id");
  if (form.get("container_name") || form.get("script")) request.inference = { container_name: form.get("container_name"), script: form.get("script"), arguments: lines(form.get("inference_arguments")), result_file: "{reports_dir}/runtime-smoke-output.json" };
  const evaluationRequired = ["quantize_and_eval", "eval_only"].includes(form.get("run_mode"));
  const evaluation = lines(form.get("evaluation_command"));
  if (evaluationRequired && !evaluation.length) {
    const selectedTool = form.get("evaluation_tool");
    const selectedPython = form.get("evaluation_python") || (selectedTool === "evalscope" ? "/root/miniconda/envs/model_quality_evalscope/bin/python" : "/root/miniconda/envs/model_quality_lm_eval/bin/python");
    evaluation.push(
      selectedPython, "-m",
      selectedTool === "evalscope" ? "ci.model_quality.evaluators.evalscope_api" : "ci.model_quality.evaluators.lm_eval_api_pair"
    );
  }
  if (evaluationRequired && form.get("baseline_source") === "reference" && form.get("reference_values")) {
    if (!evaluation.includes("--reference-values-json")) evaluation.push("--reference-values-json", form.get("reference_values"));
    if (form.get("reference_id") && !evaluation.includes("--reference-id")) evaluation.push("--reference-id", form.get("reference_id"));
  }
  if (evaluationRequired && evaluation.length) request.evaluation_command = evaluation;
  setPreviewPending(true);
  setPlanFeedback("Submitting plan preview…");
  try {
    state.plan = await api("/api/plans/preview", { method: "POST", body: JSON.stringify(request) });
    const risks = state.plan.review?.risks || [];
    $("#plan-risks").innerHTML = risks.map((risk) => `<div class="risk ${risk.severity === "error" ? "error" : ""}"><strong>${escapeHtml(risk.code)}</strong> · ${escapeHtml(risk.message)}</div>`).join("") || '<div class="notice">No policy risks found.</div>';
    $("#plan-selected").innerHTML = (state.plan.plan.selected || []).map((model) => `<article class="review-card"><h3>${escapeHtml(model.id)}</h3><p>${formatNumber(model.estimated_gpu_hours)} GPU-h · ${escapeHtml(model.run_mode)}</p><p>artifact ${escapeHtml(short(model.fingerprint, 18))}</p><p>evaluation ${escapeHtml(short(model.evaluation_fingerprint, 18))}</p><p>${model.upload_enabled ? "Publish will be appended" : "No automatic publish"}</p></article>`).join("") || '<p class="empty">No model selected.</p>';
    $("#plan-json").textContent = JSON.stringify(state.plan, null, 2);
    $("#plan-review").classList.remove("hidden");
    $("#launch-plan").disabled = risks.some((risk) => risk.severity === "error");
    setPlanFeedback("Plan preview is ready. Review the launch contract below.");
    $("#plan-review").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showPlanError(error);
    notify(error.message, true);
  } finally {
    setPreviewPending(false);
  }
}

function lines(value) {
  return String(value || "").split("\n").map((token) => token.trim()).filter(Boolean);
}

async function launchPlan() {
  if (!state.plan) return;
  try {
    const result = await api("/api/runs", { method: "POST", body: JSON.stringify({ plan_hash: state.plan.plan_hash, idempotency_key: `ui-start-${key()}` }) });
    notify(`Run ${result.run_id} queued.`);
    location.hash = `run/${encodeURIComponent(result.run_id)}`;
  } catch (error) { notify(error.message, true); }
}

async function runAction(action) {
  const runId = encodeURIComponent(state.run.run_id);
  if (action === "backup-preview") return showPreview(await api(`/api/runs/${runId}/backup/preview`, { method: "POST", body: "{}" }), "Backup preview");
  if (action === "publish-preview") return showPreview(await api(`/api/runs/${runId}/publish/preview`, { method: "POST", body: "{}" }), "Publish preview");
  const request = { idempotency_key: `ui-${action}-${key()}` };
  if (action === "cancel") request.reason = await confirmAction("Cancel run", "Cancel all active jobs and release outstanding reservations?", true);
  else {
    const confirmed = await confirmAction(action === "retry" ? "Retry failed stages" : "Evaluate artifact", action === "retry" ? "Create a new attempt from the first failed stage?" : "Create an eval-only attempt for the validated artifact?");
    if (!confirmed) return;
  }
  if (action === "cancel" && request.reason === null) return;
  const result = await api(`/api/runs/${runId}/${action}`, { method: "POST", body: JSON.stringify(request) });
  notify(`${action} accepted for ${result.run_id || state.run.run_id}.`);
  await loadRun(state.run.run_id);
}

function showPreview(payload, title) {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-message").textContent = JSON.stringify(payload, null, 2);
  $("#confirm-submit").classList.add("hidden");
  dialog.showModal();
  dialog.addEventListener("close", () => $("#confirm-submit").classList.remove("hidden"), { once: true });
}

function confirmAction(title, message, reason = false) {
  return new Promise((resolve) => {
    const dialog = $("#confirm-dialog");
    $("#confirm-title").textContent = title;
    $("#confirm-message").textContent = message;
    $("#confirm-reason-label").classList.toggle("hidden", !reason);
    $("#confirm-reason").value = "";
    dialog.showModal();
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "default" ? (reason ? $("#confirm-reason").value || "operator requested" : true) : null), { once: true });
  });
}

async function loadOperations() {
  try {
    const [cost, capabilitiesResult, schedulesResult] = await Promise.all([
      api("/api/trends/cost"),
      api("/api/capabilities").then((value) => ({ value })).catch((error) => ({ error })),
      api("/api/ops/schedules").then((value) => ({ value })).catch((error) => ({ error })),
    ]);
    if (capabilitiesResult.error) {
      $("#capacity-cards").innerHTML = metric("Capacity", "Sign in", "Operator role required");
      $("#capacity-detail").innerHTML = `<p class="empty">${escapeHtml(capabilitiesResult.error.message)} Open Identity and select operator, publisher, or admin.</p>`;
    }
    if (schedulesResult.error) {
      $("#schedule-list").innerHTML = `<p class="empty">${escapeHtml(schedulesResult.error.message)} Open Identity to view schedules.</p>`;
    }
    const capabilities = capabilitiesResult.value;
    const schedules = schedulesResult.value;
    if (capabilities) {
      const capacity = capabilities.capacity || {};
      const forecast = capabilities.forecast || {};
      $("#capacity-cards").innerHTML = metric("GPU-hour capacity", formatNumber(capacity.gpu_hour_capacity)) + metric("Reserved", formatNumber(capacity.reserved_gpu_hours)) + metric("Available", formatNumber(capacity.available_gpu_hours)) + metric("Drain estimate", forecast.eta_hours === null ? "No history" : `${formatNumber(forecast.eta_hours)} h`, `${forecast.confidence || "none"} confidence`);
      $("#capacity-detail").innerHTML = `<div class="bar-list">${(capabilities.queues || []).map((queue) => { const percent = capacity.gpu_hour_capacity ? Math.min(100, 100 * queue.reserved_gpu_hours / capacity.gpu_hour_capacity) : 0; return `<div class="bar-row"><div class="bar-label"><strong>${escapeHtml(queue.name)}</strong><span>${queue.queued_jobs} queued · ${queue.active_jobs} active · ${queue.gpu_devices} GPU</span></div><div class="bar"><i style="width:${percent}%"></i></div></div>`; }).join("")}${(capabilities.hosts || []).map((host) => `<p><strong>${escapeHtml(host.name)}</strong><br><span class="muted">${escapeHtml(host.platform)} · ${host.gpu_devices} GPU</span></p>`).join("")}</div>`;
    }
    const points = cost.points || [];
    const max = Math.max(1, ...points.map((point) => Number(point.actual_gpu_hours || point.estimated_gpu_hours || 0)));
    $("#cost-trend").innerHTML = points.length ? `<div class="bar-list">${points.slice(-10).map((point) => `<div class="bar-row"><div class="bar-label"><span class="mono">${escapeHtml(short(point.run_id, 22))}</span><span>${formatNumber(point.actual_gpu_hours || point.estimated_gpu_hours)} GPU-h</span></div><div class="bar"><i style="width:${Math.round(100 * Number(point.actual_gpu_hours || point.estimated_gpu_hours || 0) / max)}%"></i></div></div>`).join("")}</div>` : '<p class="empty">No cost history yet.</p>';
    if (schedules) $("#schedule-list").innerHTML = schedules.items?.length ? `<div class="table-shell"><table><thead><tr><th>ID</th><th>Cron</th><th>Status</th><th>Next run</th><th>Last run</th></tr></thead><tbody>${schedules.items.map((schedule) => `<tr><td class="mono">${escapeHtml(schedule.schedule_id)}</td><td><code>${escapeHtml(schedule.cron)}</code></td><td>${badge(schedule.enabled ? "ENABLED" : "DISABLED")}</td><td>${escapeHtml(formatDate(schedule.next_run_at))}</td><td>${escapeHtml(schedule.last_run_id || "—")}</td></tr>`).join("")}</tbody></table></div>` : '<p class="empty">No schedules configured.</p>';
  } catch (error) { notify(error.message, true); }
}

function openIdentity() {
  const current = identity();
  $("#actor-input").value = current.actor;
  $("#role-input").value = current.role;
  $("#token-input").value = current.token;
  $("#identity-dialog").showModal();
}

function saveIdentity(event) {
  event.preventDefault();
  localStorage.setItem("mq.actor", $("#actor-input").value.trim());
  localStorage.setItem("mq.role", $("#role-input").value);
  localStorage.setItem("mq.token", $("#token-input").value);
  $("#identity-dialog").close();
  bootstrap();
}

window.addEventListener("hashchange", route);
$("#run-filters").addEventListener("submit", (event) => { event.preventDefault(); loadRuns(); });
$("#refresh-runs").addEventListener("click", loadRuns);
$("#plan-form").addEventListener("submit", previewPlan);
$("#plan-form [name=run_mode]").addEventListener("change", updatePlanMode);
$("#plan-form [name=baseline_source]").addEventListener("change", updateBaselineFields);
$("#launch-plan").addEventListener("click", launchPlan);
$("#identity-button").addEventListener("click", openIdentity);
$("#save-identity").addEventListener("click", saveIdentity);
$("#log-model").addEventListener("change", updateAttempts);
$("#log-stage").addEventListener("change", updateAttempts);
$("#load-log").addEventListener("click", loadLog);
$("#run-actions").addEventListener("click", async (event) => { const action = event.target.closest("[data-action]")?.dataset.action; if (action) try { await runAction(action); } catch (error) { notify(error.message, true); } });
document.addEventListener("click", (event) => { const target = event.target.closest("[data-go]"); if (target) location.hash = target.dataset.go; });
$("#failure-list").addEventListener("click", (event) => { const button = event.target.closest(".failure-log"); if (button) openFailureLog(button); });
bootstrap();
