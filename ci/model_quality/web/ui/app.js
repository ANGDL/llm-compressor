"use strict";

const STAGES = ["preflight", "quantize", "validate", "runtime-smoke", "evaluate", "report", "publish"];
const ACTIVE_JOB = new Set(["CREATED", "QUEUED", "ALLOCATED", "RUNNING"]);
const TERMINAL_STAGE = new Set(["PASS", "WARN", "FAIL", "FAILED", "SUCCESS", "SUCCEEDED", "SKIPPED", "RESOURCE_TIMEOUT", "CANCELED"]);
const state = { models: [], runs: [], run: null, jobs: [], plan: null, scriptParams: [], cloneRequest: null, eventTimer: null, logTimer: null, opsTimer: null, logFollow: true, nodeMenu: null, nodeMenuKey: null, lanesExpanded: false };
const EVAL_PRESETS = { lm_eval: ["gsm8k", "mmlu"], evalscope: ["gsm8k", "ceval"] };
const ROLE_HELP = {
  viewer: "Viewer · read-only: browse runs, logs, and the GPU/ops dashboards. Cannot plan, launch, retry, cancel, skip, or publish.",
  operator: "Operator · all Viewer rights, plus: preview & launch plans, retry, cancel, skip, and Run-from-here (incl. runtime-smoke / evaluate).",
  publisher: "Publisher · all Operator rights, plus: publish artifacts and create backups.",
  admin: "Admin · all Publisher rights, plus: restore backups and manage schedules.",
};
// Argparse destinations the backend forces onto run-scoped placeholders.
const MANAGED_DESTS = new Set(["output_dir", "output", "save_dir", "save_path", "save_directory", "out_dir", "work_dir", "workdir", "cache_dir", "tmp_dir", "scratch_dir"]);

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
    updateStageSections();
    updateEvalDatasets();
    route();
  } catch (error) {
    setConnection(false, "API unavailable");
    notify(error.message, true);
  }
}

function modelFilterIds() {
  const ids = new Set(state.models.map((model) => model.id));
  for (const run of state.runs) for (const id of (run.model_ids || [])) ids.add(id);
  return Array.from(ids).sort();
}

function populateModels() {
  const select = $("#run-filters [name=model]");
  if (!select) return;
  // The filter must list models seen in runs too, not only the catalog, so a
  // run whose model is not in models.yaml (e.g. a script-first/generated run)
  // is still selectable.
  const current = select.value;
  const ids = modelFilterIds();
  select.innerHTML = '<option value="">All models</option>'
    + ids.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
  if (ids.includes(current)) select.value = current;
}

function populateRunIds() {
  const select = $("#plan-form [name=run_id]");
  if (!select) return;
  select.innerHTML = '<option value="">Select an existing run</option>' + state.runs.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${escapeHtml(run.status)}</option>`).join("");
}

async function loadScriptParams() {
  const script = ($("#plan-form [name=script]").value || "").trim();
  const help = $("#script-params-help");
  if (!script) { setPlanFeedback("Enter a script path first.", true); return; }
  help.textContent = "Loading parameters…";
  const button = $("#load-params");
  button.disabled = true;
  try {
    const result = await api(`/api/scripts/introspect?path=${encodeURIComponent(script)}`);
    state.scriptParams = result.parameters || [];
    renderScriptParams(result);
    setPlanFeedback();
  } catch (error) {
    state.scriptParams = [];
    $("#script-params").innerHTML = "";
    help.textContent = error.message;
    setPlanFeedback(error.message, true);
  } finally {
    button.disabled = false;
  }
}

function renderScriptParams(result) {
  const container = $("#script-params");
  const manual = $("#manual-argv-label");
  const help = $("#script-params-help");
  const params = result.parameters || [];
  if (!result.parseable || !params.length) {
    container.innerHTML = "";
    manual.classList.remove("hidden");
    help.textContent = result.parseable
      ? "No argparse parameters were found. Provide a manual argv below."
      : "This script does not use a literal argparse. Provide a manual argv below.";
    return;
  }
  manual.classList.add("hidden");
  help.textContent = `${params.length} parameter(s) parsed from ${result.entrypoint}. Managed output/work directories are set automatically.`;
  container.innerHTML = params.map(renderParamField).join("");
}

function renderParamField(param) {
  const flag = param.flag;
  const meta = [param.type, param.action, param.positional ? "positional" : ""].filter(Boolean).join(" · ");
  const req = param.required ? '<span class="required-tag">Required</span>' : '<span class="optional-tag">Optional</span>';
  const help = param.help ? `<small class="muted">${escapeHtml(param.help)}</small>` : "";
  const isBool = param.type === "bool" || param.action === "store_true" || param.action === "store_false";
  if (isBool) {
    // Render booleans with the same flag/tag/meta header as the other fields,
    // then a left-aligned checkbox. (A bare checkbox inside the form grid gets
    // stretched to full width by the generic input rule, which scattered the
    // box and flag across the row.)
    const field = `<span class="param-checkbox"><input type="checkbox" data-param="${escapeHtml(flag)}" ${param.default ? "checked" : ""}></span>`;
    return `<div class="param-item">${escapeHtml(flag)} ${req}<span class="muted" style="font-weight:500;">${escapeHtml(meta)}</span>${field}${help}</div>`;
  }
  const managed = MANAGED_DESTS.has(param.dest);
  let field;
  if (managed) {
    field = `<input data-param="${escapeHtml(flag)}" value="(managed by CI)" readonly title="Set to a run-scoped directory at launch">`;
  } else if (Array.isArray(param.choices) && param.choices.length) {
    field = `<select data-param="${escapeHtml(flag)}">${param.choices.map((choice) => `<option ${String(param.default) === String(choice) ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>`;
  } else {
    const value = param.default === null || param.default === undefined ? "" : param.default;
    field = `<input data-param="${escapeHtml(flag)}" value="${escapeHtml(value)}"${param.editable ? "" : ' placeholder="default could not be resolved"'}>`;
  }
  return `<label class="param-item">${escapeHtml(flag)} ${req}<span class="muted" style="font-weight:500;">${escapeHtml(meta)}</span>${field}${help}</label>`;
}

function collectScriptParams() {
  return $$("#script-params [data-param]").map((element) => {
    const flag = element.dataset.param;
    const managed = element.hasAttribute("readonly");
    const value = element.type === "checkbox" ? element.checked : element.value;
    return { flag, value, managed };
  }).filter((item) => !item.managed).map(({ flag, value }) => ({ flag, value }));
}

function updateStageSections() {
  const form = $("#plan-form");
  const evaluate = form.elements.stage_evaluate?.checked;
  const inferenceBox = form.elements.stage_inference;
  // Evaluate reuses the smoke container, so it requires inference. Auto-check
  // and lock inference whenever evaluate is on.
  if (inferenceBox) {
    if (evaluate) { inferenceBox.checked = true; inferenceBox.disabled = true; }
    else { inferenceBox.disabled = false; }
  }
  const inference = inferenceBox?.checked;
  $$('[data-stage-section]', form).forEach((section) => {
    const want = section.dataset.stageSection === "inference" ? inference : evaluate;
    section.classList.toggle("hidden", !want);
    $$('input, select, textarea', section).forEach((field) => { field.disabled = !want; });
  });
  setPlanFeedback();
  clearPlanFieldErrors();
}

function updateEvalDatasets() {
  const tool = $("#plan-form [name=evaluation_tool]")?.value || "lm_eval";
  const select = $("#plan-form [name=evaluation_dataset]");
  if (!select) return;
  const previous = select.value;
  const datasets = EVAL_PRESETS[tool] || [];
  select.innerHTML = datasets.map((dataset) => `<option>${escapeHtml(dataset)}</option>`).join("");
  if (datasets.includes(previous)) select.value = previous;
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
    if (state.cloneRequest) { const req = state.cloneRequest; state.cloneRequest = null; applyCloneRequest(req); }
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
    populateModels();
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
  $("#run-title").innerHTML = `<span class="mono">${escapeHtml(run.run_id)}</span>`;
  $("#run-status-badge").innerHTML = badge(run.status);
  $("#run-subtitle").textContent = `${text(run.run_mode)} · git ${short(run.git_sha)} · ${formatDate(run.updated_at || run.created_at)}`;
  const budget = run.budget || {};
  $("#run-cards").innerHTML = metric("Status", text(run.status)) + metric("Attempt", short(run.attempt_id, 22)) + metric("GPU-hour budget", `${formatNumber(budget.selected_gpu_hours)} / ${formatNumber(budget.max_gpu_hours)}`, `${formatNumber(budget.remaining_gpu_hours)} remaining`) + metric("Publish", text(run.publish));
  const permissions = new Set(identity().role === "admin" ? ["retry", "evaluate", "cancel", "publish", "backup"] : identity().role === "publisher" ? ["retry", "evaluate", "cancel", "publish", "backup"] : identity().role === "operator" ? ["retry", "evaluate", "cancel"] : []);
  const actions = ['<button class="button" data-action="clone">Clone to New plan</button>'];
  if (permissions.has("retry")) actions.push('<button class="button" data-action="retry">Retry failed</button>');
  if (permissions.has("evaluate")) actions.push('<button class="button" data-action="evaluate">Evaluate</button>');
  if (permissions.has("backup")) actions.push('<button class="button" data-action="backup-preview">Backup preview</button>');
  if (permissions.has("publish")) actions.push('<button class="button" data-action="publish-preview">Publish preview</button>');
  if (permissions.has("cancel")) actions.push('<button class="button danger" data-action="cancel">Cancel run</button>');
  $("#run-actions").innerHTML = actions.join("");
  renderFailures();
  renderPipeline(run.models || []);
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
      // If the stage is live again (re-run in flight), the recorded failure is
      // history, not the current state — surface it, but not as "Current".
      const liveStatus = String(model.stages?.[stage] || "").toUpperCase();
      const superseded = ACTIVE_JOB.has(liveStatus);
      if (!failures.some((item) => item.model === model.model_id && item.stage === stage && item.attemptId === detail.attempt_id)) failures.push({ model: model.model_id, stage, reasonCode: detail.reason_code, message: failureMessage(detail), attemptId: detail.attempt_id, current: !superseded });
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
  const panel = $("#run-failures");
  panel.classList.toggle("hidden", !failures.length);
  $("#failure-count").textContent = failures.length
    ? `${failures.length} ${failures.length === 1 ? "attempt" : "attempts"}`
    : "";
  // Expand automatically only when something is currently broken; a run whose
  // failures are all historical stays collapsed so it doesn't bury the live
  // pipeline. Keyed on a signature so the 5s live refresh never fights a manual
  // open/close the operator did while reading the history.
  const signature = `${failures.length}:${hasCurrentFailure}`;
  if (panel.dataset.sig !== signature) {
    panel.dataset.sig = signature;
    panel.open = hasCurrentFailure;
  }
  $("#failure-list").innerHTML = failures.map((failure) => `<article class="failure-item ${failure.current ? "current" : "historical"}"><div><strong>${escapeHtml(failure.stage)} failed</strong> ${failure.reasonCode ? `<span class="reason-code">${escapeHtml(failure.reasonCode)}</span>` : ""} <span class="failure-age">${failure.current ? "Current" : "Historical"}</span></div><p>${escapeHtml(failure.message)}</p><small>${escapeHtml(failure.model)} · attempt ${escapeHtml(short(failure.attemptId, 24))}</small><button class="ghost failure-log" type="button" data-failure-model="${escapeHtml(failure.model)}" data-failure-stage="${escapeHtml(failure.stage)}" data-failure-attempt="${escapeHtml(failure.attemptId || "")}">Open stage log</button></article>`).join("");
}

function failureStatus(value) {
  return ["FAIL", "FAILED", "RESOURCE_TIMEOUT"].includes(String(value || "").toUpperCase());
}

function canControlPipeline() {
  return ["operator", "publisher", "admin"].includes(identity().role);
}

function canPublish() {
  return ["publisher", "admin"].includes(identity().role);
}

function findStageJob(modelId, stage) {
  // Several stage nodes can share one job (preflight/quantize/validate map to
  // the quantize job). Prefer a still-active job so Stop targets the live work;
  // otherwise fall back to the most recent matching job.
  const matches = state.jobs.filter((job) => job.model_id === modelId && (job.stages || []).includes(stage));
  return matches.find((job) => ACTIVE_JOB.has(String(job.status || "").toUpperCase())) || matches[matches.length - 1] || null;
}

function nodeMark(status) {
  if (["PASS", "SUCCESS", "SUCCEEDED"].includes(status)) return "✓";
  if (["FAIL", "FAILED", "RESOURCE_TIMEOUT"].includes(status)) return "!";
  if (status === "RUNNING") return "⟳";
  if (status === "QUEUED") return "…";
  if (status === "SKIPPED") return "⊘";
  return "·";
}

function renderPipeline(models) {
  closeNodeMenu();
  const container = $("#pipeline");
  if (!models.length) { container.innerHTML = '<p class="empty">No models in this run.</p>'; return; }
  container.innerHTML = models.map((model) => {
    const nodes = STAGES.map((stage) => {
      const stageStatus = String(model.stages?.[stage] || "PENDING").toUpperCase();
      const job = findStageJob(model.model_id, stage);
      const jobStatus = String(job?.status || "").toUpperCase();
      // Several stages share one job (preflight/quantize/validate). They run
      // sequentially, so only the earliest not-yet-finished stage of an active
      // job is actually RUNNING; the rest are still QUEUED behind it. Use the
      // active attempt's own stage states (not the model-latest rollup, which is
      // stale during a re-run) to decide which one.
      let status;
      if (ACTIVE_JOB.has(jobStatus)) {
        const attempt = (model.attempts || []).find((a) => a.attempt_id === job.attempt_id);
        const astages = (attempt && attempt.stages) || {};
        const jobStages = job.stages || [stage];
        const running = jobStages.find((s) => !TERMINAL_STAGE.has(String(astages[s] || "PENDING").toUpperCase()));
        const own = String(astages[stage] || "PENDING").toUpperCase();
        if (TERMINAL_STAGE.has(own)) status = own;
        // A newer re-run can queue a fresh job for a stage that already finished
        // in the promoted attempt. Until that job actually starts running, the
        // completed result stands — a stage genuinely re-running flips the rollup
        // to RUNNING via the store's live-log check, so a terminal rollup behind
        // a not-yet-started job means the queued re-run has not superseded the
        // finished stage yet. Don't let its QUEUED job repaint a done stage as
        // pending. Once the job is RUNNING we keep the "queued behind the running
        // stage" semantics for the other stages it owns.
        else if (jobStatus !== "RUNNING" && TERMINAL_STAGE.has(stageStatus)) status = stageStatus;
        else status = stage === running ? jobStatus : "QUEUED";
      } else if (TERMINAL_STAGE.has(stageStatus)) {
        status = stageStatus;
      } else if (job && TERMINAL_STAGE.has(jobStatus)) {
        // The stage recorded no state but its job finished (e.g. the container
        // runtime-smoke failed before writing a result) — reflect the job's
        // terminal status instead of a misleading PENDING.
        status = jobStatus;
      } else {
        status = stageStatus;
      }
      return `<button type="button" class="node ${statusClass(status)}" data-model="${escapeHtml(model.model_id)}" data-stage="${escapeHtml(stage)}" data-job="${escapeHtml(job?.job_id || "")}" data-jobstatus="${escapeHtml(job?.status || "")}" data-status="${escapeHtml(status)}" aria-label="${escapeHtml(stage)} ${escapeHtml(status)}" aria-haspopup="true"><span class="node-mark">${nodeMark(status)}</span><span class="node-name">${escapeHtml(stage)}</span><span class="node-status">${escapeHtml(status)}</span></button>`;
    }).join("");
    return `<div class="pipeline-row"><div class="pipeline-model">${badge(model.status)}<span class="mono">${escapeHtml(model.model_id)}</span></div><div class="pipeline-nodes">${nodes}</div></div>`;
  }).join("");
}

function openNodeMenu(button) {
  const menuKey = `${button.dataset.model}/${button.dataset.stage}`;
  if (state.nodeMenu && state.nodeMenuKey === menuKey) { closeNodeMenu(); return; }
  closeNodeMenu();
  const status = String(button.dataset.status || "PENDING").toUpperCase();
  const jobStatus = String(button.dataset.jobstatus || "").toUpperCase();
  const job = button.dataset.job;
  const control = canControlPipeline();
  const active = ACTIVE_JOB.has(jobStatus) || ["RUNNING", "QUEUED"].includes(status);
  const data = `data-model="${escapeHtml(button.dataset.model)}" data-stage="${escapeHtml(button.dataset.stage)}"`;
  const isPublish = button.dataset.stage === "publish";
  const items = [];
  if (control && active && job) items.push(`<button type="button" class="button danger" data-node-action="stop" data-job="${escapeHtml(job)}" ${data}>Stop</button>`);
  // Publish is never driven by a plain stage re-run: for most runs the publish
  // lane is not scheduled (``upload_enabled`` was false when the plan froze), so
  // ``stages/publish/rerun`` 400s. The on-demand publisher (preview -> confirm)
  // is the real path and works regardless, so route the publish node to it.
  if (isPublish) {
    if (canPublish()) items.push(`<button type="button" class="button" data-node-action="publish" ${data}>Publish…</button>`);
  } else if (control && !active) {
    items.push(`<button type="button" class="button" data-node-action="rerun" ${data}>${["PENDING", "PLANNED", "NOT_REQUESTED"].includes(status) ? "Run from here" : "Re-run from here"}</button>`);
  }
  // evaluate/publish each own a single-stage job, so they can be skipped live;
  // validate is bundled with quantize and is plan-time-only (no button here).
  const skippable = ["evaluate", "publish"].includes(button.dataset.stage);
  const terminalDone = ["PASS", "SUCCESS", "SUCCEEDED", "SKIPPED"].includes(status);
  if (control && skippable && !terminalDone) items.push(`<button type="button" class="button" data-node-action="skip" ${data}>Skip stage</button>`);
  items.push(`<button type="button" class="button ghost" data-node-action="log" ${data}>View log</button>`);
  const menu = document.createElement("div");
  menu.className = "node-menu";
  menu.innerHTML = `<p class="node-menu-title">${escapeHtml(button.dataset.stage)} · ${escapeHtml(status)}</p>${items.join("")}`;
  const row = button.closest(".pipeline-row");
  row.appendChild(menu);
  menu.style.left = `${Math.min(button.offsetLeft, row.clientWidth - menu.offsetWidth - 4)}px`;
  menu.style.top = `${button.offsetTop + button.offsetHeight + 6}px`;
  state.nodeMenu = menu;
  state.nodeMenuKey = menuKey;
}

function closeNodeMenu() {
  if (state.nodeMenu) state.nodeMenu.remove();
  state.nodeMenu = null;
  state.nodeMenuKey = null;
}

async function nodeAction(action, data) {
  if (action === "log") {
    $("#log-model").value = data.model;
    $("#log-stage").value = data.stage;
    updateAttempts();
    loadLog();
    startLogFollow();
    $("#log-output").scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  const runId = encodeURIComponent(state.run.run_id);
  const model = encodeURIComponent(data.model);
  if (action === "publish") {
    await publishFlow(data.model);
    return;
  }
  if (action === "stop") {
    const reason = await confirmAction("Stop stage", `Stop the ${data.stage} job for ${data.model}? A running stage is terminated and its reservation released.`, true);
    if (reason === null) return;
    await api(`/api/runs/${runId}/models/${model}/jobs/${encodeURIComponent(data.job)}/cancel`, { method: "POST", body: JSON.stringify({ reason, idempotency_key: `ui-stop-${key()}` }) });
    notify(`Stop requested for ${data.stage} · ${data.model}.`);
  } else if (action === "rerun") {
    let extra = {};
    if (data.stage === "runtime-smoke" || data.stage === "evaluate") {
      let src = null;
      try { src = await api(`/api/runs/${runId}/plan-request`); } catch (error) { src = null; }
      const cfg = await openRerunDialog(data.stage, data.model, src);
      if (cfg === null) return;
      extra = cfg;
    } else {
      const confirmed = await confirmAction("Re-run from stage", `Create a new attempt for ${data.model} starting at ${data.stage}?`);
      if (!confirmed) return;
    }
    await api(`/api/runs/${runId}/models/${model}/stages/${encodeURIComponent(data.stage)}/rerun`, { method: "POST", body: JSON.stringify({ ...extra, idempotency_key: `ui-rerun-${key()}` }) });
    notify(`Re-run from ${data.stage} queued for ${data.model}.`);
  } else if (action === "skip") {
    const confirmed = await confirmAction("Skip stage", `Skip ${data.stage} for ${data.model}? Any queued or running job for it is canceled and the stage is recorded as SKIPPED.`);
    if (!confirmed) return;
    await api(`/api/runs/${runId}/models/${model}/stages/${encodeURIComponent(data.stage)}/skip`, { method: "POST", body: JSON.stringify({ idempotency_key: `ui-skip-${key()}` }) });
    notify(`${data.stage} skipped for ${data.model}.`);
  }
  await loadRun(state.run.run_id);
}

function renderJobs() {
  const container = $("#job-lanes");
  const laneCard = (job) => `<article class="lane ${failureStatus(job.status) ? "failed" : ""}"><h3>${escapeHtml(job.kind)} ${badge(job.status)}</h3><dl><dt>Queue</dt><dd>${escapeHtml(text(job.queue))}</dd><dt>GPU</dt><dd>${escapeHtml(text(job.gpu_count, "0"))} · ${escapeHtml(formatNumber(job.gpu_hours))} h</dd><dt>Attempt</dt><dd class="mono">${escapeHtml(short(job.attempt_id, 18))}</dd><dt>Stages</dt><dd>${escapeHtml((job.stages || []).join(" → "))}</dd><dt>Exit</dt><dd>${escapeHtml(text(job.exit_code))}</dd>${failureStatus(job.status) ? `<dt>Reason</dt><dd class="job-failure">${escapeHtml(job.failure_class || job.details?.reason_code || "FAILED")} · ${escapeHtml(job.details?.message || "Open the stage log for details.")}</dd>` : ""}</dl></article>`;
  if (!state.jobs.length) {
    container.innerHTML = '<p class="empty">No execution jobs recorded.</p>';
  } else {
    // Jobs arrive newest-first. Show the latest attempt's lanes by default and
    // fold every earlier attempt behind a disclosure so a run with a long
    // re-run history doesn't flood the page. The open state is kept in `state`
    // so the 5s live refresh never collapses a toolbar the operator opened.
    const latestAttempt = state.jobs[0].attempt_id;
    const latest = state.jobs.filter((job) => job.attempt_id === latestAttempt);
    const older = state.jobs.filter((job) => job.attempt_id !== latestAttempt);
    const attempts = new Set(older.map((job) => job.attempt_id)).size;
    const olderBlock = older.length
      ? `<details class="lane-more" ${state.lanesExpanded ? "open" : ""}><summary>${older.length} earlier job${older.length === 1 ? "" : "s"} · ${attempts} attempt${attempts === 1 ? "" : "s"}</summary><div class="lanes">${older.map(laneCard).join("")}</div></details>`
      : "";
    container.innerHTML = `<div class="lanes">${latest.map(laneCard).join("")}</div>${olderBlock}`;
    const more = container.querySelector(".lane-more");
    if (more) more.addEventListener("toggle", () => { state.lanesExpanded = more.open; });
  }
  renderFailures();
}

function openFailureLog(button) {
  $("#log-model").value = button.dataset.failureModel;
  $("#log-stage").value = button.dataset.failureStage;
  updateAttempts();
  if (button.dataset.failureAttempt) $("#log-attempt").value = button.dataset.failureAttempt;
  loadLog();
  startLogFollow();
  $("#log-output").scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderModels() {
  $("#model-list").innerHTML = (state.run.models || []).map((model) => {
    const attempts = (model.attempts || []).map((attempt) => `${short(attempt.attempt_id, 18)} · ${attempt.status}`).join(" | ");
    const report = model.report || {};
    const metrics = Array.isArray(report.metrics) ? report.metrics : [];
    const scores = metrics.length
      ? `<div class="eval-scores"><p class="eyebrow">Evaluation scores · review manually</p><table class="eval-score-table"><thead><tr><th>Metric</th><th>Score</th></tr></thead><tbody>${metrics.map((metric) => `<tr><td>${escapeHtml(metric.name)}</td><td><code>${escapeHtml(formatNumber(metric.compressed_value))}</code></td></tr>`).join("")}</tbody></table></div>`
      : "";
    return `<article class="model-card"><div class="model-head"><div><h3>${escapeHtml(model.model_id)}</h3><span class="muted">${escapeHtml(attempts || "No attempt history")}</span></div>${badge(model.status)}</div><div class="fingerprints"><span>Artifact <code>${escapeHtml(model.artifact_fingerprint || "not finalized")}</code></span><span>Evaluation <code>${escapeHtml(model.evaluation_fingerprint || "not finalized")}</code></span><span>Quality report <code>${escapeHtml(report.status || "not available")}</code></span></div>${scores}</article>`;
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
  // A stage writes its state only when it finishes, so an in-flight stage has a
  // log but no recorded state yet. Always include the current (active) attempt
  // so a running stage's log is selectable, plus any attempt that recorded the
  // stage. Order a stage that is actively RUNNING first (that is the live log
  // the operator wants by default), then the current attempt, then by recency.
  const isRunning = (attempt) => String((attempt.stages || {})[stage] || "").toUpperCase() === "RUNNING";
  const attempts = allAttempts
    .filter((attempt) => attempt.current || Object.prototype.hasOwnProperty.call(attempt.stages || {}, stage))
    .sort((a, b) => (isRunning(b) ? 1 : 0) - (isRunning(a) ? 1 : 0) || (b.current ? 1 : 0) - (a.current ? 1 : 0));
  const selectedAttempt = $("#log-attempt").value;
  $("#log-attempt").innerHTML = attempts.length ? attempts.map((attempt) => `<option>${escapeHtml(attempt.attempt_id)}</option>`).join("") : '<option value="">No recorded attempt</option>';
  if (attempts.some((attempt) => attempt.attempt_id === selectedAttempt)) $("#log-attempt").value = selectedAttempt;
}

async function loadLog(options = {}) {
  const out = $("#log-output");
  const atBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 24;
  const runId = encodeURIComponent(state.run.run_id);
  const model = encodeURIComponent($("#log-model").value);
  const query = new URLSearchParams({ stage: $("#log-stage").value, attempt_id: $("#log-attempt").value, limit: "1000", tail: "1" });
  if (!options.quiet) out.textContent = "Loading log…";
  try {
    const result = await api(`/api/runs/${runId}/models/${model}/logs?${query}`);
    const items = result.items || [];
    const body = items.map((item) => item.text).join("\n") || "No log lines found.";
    const total = Number(result.total || 0);
    const shown = items.length;
    const banner = total > shown ? `… showing last ${shown} of ${total} lines (tailing) …\n` : "";
    out.textContent = banner + body;
    // Keep following the tail while the reader is already at the bottom (or on
    // an explicit load) so a running stage's log streams in without a jump.
    if (atBottom || !options.quiet) out.scrollTop = out.scrollHeight;
  } catch (error) {
    if (!options.quiet) out.textContent = error.message;
  }
}

function startLogFollow() {
  stopLogFollow();
  if (!state.logFollow) return;
  state.logTimer = window.setInterval(() => {
    if (!location.hash.startsWith("#run/") || !$("#log-model").value) return;
    loadLog({ quiet: true });
  }, 3000);
}

function stopLogFollow() {
  if (state.logTimer) window.clearInterval(state.logTimer);
  state.logTimer = null;
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
  stopLogFollow();
  stopGpuPolling();
}

async function previewPlan(event) {
  event.preventDefault();
  setPlanFeedback();
  clearPlanFieldErrors();
  const form = new FormData(event.currentTarget);
  const script = (form.get("script") || "").trim();
  if (!script) {
    showPlanError(Object.assign(new Error("A quantization script path is required."), { fields: ["script"] }));
    return;
  }
  const stages = ["quantize"];
  if ($("#plan-form [name=stage_validate]").checked) stages.push("validate");
  if ($("#plan-form [name=stage_inference]").checked) stages.push("inference");
  if ($("#plan-form [name=stage_evaluate]").checked) stages.push("evaluate");

  const request = { script, stages, parameters: collectScriptParams() };
  const manual = lines(form.get("manual_argv"));
  if (!$("#manual-argv-label").classList.contains("hidden") && manual.length) request.manual_argv = manual;
  const uploadPrefix = (form.get("upload_remote_prefix") || "").trim();
  if (uploadPrefix) request.upload = { remote_prefix: uploadPrefix };

  request.resources = {
    gpu_count: Number(form.get("gpu_count")) || 1,
    estimated_gpu_hours: Number(form.get("estimated_gpu_hours")) || 1,
  };
  if (form.get("estimated_eval_gpu_hours")) request.resources.estimated_eval_gpu_hours = Number(form.get("estimated_eval_gpu_hours"));
  if (form.get("max_gpu_hours")) request.max_gpu_hours = Number(form.get("max_gpu_hours"));

  if (stages.includes("inference")) {
    const missing = [];
    if (!form.get("container_name")) missing.push(["container_name", "inference container"]);
    if (!form.get("inference_script")) missing.push(["inference_script", "inference script"]);
    if (missing.length) {
      const error = new Error(`Required fields missing: ${missing.map((item) => item[1]).join(", ")}`);
      error.fields = missing.map((item) => item[0]);
      showPlanError(error);
      return;
    }
    request.inference = { container_name: form.get("container_name"), script: form.get("inference_script"), port: Number(form.get("inference_port")) || 8025, arguments: lines(form.get("inference_arguments")), result_file: "{reports_dir}/runtime-smoke-output.json" };
  }
  if (stages.includes("evaluate")) {
    request.evaluation_tool = form.get("evaluation_tool");
    request.evaluation_dataset = form.get("evaluation_dataset");
    const override = lines(form.get("evaluation_command"));
    if (override.length) request.evaluation_command = override;
  }

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

async function applyCloneRequest(req) {
  const form = $("#plan-form");
  form.elements.script.value = req.script || "";
  const res = req.resources || {};
  if (res.gpu_count != null) form.elements.gpu_count.value = res.gpu_count;
  if (res.estimated_gpu_hours != null) form.elements.estimated_gpu_hours.value = res.estimated_gpu_hours;
  if (res.estimated_eval_gpu_hours != null) form.elements.estimated_eval_gpu_hours.value = res.estimated_eval_gpu_hours;
  if (req.max_gpu_hours != null) form.elements.max_gpu_hours.value = req.max_gpu_hours;
  const stages = new Set(req.stages || []);
  if (form.elements.stage_validate) form.elements.stage_validate.checked = stages.has("validate");
  if (form.elements.stage_inference) form.elements.stage_inference.checked = stages.has("inference");
  if (form.elements.stage_evaluate) form.elements.stage_evaluate.checked = stages.has("evaluate");
  const inf = req.inference || {};
  if (inf.container_name) form.elements.container_name.value = inf.container_name;
  if (inf.script) form.elements.inference_script.value = inf.script;
  if (inf.port != null) form.elements.inference_port.value = inf.port;
  if (Array.isArray(inf.arguments)) form.elements.inference_arguments.value = inf.arguments.join("\n");
  if (req.evaluation_tool) form.elements.evaluation_tool.value = req.evaluation_tool;
  updateStageSections();
  updateEvalDatasets();
  if (req.evaluation_dataset) form.elements.evaluation_dataset.value = req.evaluation_dataset;
  if (Array.isArray(req.evaluation_command)) form.elements.evaluation_command.value = req.evaluation_command.join("\n");
  if (req.script) {
    await loadScriptParams();
    for (const p of (req.parameters || [])) {
      const el = $(`#script-params [data-param="${p.flag}"]`);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = Boolean(p.value);
      else el.value = p.value != null ? p.value : "";
    }
  }
  setPlanFeedback("Cloned from a previous run — edit any field, then Preview plan.");
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
  if (action === "clone") {
    try {
      state.cloneRequest = await api(`/api/runs/${runId}/plan-request`);
      notify("Loaded this run's config into New plan — edit and preview.");
      location.hash = "plan";
    } catch (error) {
      notify(error.message || "This run has no clonable config.", true);
    }
    return;
  }
  if (action === "backup-preview") return showPreview(await api(`/api/runs/${runId}/backup/preview`, { method: "POST", body: "{}" }), "Backup preview");
  if (action === "publish-preview") return publishFlow(null);
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

async function publishFlow(modelId) {
  const runId = encodeURIComponent(state.run.run_id);
  const previewPath = modelId
    ? `/api/runs/${runId}/models/${encodeURIComponent(modelId)}/publish/preview`
    : `/api/runs/${runId}/publish/preview`;
  let preview;
  try {
    preview = await api(previewPath, { method: "POST", body: "{}" });
  } catch (error) {
    notify(error.message || "Publish preview failed.", true);
    return;
  }
  const target = modelId || preview.model_id;
  const decision = await publishConfirmDialog(preview);
  if (!decision) return;
  const body = { confirm: true, idempotency_key: `ui-publish-${key()}` };
  if ((preview.approvals_required || 1) >= 2) body.approvals = decision.approvals;
  try {
    const result = await api(`/api/runs/${runId}/models/${encodeURIComponent(target)}/publish`, { method: "POST", body: JSON.stringify(body) });
    notify(`Publish queued for ${target} → ${result.remote_target || preview.remote_target}.`);
  } catch (error) {
    notify(error.message || "Publish failed.", true);
  }
  await loadRun(state.run.run_id);
}

function publishConfirmDialog(preview) {
  return new Promise((resolve) => {
    const dialog = $("#confirm-dialog");
    const submit = $("#confirm-submit");
    $("#confirm-title").textContent = "Publish artifact";
    const gib = (Number(preview.total_bytes || 0) / 1024 ** 3).toFixed(2);
    const errors = (preview.risks || []).filter((r) => r.severity === "error");
    const warns = (preview.risks || []).filter((r) => r.severity === "warning");
    const lines = [
      `Model:    ${preview.model_id}`,
      `Target:   ${preview.remote_target}`,
      `Files:    ${preview.file_count} · ${gib} GiB`,
      `Artifact: ${preview.identity?.artifact_fingerprint || "n/a"}`,
    ];
    if ((preview.approvals_required || 1) >= 2) lines.push(`Approvals: ${preview.approvals_required} distinct approvers required`);
    if (warns.length) lines.push("", "Warnings:", ...warns.map((r) => `• ${r.message}`));
    if (!preview.ready) lines.push("", "Blocked — cannot publish:", ...errors.map((r) => `• ${r.message}`));
    const message = $("#confirm-message");
    message.style.whiteSpace = "pre-wrap";
    message.textContent = lines.join("\n");
    $("#confirm-reason-label").classList.add("hidden");
    submit.textContent = "Publish";
    submit.classList.toggle("hidden", !preview.ready);
    dialog.showModal();
    dialog.addEventListener("close", () => {
      submit.textContent = "Confirm";
      submit.classList.remove("hidden");
      if (dialog.returnValue !== "default" || !preview.ready) return resolve(null);
      let approvals;
      if ((preview.approvals_required || 1) >= 2) {
        const raw = window.prompt("This deployment requires two distinct approvers. Enter two names, comma-separated:");
        approvals = (raw || "").split(",").map((s) => s.trim()).filter(Boolean);
        if (new Set(approvals).size < 2) {
          notify("Two distinct approvers are required.", true);
          return resolve(null);
        }
      }
      resolve({ approvals });
    }, { once: true });
  });
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

function populateRerunDatasets() {
  const tool = $("#rerun-eval-tool").value || "lm_eval";
  const select = $("#rerun-eval-dataset");
  const previous = select.value;
  const datasets = EVAL_PRESETS[tool] || [];
  select.innerHTML = datasets.map((d) => `<option>${escapeHtml(d)}</option>`).join("");
  if (datasets.includes(previous)) select.value = previous;
}

function openRerunDialog(stage, model, src) {
  return new Promise((resolve) => {
    const dialog = $("#rerun-dialog");
    const isInference = stage === "runtime-smoke";
    const inf = (src && src.inference) || {};
    $("#rerun-title").textContent = `Re-run ${stage}`;
    $("#rerun-sub").textContent = isInference
      ? `Configure the prestarted container and smoke script for ${model}.`
      : `Choose the evaluation tool and dataset for ${model} (or paste a command).`;
    $('[data-rerun-group="inference"]', dialog).classList.toggle("hidden", !isInference);
    $('[data-rerun-group="evaluate"]', dialog).classList.toggle("hidden", isInference);
    // Pre-fill from the run's original config so a re-run is one click.
    $("#rerun-container").value = inf.container_name || "";
    $("#rerun-script").value = inf.script || "";
    $("#rerun-port").value = inf.port != null ? inf.port : 8025;
    $("#rerun-args").value = Array.isArray(inf.arguments) ? inf.arguments.join("\n") : "";
    $("#rerun-eval-command").value = Array.isArray(src && src.evaluation_command) ? src.evaluation_command.join("\n") : "";
    if (!isInference) {
      if (src && src.evaluation_tool) $("#rerun-eval-tool").value = src.evaluation_tool;
      populateRerunDatasets();
      if (src && src.evaluation_dataset) $("#rerun-eval-dataset").value = src.evaluation_dataset;
    }
    dialog.showModal();
    dialog.addEventListener("close", () => {
      if (dialog.returnValue !== "default") { resolve(null); return; }
      if (isInference) {
        const container = $("#rerun-container").value.trim();
        const script = $("#rerun-script").value.trim();
        if (!container || !script) {
          notify("Container and script are required for runtime-smoke.", true);
          resolve(null);
          return;
        }
        resolve({ inference: { container_name: container, script, port: Number($("#rerun-port").value) || 8025, arguments: lines($("#rerun-args").value), result_file: "{reports_dir}/runtime-smoke-output.json" } });
      } else {
        const body = { evaluation_tool: $("#rerun-eval-tool").value, evaluation_dataset: $("#rerun-eval-dataset").value };
        const cmd = lines($("#rerun-eval-command").value);
        if (cmd.length) body.evaluation_command = cmd;
        resolve(body);
      }
    }, { once: true });
  });
}

function gib(bytes) {
  return Number.isFinite(Number(bytes)) ? (Number(bytes) / 1073741824).toFixed(1) : "—";
}

function renderGpus(data) {
  const meta = $("#gpu-meta");
  const cards = $("#gpu-cards");
  const parts = [];
  if (data.hostname) parts.push(`host ${data.hostname}`);
  if (data.torch_version) parts.push(`torch ${data.torch_version}`);
  if (data.cuda_version) parts.push(`CUDA ${data.cuda_version}`);
  meta.textContent = data.available ? `${data.device_count} device(s) · ${parts.join(" · ")}` : (parts.join(" · ") || "No accelerator");
  meta.className = `connection ${data.available ? "" : "pending"}`;
  if (!data.available || !(data.devices || []).length) {
    cards.innerHTML = `<div class="gpu-banner">${escapeHtml(data.reason || "No GPU devices are visible to this server.")}</div>`;
    return;
  }
  cards.innerHTML = data.devices.map((device) => {
    const total = Number(device.total_memory_bytes || 0);
    const used = Number(device.used_memory_bytes ?? (total - Number(device.free_memory_bytes || 0)));
    const percent = total > 0 ? Math.min(100, Math.round(100 * used / total)) : 0;
    const fill = percent >= 90 ? "full" : percent >= 70 ? "warn" : "";
    const util = device.utilization_percent;
    const props = [];
    if (device.compute_capability) props.push(`<span>Compute <b>${escapeHtml(device.compute_capability)}</b></span>`);
    if (device.multiprocessors) props.push(`<span>SMs <b>${escapeHtml(device.multiprocessors)}</b></span>`);
    if (Number.isFinite(Number(device.reserved_bytes))) props.push(`<span>Reserved <b>${gib(device.reserved_bytes)} GiB</b></span>`);
    return `<article class="gpu-card">
      <header><h3>${escapeHtml(device.name || `cuda:${device.index}`)}</h3><span class="gpu-index">#${escapeHtml(device.index)}</span></header>
      <div class="gpu-util"><strong>${util === null || util === undefined ? "—" : escapeHtml(util) + "%"}</strong><span>GPU utilization</span></div>
      <div><div class="gpu-mem-head"><span>Memory</span><span>${gib(used)} / ${gib(total)} GiB · ${percent}%</span></div><div class="gpu-mem-bar"><i class="${fill}" style="width:${percent}%"></i></div></div>
      <div class="gpu-props">${props.join("") || '<span class="muted">No device properties reported</span>'}</div>
    </article>`;
  }).join("");
}

async function loadGpus() {
  try {
    renderGpus(await api("/api/ops/gpus"));
  } catch (error) {
    $("#gpu-meta").textContent = "Unavailable";
    $("#gpu-cards").innerHTML = `<div class="gpu-banner">${escapeHtml(error.message)}</div>`;
  }
}

function startGpuPolling() {
  stopGpuPolling();
  state.opsTimer = window.setInterval(() => {
    if (location.hash.slice(1) !== "operations") { stopGpuPolling(); return; }
    loadGpus();
  }, 5000);
}

function stopGpuPolling() {
  if (state.opsTimer) window.clearInterval(state.opsTimer);
  state.opsTimer = null;
}

async function loadOperations() {
  loadGpus();
  startGpuPolling();
  try {
    const [cost, capabilitiesResult, schedulesResult] = await Promise.all([
      api("/api/trends/cost"),
      api("/api/capabilities").then((value) => ({ value })).catch((error) => ({ error })),
      api("/api/ops/schedules").then((value) => ({ value })).catch((error) => ({ error })),
    ]);
    if (capabilitiesResult.error) {
      $("#capacity-cards").innerHTML = metric("Capacity", "Unavailable", "Reload or check the server");
      $("#capacity-detail").innerHTML = `<p class="empty">${escapeHtml(capabilitiesResult.error.message)}</p>`;
    }
    if (schedulesResult.error) {
      $("#schedule-list").innerHTML = `<p class="empty">${escapeHtml(schedulesResult.error.message)}</p>`;
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
  updateRoleHelp();
  $("#identity-dialog").showModal();
}

function updateRoleHelp() {
  $("#role-help").textContent = ROLE_HELP[$("#role-input").value] || "";
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
$("#refresh-ops").addEventListener("click", loadOperations);
$("#plan-form").addEventListener("submit", previewPlan);
$("#load-params").addEventListener("click", loadScriptParams);
$("#plan-form [name=stage_inference]").addEventListener("change", updateStageSections);
$("#plan-form [name=stage_evaluate]").addEventListener("change", updateStageSections);
$("#plan-form [name=evaluation_tool]").addEventListener("change", updateEvalDatasets);
$("#rerun-eval-tool").addEventListener("change", populateRerunDatasets);
$("#launch-plan").addEventListener("click", launchPlan);
$("#identity-button").addEventListener("click", openIdentity);
$("#role-input").addEventListener("change", updateRoleHelp);
$("#save-identity").addEventListener("click", saveIdentity);
$("#log-model").addEventListener("change", () => { updateAttempts(); loadLog(); });
$("#log-stage").addEventListener("change", () => { updateAttempts(); loadLog(); });
$("#log-attempt").addEventListener("change", () => loadLog());
$("#load-log").addEventListener("click", () => { loadLog(); startLogFollow(); });
$("#log-follow").addEventListener("change", (event) => {
  state.logFollow = event.target.checked;
  if (state.logFollow) { loadLog({ quiet: true }); startLogFollow(); } else stopLogFollow();
});
$("#run-actions").addEventListener("click", async (event) => { const action = event.target.closest("[data-action]")?.dataset.action; if (action) try { await runAction(action); } catch (error) { notify(error.message, true); } });
$("#pipeline").addEventListener("click", async (event) => {
  const actionButton = event.target.closest("[data-node-action]");
  if (actionButton) {
    closeNodeMenu();
    try { await nodeAction(actionButton.dataset.nodeAction, actionButton.dataset); } catch (error) { notify(error.message, true); }
    return;
  }
  const node = event.target.closest(".node");
  if (node) openNodeMenu(node); else closeNodeMenu();
});
document.addEventListener("click", (event) => { if (!event.target.closest("#pipeline")) closeNodeMenu(); }, true);
document.addEventListener("click", (event) => { const target = event.target.closest("[data-go]"); if (target) location.hash = target.dataset.go; });
$("#failure-list").addEventListener("click", (event) => { const button = event.target.closest(".failure-log"); if (button) openFailureLog(button); });
bootstrap();
