const state = {
  tasks: [],
  currentTask: null,
  events: [],
  busy: false,
};

const $ = (id) => document.getElementById(id);

const els = {
  apiStatus: $("apiStatus"),
  taskForm: $("taskForm"),
  goalInput: $("goalInput"),
  imageT1Input: $("imageT1Input"),
  imageT2Input: $("imageT2Input"),
  autoConfirmInput: $("autoConfirmInput"),
  taskList: $("taskList"),
  refreshTasksBtn: $("refreshTasksBtn"),
  currentTitle: $("currentTitle"),
  currentStatus: $("currentStatus"),
  interruptPanel: $("interruptPanel"),
  approveBtn: $("approveBtn"),
  planList: $("planList"),
  planConfidence: $("planConfidence"),
  previewBox: $("previewBox"),
  areaSummary: $("areaSummary"),
  artifactList: $("artifactList"),
  artifactCount: $("artifactCount"),
  contextList: $("contextList"),
  contextCount: $("contextCount"),
  eventList: $("eventList"),
  refreshDetailBtn: $("refreshDetailBtn"),
  feedbackForm: $("feedbackForm"),
  ratingInput: $("ratingInput"),
  acceptedInput: $("acceptedInput"),
  commentInput: $("commentInput"),
  feedbackStatus: $("feedbackStatus"),
  toast: $("toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
      // keep status text
    }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.add("hidden"), 2600);
}

function setBusy(value) {
  state.busy = value;
  els.approveBtn.disabled = value;
  els.taskForm.querySelector("button[type='submit']").disabled = value;
}

async function checkHealth() {
  try {
    await api("/health");
    els.apiStatus.textContent = "已连接";
  } catch (error) {
    els.apiStatus.textContent = "未连接";
  }
}

async function loadTasks(selectLatest = false) {
  const tasks = await api("/api/tasks");
  state.tasks = tasks.sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)));
  renderTaskList();
  if (selectLatest && state.tasks.length) {
    await selectTask(state.tasks[0].task_id);
  } else if (state.currentTask) {
    const exists = state.tasks.some((task) => task.task_id === state.currentTask.task_id);
    if (exists) {
      await selectTask(state.currentTask.task_id, false);
    }
  }
}

async function selectTask(taskId, rerenderList = true) {
  const task = await api(`/api/tasks/${taskId}`);
  const events = await api(`/api/tasks/${taskId}/events`);
  state.currentTask = task;
  state.events = events;
  renderCurrentTask();
  if (rerenderList) renderTaskList();
}

function renderTaskList() {
  if (!state.tasks.length) {
    els.taskList.innerHTML = `<div class="empty-state">暂无任务</div>`;
    return;
  }
  els.taskList.innerHTML = state.tasks
    .map((task) => {
      const active = state.currentTask && state.currentTask.task_id === task.task_id ? " active" : "";
      return `
        <button class="task-item${active}" data-task-id="${task.task_id}" type="button">
          <span class="task-title">${escapeHtml(task.title || task.user_goal)}</span>
          <span class="task-meta">
            <span>${formatDate(task.updated_at)}</span>
            <span class="badge ${task.status}">${task.status}</span>
          </span>
        </button>`;
    })
    .join("");
  els.taskList.querySelectorAll("[data-task-id]").forEach((button) => {
    button.addEventListener("click", () => selectTask(button.dataset.taskId));
  });
}

function renderCurrentTask() {
  const task = state.currentTask;
  if (!task) return;
  els.currentTitle.textContent = task.title || task.user_goal;
  els.currentStatus.textContent = task.status;
  els.currentStatus.className = `status-pill status-${task.status}`;

  const openInterrupt = (task.interrupts || []).find((item) => item.status === "open");
  els.interruptPanel.classList.toggle("hidden", !openInterrupt);
  els.approveBtn.dataset.interruptId = openInterrupt ? openInterrupt.interrupt_id : "";

  renderPlan(task);
  renderPreview(task);
  renderArtifacts(task);
  renderContext(task);
  renderEvents();
}

function renderPlan(task) {
  const steps = task.plan || [];
  const confidence = task.plan_summary ? Math.round((task.plan_summary.confidence || 0) * 100) : 0;
  els.planConfidence.textContent = confidence ? `${confidence}%` : "";
  if (!steps.length) {
    els.planList.innerHTML = `<div class="empty-state">暂无计划</div>`;
    return;
  }
  els.planList.innerHTML = steps
    .map(
      (step, index) => `
      <div class="step">
        <div class="step-index">${index + 1}</div>
        <div>
          <div class="step-name">${escapeHtml(step.name)}</div>
          <div class="step-tool">${escapeHtml(step.tool_name || "")}</div>
        </div>
        <span class="badge ${step.status}">${step.status}</span>
      </div>`
    )
    .join("");
}

function renderPreview(task) {
  const preview = artifactByAlias(task, "change_preview");
  const area = task.working_memory?.area_statistics?.summary;
  els.areaSummary.textContent = area ? `${area.area_ha} 公顷` : "";
  if (!preview) {
    els.previewBox.innerHTML = "暂无预览";
    els.previewBox.classList.add("empty-state");
    return;
  }
  els.previewBox.classList.remove("empty-state");
  els.previewBox.innerHTML = `<img src="${artifactFileUrl(task.task_id, preview.artifact_id)}" alt="变化预览" />`;
}

function renderArtifacts(task) {
  const artifacts = Object.values(task.artifacts || {});
  els.artifactCount.textContent = artifacts.length ? `${artifacts.length} 个` : "";
  if (!artifacts.length) {
    els.artifactList.innerHTML = `<div class="empty-state">暂无产物</div>`;
    return;
  }
  els.artifactList.innerHTML = artifacts
    .map((artifact) => {
      const label = artifact.alias || artifact.artifact_id;
      return `
        <div class="artifact">
          <div>
            <div class="artifact-name">${escapeHtml(label)}</div>
            <div class="artifact-uri">${escapeHtml(artifact.type)} · ${escapeHtml(shortPath(artifact.uri))}</div>
          </div>
          <a href="${artifactFileUrl(task.task_id, artifact.artifact_id)}" target="_blank" rel="noreferrer">
            <button type="button">打开</button>
          </a>
        </div>`;
    })
    .join("");
}

function renderContext(task) {
  const chunks = task.retrieved_context || [];
  els.contextCount.textContent = chunks.length ? `${chunks.length} 条` : "";
  if (!chunks.length) {
    els.contextList.innerHTML = `<div class="empty-state">暂无依据</div>`;
    return;
  }
  els.contextList.innerHTML = chunks
    .map(
      (chunk) => `
      <div class="context">
        <div class="context-title">${escapeHtml(chunk.title)}</div>
        <div class="context-body">${escapeHtml(chunk.content)}</div>
      </div>`
    )
    .join("");
}

function renderEvents() {
  const events = state.events || [];
  if (!events.length) {
    els.eventList.innerHTML = `<div class="empty-state">暂无事件</div>`;
    return;
  }
  els.eventList.innerHTML = events
    .slice()
    .reverse()
    .map(
      (event) => `
      <div class="event">
        <div class="event-type">${escapeHtml(event.event_type)}</div>
        <div class="event-time">${formatDate(event.created_at)}</div>
      </div>`
    )
    .join("");
}

async function createTask(event) {
  event.preventDefault();
  setBusy(true);
  try {
    const task = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify({
        user_goal: els.goalInput.value.trim(),
        image_t1_uri: els.imageT1Input.value.trim(),
        image_t2_uri: els.imageT2Input.value.trim(),
        auto_confirm: els.autoConfirmInput.checked,
      }),
    });
    showToast(task.status === "waiting_human" ? "计划已生成" : "任务已完成");
    await loadTasks();
    await selectTask(task.task_id);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function approveCurrentPlan() {
  const task = state.currentTask;
  const interruptId = els.approveBtn.dataset.interruptId;
  if (!task || !interruptId) return;
  setBusy(true);
  try {
    const updated = await api(`/api/tasks/${task.task_id}/interrupts/${interruptId}/approve`, {
      method: "POST",
      body: "{}",
    });
    state.currentTask = updated;
    state.events = await api(`/api/tasks/${task.task_id}/events`);
    renderCurrentTask();
    await loadTasks();
    showToast("任务执行完成");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function submitFeedback(event) {
  event.preventDefault();
  const task = state.currentTask;
  if (!task) return;
  setBusy(true);
  try {
    await api(`/api/tasks/${task.task_id}/feedback`, {
      method: "POST",
      body: JSON.stringify({
        rating: Number(els.ratingInput.value),
        accepted: els.acceptedInput.checked,
        comment: els.commentInput.value.trim(),
      }),
    });
    els.commentInput.value = "";
    els.feedbackStatus.textContent = "已写入";
    await selectTask(task.task_id);
    showToast("反馈已提交");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function artifactByAlias(task, alias) {
  const artifactId = task.artifact_refs ? task.artifact_refs[alias] : null;
  return artifactId ? task.artifacts[artifactId] : null;
}

function artifactFileUrl(taskId, artifactId) {
  return `/api/tasks/${taskId}/artifacts/${artifactId}/file`;
}

function shortPath(uri) {
  const normalized = String(uri || "").replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts.slice(-3).join("/");
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.taskForm.addEventListener("submit", createTask);
els.refreshTasksBtn.addEventListener("click", () => loadTasks());
els.refreshDetailBtn.addEventListener("click", () => {
  if (state.currentTask) selectTask(state.currentTask.task_id);
});
els.approveBtn.addEventListener("click", approveCurrentPlan);
els.feedbackForm.addEventListener("submit", submitFeedback);

checkHealth();
loadTasks(true).catch((error) => showToast(error.message));
