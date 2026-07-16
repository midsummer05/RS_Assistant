import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Check,
  Clock,
  Download,
  FileText,
  Layers,
  Map,
  MessageSquare,
  Play,
  RefreshCw,
  Satellite,
  Search,
  Send,
  Table2
} from "lucide-react";
import { api, artifactFileUrl } from "./api";
import type { Artifact, EventRecord, Interrupt, KnowledgeChunk, StepState, TaskState } from "./types";
import "./styles.css";

function App() {
  const [apiStatus, setApiStatus] = useState("连接中");
  const [tasks, setTasks] = useState<TaskState[]>([]);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");

  async function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  }

  async function refreshTasks(selectLatest = false) {
    const nextTasks = (await api.listTasks()).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    setTasks(nextTasks);
    if (selectLatest && nextTasks.length) {
      await selectTask(nextTasks[0].task_id);
    } else if (currentTask && nextTasks.some((task) => task.task_id === currentTask.task_id)) {
      await selectTask(currentTask.task_id);
    }
  }

  async function selectTask(taskId: string) {
    const [task, taskEvents] = await Promise.all([api.getTask(taskId), api.listEvents(taskId)]);
    setCurrentTask(task);
    setEvents(taskEvents);
  }

  async function createTask(payload: { goal: string; imageT1: string; imageT2: string; autoConfirm: boolean; agentMode: "workflow" | "agent"; executionBudget: number }) {
    setBusy(true);
    try {
      const task = await api.createTask({
        user_goal: payload.goal,
        image_t1_uri: payload.imageT1,
        image_t2_uri: payload.imageT2,
        auto_confirm: payload.autoConfirm,
        agent_mode: payload.agentMode,
        execution_budget: payload.agentMode === "agent" ? payload.executionBudget : undefined
      });
      await refreshTasks();
      await selectTask(task.task_id);
      showToast(task.status === "waiting_human" ? "计划已生成，等待确认" : "任务已完成");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "任务创建失败");
    } finally {
      setBusy(false);
    }
  }

  async function continueTask() {
    if (!currentTask) return;
    setBusy(true);
    try {
      const task =
        currentTask.status === "failed"
          ? await api.retryTask(currentTask.task_id)
          : await api.resumeTask(currentTask.task_id);
      setCurrentTask(task);
      setEvents(await api.listEvents(task.task_id));
      await refreshTasks();
      showToast(task.status === "paused" ? "本轮执行完成，可继续下一批" : "任务已继续执行");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "任务继续失败");
    } finally {
      setBusy(false);
    }
  }

  async function approvePlan(interrupt: Interrupt) {
    if (!currentTask) return;
    setBusy(true);
    try {
      const task = await api.approvePlan(currentTask.task_id, interrupt.interrupt_id);
      const taskEvents = await api.listEvents(task.task_id);
      setCurrentTask(task);
      setEvents(taskEvents);
      await refreshTasks();
      showToast("计划已批准并执行完成");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "计划批准失败");
    } finally {
      setBusy(false);
    }
  }

  async function submitFeedback(payload: { rating: number; accepted: boolean; comment: string }) {
    if (!currentTask) return;
    setBusy(true);
    try {
      const task = await api.submitFeedback(currentTask.task_id, payload);
      const taskEvents = await api.listEvents(task.task_id);
      setCurrentTask(task);
      setEvents(taskEvents);
      showToast("反馈已写入项目记忆");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "反馈提交失败");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    api
      .health()
      .then(() => setApiStatus("已连接"))
      .catch(() => setApiStatus("未连接"));
    refreshTasks(true).catch((error) => showToast(error instanceof Error ? error.message : "任务加载失败"));
  }, []);

  const openInterrupt = useMemo(
    () => currentTask?.interrupts.find((interrupt) => interrupt.status === "open"),
    [currentTask]
  );

  return (
    <div className="shell">
      <aside className="sidebar">
        <Brand status={apiStatus} />
        <TaskComposer busy={busy} onCreate={createTask} />
        <TaskList
          tasks={tasks}
          currentTaskId={currentTask?.task_id}
          onRefresh={() => refreshTasks()}
          onSelect={(taskId) => selectTask(taskId)}
        />
      </aside>

      <main className="workspace">
        <TaskHeader task={currentTask} />

        {currentTask && openInterrupt ? (
          <section className="confirm-band">
            <div>
              <p className="eyebrow">人工确认</p>
              <h3>{openInterrupt.reason}</h3>
            </div>
            <button className="primary" disabled={busy} onClick={() => approvePlan(openInterrupt)}>
              <Play size={16} />
              批准并执行
            </button>
          </section>
        ) : null}

        {currentTask && (currentTask.status === "paused" || currentTask.status === "failed") ? (
          <section className="confirm-band">
            <div>
              <p className="eyebrow">{currentTask.status === "paused" ? "长程任务暂停" : "任务失败"}</p>
              <h3>
                {currentTask.status === "paused"
                  ? "已保存 checkpoint，可从下一批步骤继续。"
                  : currentTask.working_memory?.last_failure?.message || "可重试最近失败阶段。"}
              </h3>
            </div>
            <button className="primary" disabled={busy} onClick={continueTask}>
              <Play size={16} />
              {currentTask.status === "paused" ? "继续执行" : "重试"}
            </button>
          </section>
        ) : null}

        <section className="grid">
          <PlanPanel task={currentTask} />
          <PreviewPanel task={currentTask} />
        </section>

        <section className="grid lower-grid">
          <ArtifactPanel task={currentTask} />
          <QualityPanel task={currentTask} />
        </section>

        <ContextPanel chunks={currentTask?.retrieved_context || []} />

        <section className="panel event-panel">
          <div className="section-head">
            <h2>事件流</h2>
            <button className="icon-button" type="button" title="刷新详情" onClick={() => currentTask && selectTask(currentTask.task_id)}>
              <RefreshCw size={16} />
            </button>
          </div>
          <EventList events={events} />
        </section>

        <FeedbackPanel disabled={!currentTask || busy} onSubmit={submitFeedback} />
      </main>

      {toast ? <div className="toast">{toast}</div> : null}
    </div>
  );
}

function Brand({ status }: { status: string }) {
  return (
    <div className="brand">
      <div className="brand-mark">
        <Satellite size={24} />
      </div>
      <div>
        <h1>遥感任务工作台</h1>
        <p>{status}</p>
      </div>
    </div>
  );
}

function TaskComposer({
  busy,
  onCreate
}: {
  busy: boolean;
  onCreate: (payload: { goal: string; imageT1: string; imageT2: string; autoConfirm: boolean; agentMode: "workflow" | "agent"; executionBudget: number }) => void;
}) {
  const [goal, setGoal] = useState("帮我对两期 Sentinel-2 影像做建设用地扩张变化检测，输出图斑、面积统计和报告。");
  const [imageT1, setImageT1] = useState("demo://image_t1");
  const [imageT2, setImageT2] = useState("demo://image_t2");
  const [autoConfirm, setAutoConfirm] = useState(false);
  const [agentMode, setAgentMode] = useState<"workflow" | "agent">("agent");
  const [executionBudget, setExecutionBudget] = useState(3);

  return (
    <form
      className="panel task-form"
      onSubmit={(event) => {
        event.preventDefault();
        onCreate({ goal, imageT1, imageT2, autoConfirm, agentMode, executionBudget });
      }}
    >
      <label>
        <span>任务目标</span>
        <textarea rows={5} value={goal} onChange={(event) => setGoal(event.target.value)} />
      </label>
      <div className="field-grid">
        <label>
          <span>第一期影像</span>
          <input value={imageT1} onChange={(event) => setImageT1(event.target.value)} />
        </label>
        <label>
          <span>第二期影像</span>
          <input value={imageT2} onChange={(event) => setImageT2(event.target.value)} />
        </label>
      </div>
      <div className="form-row">
        <label>
          <span>执行模式</span>
          <select value={agentMode} onChange={(event) => setAgentMode(event.target.value as "workflow" | "agent")}>
            <option value="agent">真实模型 Agent</option>
            <option value="workflow">固定 Workflow</option>
          </select>
        </label>
        {agentMode === "agent" ? (
          <label>
            <span>每轮步骤数</span>
            <input type="number" min={1} max={100} value={executionBudget} onChange={(event) => setExecutionBudget(Number(event.target.value))} />
          </label>
        ) : null}
      </div>
      <div className="form-row">
        <label className="toggle">
          <input type="checkbox" checked={autoConfirm} onChange={(event) => setAutoConfirm(event.target.checked)} />
          <span />
          自动执行
        </label>
        <button className="primary" type="submit" disabled={busy || !goal || !imageT1 || !imageT2}>
          <Send size={16} />
          创建任务
        </button>
      </div>
    </form>
  );
}

function TaskList({
  tasks,
  currentTaskId,
  onRefresh,
  onSelect
}: {
  tasks: TaskState[];
  currentTaskId?: string;
  onRefresh: () => void;
  onSelect: (taskId: string) => void;
}) {
  return (
    <section className="panel">
      <div className="section-head">
        <h2>任务</h2>
        <button className="icon-button" type="button" title="刷新任务" onClick={onRefresh}>
          <RefreshCw size={16} />
        </button>
      </div>
      <div className="task-list">
        {tasks.length ? (
          tasks.map((task) => (
            <button
              className={`task-item${currentTaskId === task.task_id ? " active" : ""}`}
              key={task.task_id}
              type="button"
              onClick={() => onSelect(task.task_id)}
            >
              <span className="task-title">{task.title || task.user_goal}</span>
              <span className="task-meta">
                <span>{formatDate(task.updated_at)}</span>
                <StatusBadge status={task.status} />
              </span>
            </button>
          ))
        ) : (
          <div className="empty-state">暂无任务</div>
        )}
      </div>
    </section>
  );
}

function TaskHeader({ task }: { task: TaskState | null }) {
  return (
    <section className="topbar">
      <div>
        <p className="eyebrow">当前任务</p>
        <h2>{task ? task.title || task.user_goal : "未选择任务"}</h2>
      </div>
      <div className={`status-pill status-${task?.status || "idle"}`}>{task?.status || "idle"}</div>
    </section>
  );
}

function PlanPanel({ task }: { task: TaskState | null }) {
  const confidence = task?.plan_summary?.confidence ? Math.round(task.plan_summary.confidence * 100) : 0;
  return (
    <div className="panel plan-panel">
      <div className="section-head">
        <h2>执行计划</h2>
        <span className="muted">{confidence ? `${confidence}%` : ""}</span>
      </div>
      <div className="step-list">
        {task?.plan.length ? task.plan.map((step, index) => <StepRow key={step.step_id} step={step} index={index} />) : <div className="empty-state">暂无计划</div>}
      </div>
    </div>
  );
}

function StepRow({ step, index }: { step: StepState; index: number }) {
  return (
    <div className="step">
      <div className="step-index">{index + 1}</div>
      <div>
        <div className="step-name">{step.name}</div>
        <div className="step-tool">{step.tool_name}</div>
      </div>
      <StatusBadge status={step.status} />
    </div>
  );
}

function PreviewPanel({ task }: { task: TaskState | null }) {
  const preview = task ? artifactByAlias(task, "change_preview") : null;
  const area = task?.working_memory?.area_statistics?.summary;
  const vector = task ? artifactByAlias(task, "change_vector") : null;
  return (
    <div className="panel preview-panel">
      <div className="section-head">
        <h2>结果预览</h2>
        <span className="muted">{area ? `${area.area_ha} 公顷` : ""}</span>
      </div>
      <div className="preview-box">
        {task && preview ? (
          <>
            <img src={artifactFileUrl(task.task_id, preview.artifact_id)} alt="变化预览" />
            {vector ? <VectorHint artifact={vector} /> : null}
          </>
        ) : (
          <div className="empty-state">暂无预览</div>
        )}
      </div>
    </div>
  );
}

function VectorHint({ artifact }: { artifact: Artifact }) {
  const featureCount = artifact.metadata?.feature_count as number | undefined;
  const totalArea = artifact.metadata?.total_area_m2 as number | undefined;
  return (
    <div className="vector-hint">
      <Map size={15} />
      <span>{featureCount ?? 0} 个图斑</span>
      <span>{totalArea ? `${Math.round(totalArea)} 平方米` : ""}</span>
    </div>
  );
}

function ArtifactPanel({ task }: { task: TaskState | null }) {
  const artifacts = task ? Object.values(task.artifacts) : [];
  return (
    <div className="panel">
      <div className="section-head">
        <h2>产物</h2>
        <span className="muted">{artifacts.length ? `${artifacts.length} 个` : ""}</span>
      </div>
      <div className="artifact-list">
        {task && artifacts.length ? (
          artifacts.map((artifact) => (
            <div className="artifact" key={artifact.artifact_id}>
              <div>
                <div className="artifact-name">
                  <ArtifactIcon type={artifact.type} />
                  {artifact.alias || artifact.artifact_id}
                </div>
                <div className="artifact-uri">
                  {artifact.type} · {shortPath(artifact.uri)}
                </div>
              </div>
              <a href={artifactFileUrl(task.task_id, artifact.artifact_id)} target="_blank" rel="noreferrer">
                <button type="button" title="打开产物">
                  <Download size={15} />
                </button>
              </a>
            </div>
          ))
        ) : (
          <div className="empty-state">暂无产物</div>
        )}
      </div>
    </div>
  );
}

function QualityPanel({ task }: { task: TaskState | null }) {
  const gates = [
    ["输入质量", task?.working_memory?.input_quality],
    ["配准质量", task?.working_memory?.alignment_quality],
    ["结果质量", task?.working_memory?.change_result_quality]
  ] as const;
  return (
    <div className="panel">
      <div className="section-head">
        <h2>质量门</h2>
        <span className="muted">{task?.working_memory?.quality?.score != null ? `总分 ${task.working_memory.quality.score}` : ""}</span>
      </div>
      <div className="context-list">
        {gates.some(([, value]) => value) ? (
          gates.map(([name, value]) =>
            value ? (
              <div className="context" key={name}>
                <div className="context-title">
                  {value.passed ? <Check size={15} /> : <Clock size={15} />}
                  {name}
                  <StatusBadge status={value.passed ? "succeeded" : "failed"} />
                </div>
                <div className="context-body">
                  {value.recommendation || ""}
                  {value.correlation != null ? ` 相关性：${Number(value.correlation).toFixed(3)}` : ""}
                  {value.changed_ratio != null ? ` 变化比例：${(Number(value.changed_ratio) * 100).toFixed(2)}%` : ""}
                </div>
              </div>
            ) : null
          )
        ) : (
          <div className="empty-state">质量检查尚未执行</div>
        )}
      </div>
    </div>
  );
}

function ArtifactIcon({ type }: { type: Artifact["type"] }) {
  if (type === "table") return <Table2 size={15} />;
  if (type === "report") return <FileText size={15} />;
  if (type === "vector" || type === "raster") return <Layers size={15} />;
  return <FileText size={15} />;
}

function ContextPanel({ chunks }: { chunks: KnowledgeChunk[] }) {
  return (
    <div className="panel">
      <div className="section-head">
        <h2>知识依据</h2>
        <span className="muted">{chunks.length ? `${chunks.length} 条` : ""}</span>
      </div>
      <div className="context-list">
        {chunks.length ? (
          chunks.map((chunk) => (
            <div className="context" key={chunk.chunk_id}>
              <div className="context-title">
                <Search size={15} />
                {chunk.title}
              </div>
              <div className="context-body">{chunk.content}</div>
            </div>
          ))
        ) : (
          <div className="empty-state">暂无依据</div>
        )}
      </div>
    </div>
  );
}

function EventList({ events }: { events: EventRecord[] }) {
  return (
    <div className="event-list">
      {events.length ? (
        events
          .slice()
          .reverse()
          .map((event) => (
            <div className="event" key={event.event_id}>
              <div className="event-type">
                <Clock size={15} />
                {event.event_type}
              </div>
              <div className="event-time">{formatDate(event.created_at)}</div>
            </div>
          ))
      ) : (
        <div className="empty-state">暂无事件</div>
      )}
    </div>
  );
}

function FeedbackPanel({
  disabled,
  onSubmit
}: {
  disabled: boolean;
  onSubmit: (payload: { rating: number; accepted: boolean; comment: string }) => void;
}) {
  const [rating, setRating] = useState(4);
  const [accepted, setAccepted] = useState(true);
  const [comment, setComment] = useState("");

  return (
    <section className="panel feedback-panel">
      <div className="section-head">
        <h2>结果反馈</h2>
        <MessageSquare size={16} />
      </div>
      <form
        className="feedback-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ rating, accepted, comment });
          setComment("");
        }}
      >
        <select value={rating} disabled={disabled} onChange={(event) => setRating(Number(event.target.value))}>
          {[5, 4, 3, 2, 1].map((score) => (
            <option key={score} value={score}>
              {score} 分
            </option>
          ))}
        </select>
        <label className="toggle compact">
          <input type="checkbox" checked={accepted} disabled={disabled} onChange={(event) => setAccepted(event.target.checked)} />
          <span />
          通过
        </label>
        <input value={comment} disabled={disabled} placeholder="反馈" onChange={(event) => setComment(event.target.value)} />
        <button type="submit" disabled={disabled}>
          <Check size={16} />
          提交
        </button>
      </form>
    </section>
  );
}

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

function artifactByAlias(task: TaskState, alias: string): Artifact | null {
  const artifactId = task.artifact_refs?.[alias];
  return artifactId ? task.artifacts[artifactId] : null;
}

function shortPath(uri: string): string {
  return uri.replaceAll("\\", "/").split("/").slice(-3).join("/");
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
