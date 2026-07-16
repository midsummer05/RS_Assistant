export type TaskStatus =
  | "created"
  | "planning"
  | "waiting_human"
  | "running"
  | "paused"
  | "finalizing"
  | "succeeded"
  | "failed"
  | "cancelled";

export type StepStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped"
  | "waiting_human";

export interface Artifact {
  artifact_id: string;
  type: "raster" | "vector" | "table" | "image" | "report" | "model" | "log";
  uri: string;
  alias?: string | null;
  crs?: string | null;
  bbox?: number[] | null;
  metadata: Record<string, unknown>;
  checksum?: string | null;
  created_at: string;
}

export interface StepState {
  step_id: string;
  name: string;
  status: StepStatus;
  tool_name?: string | null;
  params: Record<string, unknown>;
  expected_outputs: string[];
  quality_gate?: string | null;
  output_refs: string[];
}

export interface Interrupt {
  interrupt_id: string;
  type: string;
  reason: string;
  status: "open" | "approved" | "revised" | "rejected";
  payload: Record<string, unknown>;
  created_at: string;
}

export interface KnowledgeChunk {
  chunk_id: string;
  title: string;
  content: string;
  source_type: string;
  task_tags: string[];
  metadata: Record<string, unknown>;
  score: number;
}

export interface PlanSummary {
  confidence: number;
  assumptions: string[];
  risks: string[];
  retrieved_context_refs: string[];
}

export interface TaskState {
  task_id: string;
  title?: string | null;
  user_goal: string;
  status: TaskStatus;
  task_type?: string | null;
  plan: StepState[];
  plan_summary?: PlanSummary | null;
  artifacts: Record<string, Artifact>;
  artifact_refs: Record<string, string>;
  retrieved_context: KnowledgeChunk[];
  interrupts: Interrupt[];
  working_memory: Record<string, any>;
  user_feedback: Record<string, unknown>[];
  agent_mode: "workflow" | "agent";
  execution_budget?: number | null;
  updated_at: string;
  created_at: string;
}

export interface EventRecord {
  event_id: string;
  task_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}
