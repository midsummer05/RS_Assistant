import type { EventRecord, TaskState } from "./types";

export interface CreateTaskPayload {
  user_goal: string;
  image_t1_uri?: string;
  image_t2_uri?: string;
  asset_t1_id?: string;
  asset_t2_id?: string;
  user_id?: string;
  project_id?: string;
  auto_confirm?: boolean;
  agent_mode?: "workflow" | "agent";
  execution_budget?: number;
}

export interface FeedbackPayload {
  rating: number;
  accepted: boolean;
  comment: string;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep status text.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  listTasks: () => request<TaskState[]>("/api/tasks"),
  getTask: (taskId: string) => request<TaskState>(`/api/tasks/${taskId}`),
  createTask: (payload: CreateTaskPayload) =>
    request<TaskState>("/api/tasks", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  approvePlan: (taskId: string, interruptId: string) =>
    request<TaskState>(`/api/tasks/${taskId}/interrupts/${interruptId}/approve`, {
      method: "POST",
      body: "{}"
    }),
  resumeTask: (taskId: string) =>
    request<TaskState>(`/api/tasks/${taskId}/resume`, {
      method: "POST",
      body: "{}"
    }),
  retryTask: (taskId: string) =>
    request<TaskState>(`/api/tasks/${taskId}/retry`, {
      method: "POST",
      body: "{}"
    }),
  submitFeedback: (taskId: string, payload: FeedbackPayload) =>
    request<TaskState>(`/api/tasks/${taskId}/feedback`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  listEvents: (taskId: string) => request<EventRecord[]>(`/api/tasks/${taskId}/events`)
};

export function artifactFileUrl(taskId: string, artifactId: string): string {
  return `/api/tasks/${taskId}/artifacts/${artifactId}/file`;
}
