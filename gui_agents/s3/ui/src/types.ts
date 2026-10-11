/** Shared shapes for the Agent S3 UI: server config, tasks, and steps. */

export interface ScreenSize {
  width: number;
  height: number;
}

export interface StatusResponse {
  platform: string;
  screen: ScreenSize;
  active_tasks: number;
  server_time: number;
  monitors: Array<ScreenSize & { id: number; left: number; top: number }>;
}

export interface ProvidersResponse {
  engine_types: string[];
  presets: Record<string, { base_url: string; api_key: string; hint: string }>;
}

export interface AgentConfig {
  provider: string;
  model: string;
  model_url: string;
  model_api_key?: string;
  model_temperature: number;
  ground_provider: string;
  ground_model: string;
  ground_url: string;
  ground_api_key?: string;
  grounding_width: number;
  grounding_height: number;
  max_steps: number;
  max_trajectory_length: number;
  enable_reflection: boolean;
  enable_local_env: boolean;
  approval: "auto" | "manual" | "dry_run";
  overlay: boolean;
  monitor: number;
  inference_timeout: number;
  action_timeout: number;
  task_timeout: number;
  azure_api_version: string;
  ground_azure_api_version: string;
}

export interface StepInfo {
  n: number;
  status: string;
  plan: string;
  plan_code: string;
  exec_code: string;
  reflection: string;
  error: string;
  timestamp: string;
}

export interface TaskState {
  id: string;
  instruction: string;
  status: string;
  current_step: number;
  max_steps: number;
  approval: string;
  created_at: string;
  finished_at: string | null;
  done_reason: string | null;
  error: string | null;
  steps: StepInfo[];
  logs: string[];
  has_screenshot: boolean;
  awaiting_approval: boolean;
  overlay: string;
  pending_approval: { step: number; approval_token: string } | null;
  version: number;
}

export interface ModelsListResponse {
  ok: boolean;
  models: string[];
  raw: Array<{ id?: string } | string>;
}

export interface ConnectionTestResponse {
  ok: boolean;
  aspect: string;
  models?: string[];
  count?: number;
  hint?: string;
  error?: string;
  vision_request_accepted?: boolean;
}

export const ACTIVE_STATES = new Set(["queued", "planning", "awaiting_approval", "executing", "stopping", "finishing"]);

export function taskState(value: unknown): TaskState {
  const task = value as Partial<TaskState> | null;
  if (!task || typeof task.id !== "string" || typeof task.status !== "string" || !Array.isArray(task.steps) || !Array.isArray(task.logs)) {
    throw new Error("Server returned an invalid task state");
  }
  if (task.steps.some(step => !step || typeof step.n !== "number" || typeof step.status !== "string" || typeof step.exec_code !== "string")) {
    throw new Error("Server returned an invalid step");
  }
  if (task.pending_approval && (typeof task.pending_approval.step !== "number" || typeof task.pending_approval.approval_token !== "string")) {
    throw new Error("Server returned an invalid approval token");
  }
  return task as TaskState;
}
