/** Shared shapes for the Agent S3 UI: server config, tasks, and steps. */
export const ACTIVE_STATES = new Set(["queued", "planning", "awaiting_approval", "executing", "stopping", "finishing"]);
export function taskState(value) {
    const task = value;
    if (!task || typeof task.id !== "string" || typeof task.status !== "string" || !Array.isArray(task.steps) || !Array.isArray(task.logs)) {
        throw new Error("Server returned an invalid task state");
    }
    if (task.steps.some(step => !step || typeof step.n !== "number" || typeof step.status !== "string" || typeof step.exec_code !== "string")) {
        throw new Error("Server returned an invalid step");
    }
    if (task.pending_approval && (typeof task.pending_approval.step !== "number" || typeof task.pending_approval.approval_token !== "string")) {
        throw new Error("Server returned an invalid approval token");
    }
    return task;
}
