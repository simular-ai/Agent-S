/** Task supervision: one poll at a time, generation-bound controls and images. */
import { api, el, inputVal, msg, screenshot } from "./api.js";
import { readConfig } from "./configPanel.js";
import { ACTIVE_STATES, taskState } from "./types.js";
let current = null;
let generation = 0;
let timer;
let abort = null;
let imageUrl = null;
let starting = false;
function controls() {
    const active = current !== null && ACTIVE_STATES.has(current.status);
    el("start").disabled = starting || active;
    el("stop").disabled = !active || current?.status === "stopping";
    el("approve").disabled = !(active && current?.pending_approval);
    el("reject").disabled = !(active && current?.pending_approval);
}
function render(task) {
    const logs = el("logs");
    logs.textContent = task.logs.join("\n");
    logs.scrollTop = logs.scrollHeight;
    const box = el("steps");
    box.replaceChildren();
    for (const step of task.steps) {
        const card = document.createElement("div");
        card.className = "step";
        const heading = document.createElement("b");
        heading.textContent = `Step ${step.n} · ${step.status}`;
        card.appendChild(heading);
        for (const [label, body] of [["Plan", step.plan], ["Prepared action", step.exec_code], ["Reflection", step.reflection], ["Error", step.error]]) {
            if (!body)
                continue;
            const details = document.createElement("details");
            details.open = label === "Prepared action";
            const summary = document.createElement("summary");
            summary.textContent = label;
            const pre = document.createElement("pre");
            pre.textContent = body;
            details.append(summary, pre);
            card.appendChild(details);
        }
        box.appendChild(card);
    }
}
function reset() {
    generation++;
    window.clearTimeout(timer);
    abort?.abort();
    abort = new AbortController();
    current = null;
    return generation;
}
async function poll(id, epoch) {
    if (epoch !== generation || !abort)
        return;
    const signal = abort.signal;
    try {
        const task = taskState(await api(`/api/tasks/${encodeURIComponent(id)}`, { signal }));
        if (epoch !== generation)
            return;
        current = task;
        render(task);
        controls();
        if (task.has_screenshot) {
            const image = await screenshot(id, signal);
            if (epoch !== generation)
                return;
            const nextUrl = URL.createObjectURL(image);
            el("shot").src = nextUrl;
            if (imageUrl)
                URL.revokeObjectURL(imageUrl);
            imageUrl = nextUrl;
        }
        if (!ACTIVE_STATES.has(task.status)) {
            msg("msgRun", `finished: ${task.done_reason ?? task.status}${task.error ? ` — ${task.error.slice(0, 300)}` : ""}`, task.status === "done" ? "ok" : "warn");
            return;
        }
        msg("msgRun", `task ${id}: ${task.status}`, "ok");
    }
    catch (error) {
        if (epoch !== generation || signal.aborted)
            return;
        msg("msgRun", `poll error: ${error.message}; retrying`, "err");
    }
    if (epoch === generation)
        timer = window.setTimeout(() => void poll(id, epoch), 1000);
}
async function history() {
    const response = await api("/api/tasks");
    if (!Array.isArray(response.tasks))
        throw new Error("Invalid task list");
    const tasks = response.tasks.map(taskState).reverse();
    const select = el("task_pick");
    select.replaceChildren();
    for (const task of tasks) {
        const option = document.createElement("option");
        option.value = task.id;
        option.textContent = `${task.status}: ${task.instruction.slice(0, 60)}`;
        select.appendChild(option);
    }
    select.onchange = () => { const epoch = reset(); void poll(select.value, epoch); };
    const active = tasks.find(task => ACTIVE_STATES.has(task.status));
    if (active) {
        select.value = active.id;
        const epoch = reset();
        await poll(active.id, epoch);
    }
}
async function start() {
    if (starting || (current && ACTIVE_STATES.has(current.status)))
        return;
    const instruction = inputVal("instruction").trim();
    if (!instruction) {
        msg("msgRun", "Type a task first", "warn");
        return;
    }
    starting = true;
    controls();
    const epoch = reset();
    try {
        const task = taskState(await api("/api/tasks", { method: "POST", body: JSON.stringify({ instruction, config: readConfig() }) }));
        if (epoch !== generation)
            return;
        current = task;
        await poll(task.id, epoch);
    }
    catch (error) {
        msg("msgRun", `Start failed: ${error.message}`, "err");
    }
    finally {
        starting = false;
        controls();
    }
}
async function control(approved) {
    const task = current;
    const epoch = generation;
    if (!task)
        return;
    try {
        const pending = task.pending_approval;
        if (approved !== undefined && !pending)
            throw new Error("No step is awaiting approval");
        const suffix = approved === undefined ? "stop" : "approve";
        const updated = taskState(await api(`/api/tasks/${encodeURIComponent(task.id)}/${suffix}`, { method: "POST", ...(approved === undefined ? {} : { body: JSON.stringify({ ...pending, approved }) }) }));
        if (epoch !== generation)
            return;
        current = updated;
        controls();
    }
    catch (error) {
        msg("msgRun", `Control failed: ${error.message}`, "err");
    }
}
export async function initRunPanel() {
    el("start").onclick = () => void start();
    el("stop").onclick = () => void control();
    el("approve").onclick = () => void control(true);
    el("reject").onclick = () => void control(false);
    el("refreshTasks").onclick = () => void history().catch(error => msg("msgRun", String(error), "err"));
    controls();
    try {
        await history();
    }
    catch (error) {
        msg("msgRun", `Cannot restore tasks: ${error.message}`, "err");
    }
    window.addEventListener("beforeunload", () => { abort?.abort(); if (imageUrl)
        URL.revokeObjectURL(imageUrl); });
}
