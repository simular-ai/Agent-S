/** Compiled frontend lifecycle check; no browser or desktop required. */
import assert from "node:assert/strict";
const elements = new Map();
const defaults = { instruction: "synthetic", provider: "lmstudio", model: "stub", model_url: "http://stub/v1", ground_provider: "lmstudio", ground_model: "stub", ground_url: "http://stub/v1", approval: "dry_run", profile_pick: "default", profile_name: "Default" };
globalThis.document = {
  querySelector: () => ({ content: "test-token" }),
  getElementById(id) {
    if (!elements.has(id)) {
      const node = (id === "provider" || id === "ground_provider" || id === "approval" || id === "monitor" || id === "profile_pick" || id === "model_pick" || id === "ground_pick")
        ? new globalThis.HTMLSelectElement()
        : new globalThis.HTMLInputElement();
      Object.assign(node, { value: defaults[id] ?? "", checked: false, disabled: false, style: {}, textContent: "", scrollHeight: 0, replaceChildren() {}, appendChild() {}, addEventListener() {}, className: "", innerHTML: "", onclick: null, onchange: null });
      if (id.includes("enable") || id === "overlay") node.type = "checkbox";
      elements.set(id, node);
    }
    return elements.get(id);
  },
  createElement: (tag) => {
    if (tag === "option") return { appendChild() {}, append() {}, textContent: "", value: "" };
    return { appendChild() {}, append() {}, textContent: "" };
  },
};
globalThis.window = globalThis.window ?? {};
globalThis.window.confirm = () => true;
class TestElement {}
class TestInput extends TestElement { constructor() { super(); this.type = "text"; } }
class TestSelect extends TestElement {}
globalThis.HTMLElement = TestElement;
globalThis.HTMLButtonElement = TestElement;
globalThis.HTMLImageElement = TestElement;
globalThis.HTMLTextAreaElement = TestElement;
globalThis.HTMLInputElement = TestInput;
globalThis.HTMLSelectElement = TestSelect;
let timerId = 0;
const timers = new Map();
Object.assign(globalThis.window, { setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); }, addEventListener() {} });
const pending = [];
const requests = [];
globalThis.fetch = (path, options = {}) => {
  assert.equal(options.headers instanceof Headers ? options.headers.get("Authorization") : options.headers.Authorization, "Bearer test-token");
  requests.push(path);
  // Ignore AbortSignal deliberately: stale responses must still be harmless.
  return new Promise(resolve => pending.push({ path, resolve }));
};
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
const state = (id, status) => ({ id, status, steps: [], logs: [], error: null, done_reason: status === "done" ? "done" : null, has_screenshot: false, pending_approval: null, version: 1, instruction: "synthetic" });
function respond(path, value) {
  const index = pending.findIndex(request => request.path === path);
  assert.notEqual(index, -1, `missing request ${path}`);
  pending.splice(index, 1)[0].resolve({ ok: true, text: async () => JSON.stringify(value) });
}
const { initRunPanel } = await import("../gui_agents/s3/ui/static/runPanel.js");
const { initConfigPanel, currentProfileId } = await import("../gui_agents/s3/ui/static/configPanel.js");
const profileResponses = {
  "/api/providers": { engine_types: ["lmstudio", "codex"], presets: { codex: { base_url: "", api_key: "", hint: "codex" } } },
  "/api/config": { profile_id: "default", profile_name: "Default", provider: "codex", model: "gpt-6-astra", model_url: "", model_temperature: 0, ground_provider: "lmstudio", ground_model: "stub", ground_url: "http://stub/v1", max_steps: 15, max_trajectory_length: 8, enable_reflection: true, enable_local_env: false, approval: "dry_run", overlay: false, monitor: 1, inference_timeout: 60, action_timeout: 60, task_timeout: 600, azure_api_version: "", ground_azure_api_version: "" },
  "/api/profiles": { active_profile_id: "default", profiles: [{ id: "default", name: "Default", provider: "codex", model: "gpt-6-astra", ground_provider: "lmstudio", ground_model: "stub", active: true }] },
  "/api/status": { platform: "test", screen: { width: 8, height: 8 }, monitors: [], active_tasks: 0, server_time: 0 },
};
globalThis.fetch = (path, options = {}) => {
  assert.equal(options.headers instanceof Headers ? options.headers.get("Authorization") : options.headers.Authorization, "Bearer test-token");
  requests.push(path);
  if (path in profileResponses || path.startsWith("/api/codex/status")) {
    const value = path.startsWith("/api/codex/status")
      ? { signed_in: true, account: "demo", model: "gpt-6-astra" }
      : profileResponses[path];
    return Promise.resolve({ ok: true, text: async () => JSON.stringify(value) });
  }
  return new Promise(resolve => pending.push({ path, resolve }));
};
await initConfigPanel();
assert.equal(elements.get("provider").value, "codex");
assert.equal(currentProfileId(), "default");
// Codex planner hides URL/key inputs and shows subscription status.
assert.equal(elements.get("codexBox").style.display, "");
assert.equal(elements.get("model_url").disabled, true);
assert.ok(elements.get("codexStatus").textContent.includes("Signed in"));
const initialization = initRunPanel();
await flush();
respond("/api/tasks", { tasks: [] });
await initialization;
elements.get("start").onclick();
await flush();
assert.equal(elements.get("start").disabled, true);
respond("/api/tasks", state("A", "queued"));
await flush();
assert.ok(pending.some(request => request.path === "/api/tasks/A"));
elements.get("refreshTasks").onclick();
await flush();
respond("/api/tasks", { tasks: [state("B", "planning")] });
await flush();
respond("/api/tasks/B", state("B", "planning"));
await flush();
const timerBeforeStaleResponse = timerId;
respond("/api/tasks/A", state("A", "done"));
await flush();
assert.ok(timers.has(timerBeforeStaleResponse), "stale A response cleared B's poll");
assert.equal(elements.get("stop").disabled, false);
elements.get("stop").onclick();
await flush();
assert.equal(requests.at(-1), "/api/tasks/B/stop");
respond("/api/tasks/B/stop", state("B", "stopping"));
await flush();
console.log("Frontend lifecycle: stale replies ignored, task restored, Stop targets B, authenticated requests.");
