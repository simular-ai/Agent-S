/** Compiled frontend lifecycle check; no browser or desktop required. */
import assert from "node:assert/strict";
const elements = new Map();
const defaults = { instruction: "synthetic", provider: "lmstudio", model: "stub", model_url: "http://stub/v1", ground_provider: "lmstudio", ground_model: "stub", ground_url: "http://stub/v1", approval: "dry_run" };
globalThis.document = {
  querySelector: () => ({ content: "test-token" }),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, { value: defaults[id] ?? "", checked: false, disabled: false, style: {}, textContent: "", scrollHeight: 0, replaceChildren() {}, appendChild() {} });
    return elements.get(id);
  },
  createElement: () => ({ appendChild() {}, append() {}, textContent: "" }),
};
let timerId = 0;
const timers = new Map();
globalThis.window = { setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); }, addEventListener() {} };
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
