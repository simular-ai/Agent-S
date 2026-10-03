/**
 * Left column: planner card, grounding card, agent-behaviour card.
 * Owns config load/save plus the Fetch-models / Test-connection actions.
 */
import { api, checkVal, el, inputNum, inputVal, msg } from "./api.js";
import type {
  AgentConfig,
  CodexStatusResponse,
  ConnectionTestResponse,
  ModelsListResponse,
  ProfilesResponse,
  ProvidersResponse,
  StatusResponse,
} from "./types.js";

export type Slot = "planner" | "grounding";
let presets: ProvidersResponse["presets"] = {};
let clearPlannerKey = false;
let clearGroundKey = false;
let activeProfileId = "";
let profileDirty = false;
let codexState: CodexStatusResponse | null = null;

export function currentProfileId(): string {
  return activeProfileId;
}

/** Warn about unsaved edits before the caller switches profiles. */
function confirmDiscard(): boolean {
  if (!profileDirty) return true;
  return window.confirm("This profile has unsaved changes. Switch profiles and discard them?");
}

/** Read the whole config form into an AgentConfig object. */
export function readConfig(): AgentConfig {
  return {
    provider: el<HTMLSelectElement>("provider").value,
    model: inputVal("model").trim(),
    model_url: inputVal("model_url").trim(),
    ...(inputVal("model_api_key") || clearPlannerKey ? { model_api_key: inputVal("model_api_key") } : {}),
    model_temperature: inputNum("model_temperature", 0),
    ground_provider: el<HTMLSelectElement>("ground_provider").value,
    ground_model: inputVal("ground_model").trim(),
    ground_url: inputVal("ground_url").trim(),
    ...(inputVal("ground_api_key") || clearGroundKey ? { ground_api_key: inputVal("ground_api_key") } : {}),
    grounding_width: inputNum("grounding_width", 1920),
    grounding_height: inputNum("grounding_height", 1080),
    max_steps: inputNum("max_steps", 15),
    max_trajectory_length: inputNum("max_trajectory_length", 8),
    enable_reflection: checkVal("enable_reflection"),
    enable_local_env: checkVal("enable_local_env"),
    approval: el<HTMLSelectElement>("approval").value as AgentConfig["approval"],
    overlay: checkVal("overlay"),
    monitor: inputNum("monitor", 1),
    inference_timeout: inputNum("inference_timeout", 60),
    action_timeout: inputNum("action_timeout", 60),
    task_timeout: inputNum("task_timeout", 600),
    azure_api_version: inputVal("azure_api_version"),
    ground_azure_api_version: inputVal("ground_azure_api_version"),
  };
}

/** Fill provider dropdowns from GET /api/providers. */
async function loadProviders(): Promise<void> {
  const res = await api<ProvidersResponse>("/api/providers");
  if (!Array.isArray(res.engine_types) || !res.presets) throw new Error("Invalid provider response");
  presets = res.presets;
  for (const id of ["provider", "ground_provider"]) {
    const select = el<HTMLSelectElement>(id);
    select.innerHTML = "";
    for (const engine of res.engine_types) {
      const opt = document.createElement("option");
      opt.value = engine;
      opt.textContent = engine;
      select.appendChild(opt);
    }
  }
}

/** Fill provider/credential fields from a config payload. */
function applyConfigValues(cfg: AgentConfig): void {
  for (const [key, value] of Object.entries(cfg)) {
    if (key === "profile_id" || key === "profile_name") continue;
    const node = document.getElementById(key);
    if (!node) continue;
    if (node instanceof HTMLInputElement && node.type === "checkbox") {
      node.checked = Boolean(value);
    } else if (
      node instanceof HTMLInputElement ||
      node instanceof HTMLSelectElement
    ) {
      node.value = String(value ?? "");
    }
  }
  syncCodexVisibility();
}

/** Replace the form with another profile's config payload. */
function fillConfig(cfg: AgentConfig): void {
  activeProfileId = typeof cfg.profile_id === "string" ? cfg.profile_id : activeProfileId;
  profileDirty = false;
  clearPlannerKey = clearGroundKey = false;
  el<HTMLInputElement>("model_api_key").value = "";
  el<HTMLInputElement>("ground_api_key").value = "";
  const pick = el<HTMLSelectElement>("profile_pick");
  if (activeProfileId) pick.value = activeProfileId;
  const nameBox = el<HTMLInputElement>("profile_name");
  if (document.activeElement !== nameBox) nameBox.value = cfg.profile_name ?? "";
  for (const selectId of ["model_pick", "ground_pick"]) {
    const select = el<HTMLSelectElement>(selectId);
    select.style.display = "none";
    select.innerHTML = "";
  }
  applyConfigValues(cfg);
  void refreshCodexStatus();
}

/** Show or hide planner URL/key fields for the Codex subscription provider. */
function syncCodexVisibility(): void {
  const codex = inputVal("provider") === "codex";
  el("codexBox").style.display = codex ? "" : "none";
  for (const id of ["model_url", "model_api_key"]) {
    const node = el<HTMLInputElement>(id);
    node.disabled = codex;
    if (codex) node.value = "";
  }
  if (codex) renderCodexStatus();
}

function renderCodexStatus(): void {
  const target = el("codexStatus");
  if (!codexState) {
    target.textContent = "Codex status unknown — Refresh status.";
    return;
  }
  target.textContent = codexState.signed_in
    ? `Signed in: ${codexState.account || "ChatGPT/Codex account"} · model ${codexState.model || inputVal("model") || "gpt-6-astra"}`
    : `Not signed in: ${codexState.error || "use Login with Codex"}`;
}

async function refreshCodexStatus(): Promise<void> {
  if (inputVal("provider") !== "codex") return;
  el("codexStatus").textContent = "Checking Codex sign-in…";
  try {
    const params = new URLSearchParams({ model: inputVal("model").trim() });
    codexState = await api<CodexStatusResponse>(`/api/codex/status?${params.toString()}`);
  } catch (err) {
    codexState = { signed_in: false, error: (err as Error).message };
  }
  renderCodexStatus();
}

function markDirty(): void {
  profileDirty = true;
}

function renderProfiles(res: ProfilesResponse): void {
  const select = el<HTMLSelectElement>("profile_pick");
  const current = res.active_profile_id;
  select.innerHTML = "";
  for (const profile of res.profiles) {
    const opt = document.createElement("option");
    opt.value = profile.id;
    opt.textContent = `${profile.name} — ${profile.provider}/${profile.model || "no model"}`;
    select.appendChild(opt);
  }
  select.value = current;
  activeProfileId = current;
  const active = res.profiles.find(profile => profile.id === current);
  if (active) el<HTMLInputElement>("profile_name").value = active.name;
  if (res.config) fillConfig(res.config);
  else {
    profileDirty = false;
    void refreshCodexStatus();
  }
  msg("msgProfile", `${res.profiles.length} profile(s); active: ${active?.name ?? current}`, "ok");
}

async function profileAction(path: string, body: unknown, busy: string): Promise<void> {
  msg("msgProfile", busy);
  try {
    const res = await api<ProfilesResponse>(path, { method: "POST", body: JSON.stringify(body) });
    renderProfiles(res);
  } catch (err) {
    msg("msgProfile", `Profile action failed: ${(err as Error).message}`, "err");
  }
}

/** Fill the form from GET /api/config. */
async function loadConfig(): Promise<void> {
  const cfg = await api<AgentConfig>("/api/config");
  if (!cfg || typeof cfg.provider !== "string" || typeof cfg.max_steps !== "number") throw new Error("Invalid configuration response");
  if (typeof cfg.profile_id === "string" && cfg.profile_id) activeProfileId = cfg.profile_id;
  const listed = await api<ProfilesResponse>("/api/profiles");
  const select = el<HTMLSelectElement>("profile_pick");
  select.innerHTML = "";
  for (const profile of listed.profiles) {
    const opt = document.createElement("option");
    opt.value = profile.id;
    opt.textContent = `${profile.name} — ${profile.provider}/${profile.model || "no model"}`;
    select.appendChild(opt);
  }
  select.value = listed.active_profile_id;
  activeProfileId = listed.active_profile_id;
  const active = listed.profiles.find(profile => profile.id === listed.active_profile_id);
  if (active) el<HTMLInputElement>("profile_name").value = active.name;
  applyConfigValues(cfg);
  const status = await api<StatusResponse>("/api/status");
  const monitorSelect = el<HTMLSelectElement>("monitor");
  monitorSelect.innerHTML = "";
  for (const monitor of status.monitors ?? []) {
    const option = document.createElement("option");
    option.value = String(monitor.id);
    option.textContent = `${monitor.id}: ${monitor.width}×${monitor.height} at ${monitor.left},${monitor.top}`;
    monitorSelect.appendChild(option);
  }
}

/** Nudge LM Studio defaults into empty URL fields when picked. */
function applyPresetHint(slot: Slot): void {
  const planner = slot === "planner";
  const provider = inputVal(planner ? "provider" : "ground_provider");
  const preset = presets[provider];
  if (planner && provider === "codex") {
    el<HTMLInputElement>("model_url").value = "";
    el<HTMLInputElement>("model_api_key").value = "";
    clearPlannerKey = true;
    if (!inputVal("model").trim()) el<HTMLInputElement>("model").value = "gpt-6-astra";
    msg("msgPlanner", preset?.hint ?? "Use Login with Codex, then Fetch models.");
    syncCodexVisibility();
    void refreshCodexStatus();
    markDirty();
    return;
  }
  el<HTMLInputElement>(planner ? "model_url" : "ground_url").value = preset?.base_url ?? "";
  el<HTMLInputElement>(planner ? "model_api_key" : "ground_api_key").value = "";
  if (planner) clearPlannerKey = true;
  else clearGroundKey = true;
  msg(planner ? "msgPlanner" : "msgGround", preset?.hint ?? "Configure this provider.");
  if (planner) syncCodexVisibility();
  markDirty();
}

function slotUrls(slot: Slot): { baseUrl: string; apiKey: string } {
  const cfg = readConfig();
  return slot === "planner"
    ? { baseUrl: cfg.model_url, apiKey: cfg.model_api_key ?? "" }
    : { baseUrl: cfg.ground_url, apiKey: cfg.ground_api_key ?? "" };
}

/** GET {base_url}/models (via the server) and offer the ids in a dropdown. */
async function fetchModels(slot: Slot): Promise<void> {
  const isPlanner = slot === "planner";
  const msgId = isPlanner ? "msgPlanner" : "msgGround";
  const pickId = isPlanner ? "model_pick" : "ground_pick";
  const modelId = isPlanner ? "model" : "ground_model";
  try {
    msg(msgId, "listing…");
    const { baseUrl, apiKey } = slotUrls(slot);
    const res = await api<ModelsListResponse>("/api/models/list", {
      method: "POST",
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, aspect: slot, provider: inputVal(isPlanner ? "provider" : "ground_provider") }),
    });
    const pick = el<HTMLSelectElement>(pickId);
    pick.style.display = "";
    pick.innerHTML = "";
    if (!Array.isArray(res.models) || res.models.some(id => typeof id !== "string")) throw new Error("Invalid model list");
    const existing = inputVal(modelId);
    for (const id of res.models) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      pick.appendChild(opt);
    }
    pick.onchange = () => {
      el<HTMLInputElement>(modelId).value = pick.value;
    };
    if (res.models.includes(existing)) {
      pick.value = existing;
    } else if (!existing && res.models.length > 0) {
      pick.value = res.models[0];
      el<HTMLInputElement>(modelId).value = res.models[0];
    }
    msg(msgId, `${res.models.length} model(s) found — existing selection retained.`, "ok");
  } catch (err) {
    msg(
      msgId,
      `Fetch failed: ${(err as Error).message} — is the server (e.g. LM Studio) running with a model loaded?`,
      "err",
    );
  }
}

/** POST /api/connection/test for one slot. */
async function testConnection(slot: Slot): Promise<void> {
  const isPlanner = slot === "planner";
  const msgId = isPlanner ? "msgPlanner" : "msgGround";
  try {
    msg(msgId, "testing…");
    const { baseUrl, apiKey } = slotUrls(slot);
    const res = await api<ConnectionTestResponse>("/api/connection/test", {
      method: "POST",
      body: JSON.stringify({
        base_url: baseUrl,
        api_key: apiKey,
        aspect: isPlanner ? "planner" : "grounding",
        provider: inputVal(isPlanner ? "provider" : "ground_provider"),
        model: inputVal(isPlanner ? "model" : "ground_model"),
        verify_vision: Boolean(inputVal(isPlanner ? "model" : "ground_model")),
      }),
    });
    msg(
      msgId,
      res.ok ? `OK — ${res.count} model(s). Vision request accepted: ${res.vision_request_accepted ? "yes" : "not tested"}. ${res.hint ?? ""}` : `FAIL: ${res.error}`,
      res.ok ? "ok" : "err",
    );
  } catch (err) {
    msg(msgId, `Test failed: ${(err as Error).message}`, "err");
  }
}

/** Wire all config-panel buttons; call once at startup. */
export async function initConfigPanel(): Promise<void> {
  await loadProviders();
  await loadConfig();
  syncCodexVisibility();
  await refreshCodexStatus();
  for (const id of ["provider", "ground_provider", "model", "model_url", "model_api_key", "ground_model", "ground_url", "ground_api_key", "profile_name"]) {
    el(id).addEventListener("input", markDirty);
    el(id).addEventListener("change", markDirty);
  }
  el("saveCfg").onclick = async () => {
    try {
      const payload = { ...readConfig(), profile_id: activeProfileId };
      await api("/api/config", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      profileDirty = false;
      msg("msgCfg", "saved.", "ok");
      el<HTMLInputElement>("model_api_key").value = "";
      el<HTMLInputElement>("ground_api_key").value = "";
      clearPlannerKey = clearGroundKey = false;
    } catch (err) {
      msg("msgCfg", `save failed: ${(err as Error).message}`, "err");
    }
  };
  el("fetchPlanner").onclick = () => void fetchModels("planner");
  el("fetchGround").onclick = () => void fetchModels("grounding");
  el("testPlanner").onclick = () => void testConnection("planner");
  el("testGround").onclick = () => void testConnection("grounding");
  el("provider").onchange = () => applyPresetHint("planner");
  el("ground_provider").onchange = () => applyPresetHint("grounding");
  el("profile_pick").onchange = async () => {
    const next = el<HTMLSelectElement>("profile_pick").value;
    if (!next || next === activeProfileId) return;
    if (!confirmDiscard()) {
      el<HTMLSelectElement>("profile_pick").value = activeProfileId;
      return;
    }
    await profileAction("/api/profiles/select", { profile_id: next }, "switching…");
  };
  el("profileNew").onclick = () => void profileAction("/api/profiles", { name: inputVal("profile_name") || "New profile" }, "creating…");
  el("profileDuplicate").onclick = () => {
    if (!confirmDiscard()) return;
    void profileAction("/api/profiles/duplicate", { profile_id: activeProfileId }, "duplicating…");
  };
  el("profileRename").onclick = () => void profileAction("/api/profiles/rename", { profile_id: activeProfileId, name: inputVal("profile_name") }, "renaming…");
  el("profileDelete").onclick = () => {
    if (!window.confirm(`Delete profile "${inputVal("profile_name") || activeProfileId}"?`)) return;
    void profileAction("/api/profiles/delete", { profile_id: activeProfileId }, "deleting…");
  };
  el("codexRefresh").onclick = () => void refreshCodexStatus();
  el("codexLogin").onclick = async () => {
    msg("msgPlanner", "Starting Codex login in your terminal… complete `codex login` there, then Refresh status.");
    try {
      const res = await api<{ output?: string; status?: CodexStatusResponse }>("/api/codex/login", { method: "POST" });
      codexState = res.status ?? null;
      renderCodexStatus();
      if (res.output) msg("msgPlanner", res.output.slice(-500));
    } catch (err) {
      msg("msgPlanner", `Codex login could not start here: ${(err as Error).message}. Run \`codex login\` in a terminal, then Refresh status.`, "err");
    }
  };
  el("codexLogout").onclick = async () => {
    try {
      const res = await api<{ status?: CodexStatusResponse }>("/api/codex/logout", { method: "POST" });
      codexState = res.status ?? null;
      renderCodexStatus();
      msg("msgPlanner", "Signed out of Codex.", "ok");
    } catch (err) {
      msg("msgPlanner", `Codex logout failed: ${(err as Error).message}`, "err");
    }
  };
}
