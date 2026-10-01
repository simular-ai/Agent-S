/**
 * Left column: planner card, grounding card, agent-behaviour card.
 * Owns config load/save plus the Fetch-models / Test-connection actions.
 */
import { api, checkVal, el, inputNum, inputVal, msg } from "./api.js";
let presets = {};
let clearPlannerKey = false;
let clearGroundKey = false;
/** Read the whole config form into an AgentConfig object. */
export function readConfig() {
    return {
        provider: el("provider").value,
        model: inputVal("model").trim(),
        model_url: inputVal("model_url").trim(),
        ...(inputVal("model_api_key") || clearPlannerKey ? { model_api_key: inputVal("model_api_key") } : {}),
        model_temperature: inputNum("model_temperature", 0),
        ground_provider: el("ground_provider").value,
        ground_model: inputVal("ground_model").trim(),
        ground_url: inputVal("ground_url").trim(),
        ...(inputVal("ground_api_key") || clearGroundKey ? { ground_api_key: inputVal("ground_api_key") } : {}),
        grounding_width: inputNum("grounding_width", 1920),
        grounding_height: inputNum("grounding_height", 1080),
        max_steps: inputNum("max_steps", 15),
        max_trajectory_length: inputNum("max_trajectory_length", 8),
        enable_reflection: checkVal("enable_reflection"),
        enable_local_env: checkVal("enable_local_env"),
        approval: el("approval").value,
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
async function loadProviders() {
    const res = await api("/api/providers");
    if (!Array.isArray(res.engine_types) || !res.presets)
        throw new Error("Invalid provider response");
    presets = res.presets;
    for (const id of ["provider", "ground_provider"]) {
        const select = el(id);
        select.innerHTML = "";
        for (const engine of res.engine_types) {
            const opt = document.createElement("option");
            opt.value = engine;
            opt.textContent = engine;
            select.appendChild(opt);
        }
    }
}
/** Fill the form from GET /api/config. */
async function loadConfig() {
    const cfg = await api("/api/config");
    if (!cfg || typeof cfg.provider !== "string" || typeof cfg.max_steps !== "number")
        throw new Error("Invalid configuration response");
    const status = await api("/api/status");
    const monitorSelect = el("monitor");
    monitorSelect.innerHTML = "";
    for (const monitor of status.monitors ?? []) {
        const option = document.createElement("option");
        option.value = String(monitor.id);
        option.textContent = `${monitor.id}: ${monitor.width}×${monitor.height} at ${monitor.left},${monitor.top}`;
        monitorSelect.appendChild(option);
    }
    for (const [key, value] of Object.entries(cfg)) {
        const node = document.getElementById(key);
        if (!node)
            continue;
        if (node instanceof HTMLInputElement && node.type === "checkbox") {
            node.checked = Boolean(value);
        }
        else if (node instanceof HTMLInputElement ||
            node instanceof HTMLSelectElement) {
            node.value = String(value ?? "");
        }
    }
}
/** Nudge LM Studio defaults into empty URL fields when picked. */
function applyPresetHint(slot) {
    const planner = slot === "planner";
    const provider = inputVal(planner ? "provider" : "ground_provider");
    const preset = presets[provider];
    el(planner ? "model_url" : "ground_url").value = preset?.base_url ?? "";
    el(planner ? "model_api_key" : "ground_api_key").value = "";
    if (planner)
        clearPlannerKey = true;
    else
        clearGroundKey = true;
    msg(planner ? "msgPlanner" : "msgGround", preset?.hint ?? "Configure this provider.");
}
function slotUrls(slot) {
    const cfg = readConfig();
    return slot === "planner"
        ? { baseUrl: cfg.model_url, apiKey: cfg.model_api_key ?? "" }
        : { baseUrl: cfg.ground_url, apiKey: cfg.ground_api_key ?? "" };
}
/** GET {base_url}/models (via the server) and offer the ids in a dropdown. */
async function fetchModels(slot) {
    const isPlanner = slot === "planner";
    const msgId = isPlanner ? "msgPlanner" : "msgGround";
    const pickId = isPlanner ? "model_pick" : "ground_pick";
    const modelId = isPlanner ? "model" : "ground_model";
    try {
        msg(msgId, "listing…");
        const { baseUrl, apiKey } = slotUrls(slot);
        const res = await api("/api/models/list", {
            method: "POST",
            body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, aspect: slot, provider: inputVal(isPlanner ? "provider" : "ground_provider") }),
        });
        const pick = el(pickId);
        pick.style.display = "";
        pick.innerHTML = "";
        if (!Array.isArray(res.models) || res.models.some(id => typeof id !== "string"))
            throw new Error("Invalid model list");
        const existing = inputVal(modelId);
        for (const id of res.models) {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = id;
            pick.appendChild(opt);
        }
        pick.onchange = () => {
            el(modelId).value = pick.value;
        };
        if (res.models.includes(existing)) {
            pick.value = existing;
        }
        else if (!existing && res.models.length > 0) {
            pick.value = res.models[0];
            el(modelId).value = res.models[0];
        }
        msg(msgId, `${res.models.length} model(s) found — existing selection retained.`, "ok");
    }
    catch (err) {
        msg(msgId, `Fetch failed: ${err.message} — is the server (e.g. LM Studio) running with a model loaded?`, "err");
    }
}
/** POST /api/connection/test for one slot. */
async function testConnection(slot) {
    const isPlanner = slot === "planner";
    const msgId = isPlanner ? "msgPlanner" : "msgGround";
    try {
        msg(msgId, "testing…");
        const { baseUrl, apiKey } = slotUrls(slot);
        const res = await api("/api/connection/test", {
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
        msg(msgId, res.ok ? `OK — ${res.count} model(s). Vision request accepted: ${res.vision_request_accepted ? "yes" : "not tested"}. ${res.hint ?? ""}` : `FAIL: ${res.error}`, res.ok ? "ok" : "err");
    }
    catch (err) {
        msg(msgId, `Test failed: ${err.message}`, "err");
    }
}
/** Wire all config-panel buttons; call once at startup. */
export async function initConfigPanel() {
    await loadProviders();
    await loadConfig();
    el("saveCfg").onclick = async () => {
        try {
            await api("/api/config", {
                method: "POST",
                body: JSON.stringify(readConfig()),
            });
            msg("msgCfg", "saved.", "ok");
            el("model_api_key").value = "";
            el("ground_api_key").value = "";
            clearPlannerKey = clearGroundKey = false;
        }
        catch (err) {
            msg("msgCfg", `save failed: ${err.message}`, "err");
        }
    };
    el("fetchPlanner").onclick = () => void fetchModels("planner");
    el("fetchGround").onclick = () => void fetchModels("grounding");
    el("testPlanner").onclick = () => void testConnection("planner");
    el("testGround").onclick = () => void testConnection("grounding");
    el("provider").onchange = () => applyPresetHint("planner");
    el("ground_provider").onchange = () => applyPresetHint("grounding");
}
