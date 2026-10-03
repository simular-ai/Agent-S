/** Entry point: server status pill, panel bootstrap, OpenAI snippet URLs. */
import { api, el } from "./api.js";
import { initConfigPanel } from "./configPanel.js";
import { initRunPanel } from "./runPanel.js";
import type { StatusResponse } from "./types.js";

async function refreshStatus(): Promise<void> {
  try {
    const s = await api<StatusResponse>("/api/status");
    const pill = el("svpill");
    pill.textContent = s.active_tasks
      ? `● ${s.active_tasks} running`
      : "● idle";
    pill.className = `pill ${s.active_tasks ? "run" : "ok"}`;
    el("svinfo").textContent =
      `${s.platform} · ${s.screen.width}×${s.screen.height}`;
  } catch {
    el("svpill").textContent = "○ offline";
    el("svpill").className = "pill";
  }
  // Keep the OpenAI snippet pointing at wherever this page is served.
  el("snipCurl").textContent = `curl ${location.origin}/v1/models -H "Authorization: Bearer $AGENT_S_UI_TOKEN"`;
}

async function main(): Promise<void> {
  try { await initConfigPanel(); }
  catch (error) { el("msgCfg").textContent = `Configuration initialization failed: ${(error as Error).message}. Reload to retry.`; }
  await initRunPanel();
  await refreshStatus();
  const refresh = async (): Promise<void> => { await refreshStatus(); window.setTimeout(() => void refresh(), 10000); };
  window.setTimeout(() => void refresh(), 10000);
}

void main().catch(error => { el("msgRun").textContent = `UI initialization failed: ${String(error)}`; });
