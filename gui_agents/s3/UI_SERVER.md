# Supervised Agent S3 UI, API, and MCP

## Install and start

Python 3.10–3.12 is supported:

```powershell
.venv\Scripts\python.exe -m pip install -e .
agent_s_ui
```

Open http://127.0.0.1:8000. Tasks initially use **dry_run**; the overlay is
initially disabled. Select a monitor and both model IDs before launching.

State lives in `%LOCALAPPDATA%\AgentS` on Windows, or `$XDG_CONFIG_HOME/agent-s`
(default `~/.config/agent-s`) elsewhere. Configuration is `ui-config.json` and
the server token is `ui-token`. Override with `AGENT_S_UI_CONFIG` and
`AGENT_S_UI_TOKEN`. The config file now stores named profiles
(`profiles`, `profile_order`, `active_profile_id`); a legacy single-config file
is migrated into a `Default` profile on startup. An earlier project-local config can be selected explicitly:

```powershell
$env:AGENT_S_UI_CONFIG = "C:\code\Agent-S\agent_s_ui_config.json"
agent_s_ui
```

Configuration is validated at startup and on updates. Backend keys are encrypted
using Windows user-scoped DPAPI. On other operating systems they are plaintext
in an owner-readable/writable configuration file; environment credentials avoid
storing them there. Config responses redact keys. Omitted keys retain their
value; an explicit empty key clears it. Changing provider clears its old URL/key.
Use the Configuration card to create, duplicate, rename, delete, and select
profiles; each profile keeps its own planner, grounding, and agent settings.
Tasks record the launching profile and snapshot its settings, so switching
profiles only affects later runs.

Choose planner provider `codex` to use your ChatGPT/Codex subscription through
the `openai-codex` Python SDK (installed with Agent S). Use **Login with Codex**,
then **Refresh status**, **Fetch**, and **Test connection**. The SDK handles
browser sign-in, account status, and live model discovery. Planner turns use
ephemeral, read-only SDK threads with shell tools disabled and approvals denied.
The SDK manages credential storage and refresh. Its package includes the Codex
runtime; Agent S does not invoke CLI login or `codex exec` commands. Grounding
still uses the configured UI-TARS endpoint.

## Execution contract

UI, MCP, and `/v1` desktop runs share one task manager and one spawned worker.
Active states are `queued`, `planning`, `awaiting_approval`, `executing`,
`stopping`, and `finishing`. Validation parses one allowlisted
`agent.action(...)` with literal arguments and invokes no action. Preparation
may call grounding/OCR but performs no desktop input or local-script execution.
The supervised runner dispatches structured actions, not model Python.

- **dry_run:** capture and prepare without executing. Screenshots are still sent
  to the configured models.
- **manual:** approve the published task/step token. Stale/duplicate approval
  returns 409.
- **auto:** dispatch immediately, subject to cancellation and deadlines.

`enable_local_env` enables arbitrary coding-agent scripts; it is not a sandbox.
Dry-run defers that action. A manual approval covers the **whole coding-agent
invocation**, not each individual script. Spreadsheet scripting is unsupported
by the supervised runner; the legacy SDK/CLI retains its action implementations.

Stop is checked after prediction, during approval, and before dispatch. After a
two-second grace period an uncooperative worker is terminated; Windows also
terminates its process tree. Planning, action, and overall deadlines are
configurable. Stop cannot undo completed clicks or file changes.
Windows workers enter a kill-on-close Job Object before they can start; server
termination also closes the job and kills its worker/descendant processes.
A desktop lease also prevents two UI instances from executing simultaneously:
a Windows session-scoped mutex or Unix user-scoped file lock. A task launched
through another instance fails without desktop execution when the lease is busy.

Capture and input use the selected physical monitor and its offset. Windows DPI
awareness is enabled before GUI imports. UI-TARS grounding dimensions must match
the loaded model/processor, not the monitor. During manual approval, the Windows
foreground window is restored before input; a changed title/rectangle or monitor
layout cancels the action. Keep the control UI in a separate browser window from
a browser task. Manual typing requires an explicitly grounded element to avoid
typing into the approval UI after its focus changes. These checks cannot detect
every change to a window's contents.
Discovery and a synthetic vision
request do not measure grounding accuracy or verify a GGUF/mmproj pair.

Tk runs on the main thread of a separate overlay process. The window stays
withdrawn from capture through execution, and can appear between steps/during
approval. Timeouts close it. No screenshot masking is used.

History is in memory, capped at 25 tasks and one hour; restarting loses it.
`DELETE /api/tasks/{id}` deletes an inactive task. Browser reload reconnects to
active tasks. Closing the UI tab does not stop a UI-launched task.

## Authenticated API

Every `/api/*` and `/v1/*` request needs `Authorization: Bearer <server-token>`.
The local HTML page bootstraps the UI token; foreign origins and untrusted Host
headers are rejected. Screenshots use authenticated fetch, not token URLs.
Loopback is the default binding. Supply TLS separately for remote access.

```powershell
$env:AGENT_S_UI_TOKEN = (Get-Content "$env:LOCALAPPDATA\AgentS\ui-token").Trim()
curl.exe http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $env:AGENT_S_UI_TOKEN"
```

Create a task with `POST /api/tasks`:

```json
{"instruction":"Inspect the open window","config":{"approval":"dry_run","max_steps":1}}
```

Approve using values from `GET /api/tasks/{id}`:

```json
{"step":1,"approval_token":"<32-character token>","approved":true}
```

`model="agent-s3"` in `/v1/chat/completions` uses the same manager. Manual mode
is rejected there: use `/api/tasks` or MCP. Responses include `X-Agent-S-Task-ID`.
Streaming emits live, stable-ID chunks. A disconnected `/v1` agent client
requests cancellation. Other model IDs proxy to the planner, preserving upstream
errors/statuses and request IDs with incremental streaming.

Native Anthropic planning is supported, but not OpenAI proxying/discovery.
Azure proxying uses deployment IDs, `api-key`, and API version; enter Azure IDs
manually. LM Studio grounding needs the correct UI-TARS weights and mmproj.

## MCP

Run `agent_s_mcp`. Its 12 tools reuse the local token automatically.
`approve_step` requires task ID, step, and approval token from task status.
Saved-key resolution happens on the HTTP backend, without returning secrets.
Autostart is serialized and supports only local HTTP URLs without paths.
Startup diagnostics go to `ui-startup.log`. Disable spawning with
`AGENT_S_UI_AUTOSTART=0`. Remote URLs require explicit `AGENT_S_UI_TOKEN`;
HTTPS/path-prefixed backends must be started separately.

## Checks

From `gui_agents/s3/ui`: `npm ci`, `npm run typecheck`, `npm run build`, `npm test`.
From the repository root:

```text
python -m compileall -q gui_agents/s3
python -m unittest discover -s tests -v
```

Tests use synthetic models/controllers/backends. The Windows Tk test keeps its
window withdrawn. Passing tests does not establish LM Studio quality or live
desktop task reliability; validate a short manual task before auto-execution.
