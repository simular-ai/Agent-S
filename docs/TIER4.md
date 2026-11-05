# TIER 4 — Super Agent onboarding

TIER 4 turns Agent-S3 from a default subprocess agent into a sandboxed, memory-equipped,
self-healing agent. It is **fully env-gated** — every gate defaults OFF, so the base
agent is unchanged until you opt in. This doc lists every prerequisite in one place so
you don't have to grep three module docstrings.

## Prerequisites

| Need | Why | How to check |
|---|---|---|
| **Docker daemon running** | `DockerExecutor` runs LLM code in an ephemeral `python:3.11-slim` container | `docker info` succeeds |
| **Docker SDK installed** | `docker` Python package (import-guarded; loads without it, fails at instantiation) | `python -c "import docker"` |
| **An LLM API key** | `SelfHealingEngine` needs a VLM to diagnose failures | `OPENAI_API_KEY` (gpt-4o) **or** `ANTHROPIC_API_KEY` (claude-sonnet-5) set in env |
| **~90MB model download** (first run only) | `VectorMemory` default `provider="local"` loads `all-MiniLM-L6-v2` via sentence-transformers | first `AGENT_S3_USE_MEMORY=1` run takes ~10s extra; cached after |
| **First container pull** (first run only) | `python:3.11-slim` image pulled on first `DockerExecutor` use | first `AGENT_S3_USE_DOCKER=1` run takes ~30s extra; cached after |

## Env gates (all default OFF — set to `1`)

| Env var | Effect |
|---|---|
| `AGENT_S3_USE_DOCKER=1` | CLI runs LLM code in Docker sandbox (falls back to host subprocess if daemon/SDK unavailable) |
| `AGENT_S3_USE_MEMORY=1` | CLI queries `VectorMemory` at cycle start + saves winning trajectories at "done" |
| `AGENT_S3_USE_HEALING=1` | CLI calls `SelfHealingEngine.diagnose` on failure; if recovery found, execs corrected action + continues |
| `AGENT_S3_API_TIER4=1` | API `POST /tasks` routes through `_tier4_handler` (Docker→VectorMemory→Observability) instead of the stub |
| `AGENT_S3_CHROMA_URL=<url>` | `VectorMemory` uses a remote `chroma run` server (multi-process safe) instead of local PersistentClient |
| `AGENT_S3_SHUTDOWN_TIMEOUT=<sec>` | API shutdown grace period (default 30s) |

## First run (CLI)

```bash
# 1. daemon + key
docker info                                    # must succeed
export ANTHROPIC_API_KEY=sk-...                # or OPENAI_API_KEY

# 2. enable all three CLI gates
export AGENT_S3_USE_DOCKER=1 AGENT_S3_USE_MEMORY=1 AGENT_S3_USE_HEALING=1

# 3. run — first cycle is slower (model load + image pull, one-time)
agent_s
```

## First run (API tier4)

```bash
export AGENT_S3_API_TIER4=1 AGENT_S3_USE_DOCKER=1 AGENT_S3_USE_MEMORY=1
export ANTHROPIC_API_KEY=sk-...
uvicorn gui_agents.s3.api.main:app --port 8000

# POST /tasks now runs the full tier4 cycle
curl -X POST localhost:8000/tasks -H 'Content-Type: application/json' \
  -d '{"instruction":"open calculator","code":"```python\nimport pyautogui\npyautogui.press(\"win\")\n```"}'
```

## What each gate does NOT do

- `USE_DOCKER` unset → LLM code runs on the **host** via `subprocess` with full user privileges. The Docker sandbox is the mitigation; it is opt-in. Only enable the agent in trusted environments either way.
- `USE_MEMORY` unset → no trajectory recall; every cycle starts fresh.
- `USE_HEALING` unset → failures raise; no VLM diagnosis or corrected retry.
- `API_TIER4` unset → `POST /tasks` uses the stub handler (no Docker/VectorMemory cycle).

## Multi-worker note

The Docker reaper (`reap_orphans`) scopes containers by owner PID. With
`uvicorn --workers N` (N>1), a worker only knows its own PID and may reap another
worker's live containers. **Use a single worker** for tier4, or move container labels
to a per-process UUID (see `_alive_pids` docstring).

## Observability

- Logs: `logs/*.log` under logger `desktopenv.agent.*` (JSON, `context_id` correlated).
- Metrics: `ObservabilityManager` singleton (`track_task`/`track_action`).
- Slack: set `AGENT_S3_SLACK_WEBHOOK` for alert export.