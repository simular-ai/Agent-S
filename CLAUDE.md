# Agent-S — Project Rules

GUI automation framework (Simular AI `gui_agents`), Python 3.12.
Active branch: `feat/agent-s3-foundation-orchestration-observability`.
s3 = current version (CLI `agent_s=gui_agents.s3.cli_app:main`).

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Env-var reference

All gates default OFF (unset = stub/subprocess path). Set to `1` to enable.

| Env var | Default | Effect |
|---|---|---|
| `AGENT_S3_USE_DOCKER` | unset | CLI agent runs LLM code in a Docker sandbox (`DockerExecutor`) instead of host subprocess. Needs docker daemon + SDK. |
| `AGENT_S3_USE_MEMORY` | unset | CLI agent queries `VectorMemory` at cycle start + saves winning trajectories at "done". Loads ~90MB model on first call. |
| `AGENT_S3_USE_HEALING` | unset | CLI agent calls `SelfHealingEngine.diagnose` on the except path; if recovery found, exec corrected action + continue. Needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. |
| `AGENT_S3_API_TIER4` | unset | API `POST /tasks` routes through `_tier4_handler` (Docker→VectorMemory→Observability cycle) instead of the stub handler. Needs docker + memory + keys. |
| `AGENT_S3_CHROMA_URL` | unset | `VectorMemory` auto-promotes to `provider="remote"` (HttpClient to a `chroma run` server). Multi-process safe. Unset = local PersistentClient (single-writer, 1 process). |
| `AGENT_S3_SHUTDOWN_TIMEOUT` | `30` | Seconds the API lifespan waits for worker threads on shutdown before abandoning them. |
| `AGENT_S3_CONTEXT_ID` | (set by code) | Propagated into Docker env + ChromaDB metadata for trace correlation. Set per-request by the framework, not by the user. |

See `docs/TIER4.md` for the full tier4 onboarding checklist.