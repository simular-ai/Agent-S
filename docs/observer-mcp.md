# Agent S observation-only MCP

This runtime is a safety-first integration for Codex. It captures one isolated
1920x1080 VM display and asks Agent S-style hosted models for a typed next-action
proposal. It has no desktop action executor.

## Security boundary

- Run it only in the disposable VM built by `scripts/agent_s_vm/build_base.sh`.
- The MCP surface contains `status`, `observe`, `start_task`, `propose_next`, and
  `reset_task`. There is deliberately no execute tool.
- Model-authored action syntax is parsed as one AST expression. Only literal
  arguments to the proposal methods `click`, `type`, `scroll`, `hotkey`, `wait`,
  `done`, and `fail` are accepted.
- The observer does not import PyAutoGUI, the S3 code agent, LocalEnv, or a shell
  runner. API keys are forwarded at runtime and are not stored in the VM image.
- The SSH bridge disables shell xtrace before it reads credentials. Do not
  replace that behavior with debug logging that prints environment values.

## Required environment

The observer accepts generic OpenAI-compatible model configuration:

- `AGENT_S_MAIN_BASE_URL`, `AGENT_S_MAIN_API_KEY`, and `AGENT_S_MAIN_MODEL`;
- `AGENT_S_GROUND_BASE_URL`, `AGENT_S_GROUND_API_KEY`, and
  `AGENT_S_GROUND_MODEL`.

The older hosted configuration remains compatible: `OPENAI_API_KEY` is the
main-model key, while `HF_TOKEN` and `HF_ENDPOINT_URL` configure grounding.
When either generic base URL is the dedicated VM-to-host route
`http://10.0.2.2:18082/v1`, the bridge supplies a non-secret local placeholder
key because the OpenAI client requires a non-empty value.

The other tools can be smoke-tested without model credentials. Start the stdio
server with:

```bash
python -m gui_agents.s3.observer.mcp_server
```

Use the VM bridge and Codex configuration described in
`scripts/agent_s_vm/README.md` for the operational deployment.

## Health and release identity

The guest is usable only when all of these layers pass:

1. the QEMU process exists;
2. the localhost SSH forward accepts connections;
3. the MCP client initializes and sees exactly the five expected tools;
4. `observe` returns one valid 1920x1080 PNG;
5. model configuration is present before `propose_next` is called.

`scripts/agent_s_vm/status.sh` reports the cheap process, transport, and model
configuration layers. `scripts/agent_s_vm/status.sh --deep` adds a real MCP
initialization, tool-catalog, status, and observation canary. The deep check
needs permission to reach the localhost-only SSH forward; a restricted command
sandbox may report the transport as blocked even when the host can reach it.

Every newly built guest contains `/opt/agent-s/observer-build.json`. The MCP
`status` response reports its source commit, dirty flag, build timestamp, source
archive digest, and requirements-lock digest. A development server without
that file reports `build.status=development` instead of inventing an identity.

## Acceptance evidence

Live Codex MCP validation passed on 2026-07-14 against sealed observer build
`4902e9fe683fdb57e69e78ac81ebed3e90be3a8d`.

- The exposed tool catalog was exactly `status`, `observe`, `start_task`,
  `propose_next`, and `reset_task`; no desktop action or execution tools were
  exposed.
- `status` reported `mode=observation_only`, `desktop_actions_exposed=false`,
  `build.status=sealed`, `source_dirty=false`, no OpenAI or Hugging Face
  credentials configured, and both main and grounding API keys configured for
  the local endpoint.
- `observe` returned a valid 1920x1080 PNG.
- The task lifecycle passed: `start_task`, exactly one `propose_next`, status
  showing one active task and `proposal_count=1`, `reset_task`, then status
  showing no active task and `proposal_count=0`.
- The proposal was recorded as `risk_class=proposal_only` and was not executed.
- Ten repeatability cycles of `status` followed by `observe` passed with no
  transport errors. Mean latency was 17.7 ms for `status` and 61.9 ms for
  `observe`.
- All observations had stable screenshot SHA-256
  `08865b5fefe9692b4f7887929af5e35e255a89f36686909a19a65f5e79722f93`.

The earlier observed `Transport closed` failure is attributed to a stale Codex
MCP transport that survived a VM recycle. A cached client-side tool catalog only
proves that the tool names were previously loaded; it does not prove the stdio
transport is still live.
