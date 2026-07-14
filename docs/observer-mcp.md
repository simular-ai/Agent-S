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

`OPENAI_API_KEY`, `HF_TOKEN`, and `HF_ENDPOINT_URL` are required when calling
`propose_next`. `AGENT_S_MAIN_MODEL` defaults to `gpt-5-2025-08-07`, and
`AGENT_S_GROUND_MODEL` defaults to the Hugging Face TGI model name `tgi`.

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
5. hosted-model configuration is present before `propose_next` is called.

`scripts/agent_s_vm/status.sh` reports the cheap process, transport, and model
configuration layers. `scripts/agent_s_vm/status.sh --deep` adds a real MCP
initialization, tool-catalog, status, and observation canary. The deep check
needs permission to reach the localhost-only SSH forward; a restricted command
sandbox may report the transport as blocked even when the host can reach it.

Every newly built guest contains `/opt/agent-s/observer-build.json`. The MCP
`status` response reports its source commit, dirty flag, build timestamp, source
archive digest, and requirements-lock digest. A development server without
that file reports `build.status=development` instead of inventing an identity.
