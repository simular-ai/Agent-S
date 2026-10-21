# OpenCode harness

Builder (`builder.py`) and in-VM task agent (`evaluator.py`) built on the
OpenCode CLI release binary.

## Upstream version

| | |
|---|---|
| Release | OpenCode `v1.18.34`, `opencode-linux-x64.tar.gz` from `https://github.com/anomalyco/opencode/releases` |
| Archive SHA-256 | `0f22479647226d1d2dd99595d20082ee7bda3870b62dc6a90b41efc1a71d7e9a` |
| Local cache | `osworld_setup/s4/vm_assets/opencode/1.18.34/opencode`, built by `osworld_setup/s4/scripts/assets/prepare_opencode_runtime.py` |

The pins live in `source.py` (`OPENCODE_VERSION`, `RELEASE_ARCHIVE_SHA256`,
`RELEASE_URL`), which also checks the binary's reported version and the cache
metadata before the Builder starts or any VM is launched.

## What differs from upstream

The binary is used unmodified. It is configured, supervised and extended
through OpenCode's own config file and plugin interface. Every setting is read
from `SHADOW_OPENCODE_*` variables only, so a Prime run never picks one up by
accident; see `osworld_setup/s4/docs/OPENCODE_HARNESS.md` for the full list.

- `config.py` writes the `opencode.json` that the Builder and the task VM
  share: provider and model, context window and compaction, disabled tools
  (`SHADOW_OPENCODE_DISABLED_TOOLS`), and the plugins below.
- `runner.py` supervises the CLI:
  - takes file paths, never prompt text, in argv;
  - enforces the idle timeout (`SHADOW_OPENCODE_IDLE_TIMEOUT`) and the step
    limit (`SHADOW_OPENCODE_BUILDER_MAX_STEPS`);
  - continues a reply cut off at the output limit, up to
    `SHADOW_OPENCODE_LENGTH_CONTINUATIONS` times;
  - retries a malformed tool call, up to `SHADOW_OPENCODE_TOOLCALL_RECOVERIES`
    times;
  - cleans up only the CLI's own process group.
- `hermes_sanitizer.js` is a plugin for prompt-emulated (Hermes
  `<tool_call>`) endpoints such as the LiteLLM proxy in
  `osworld_setup/s4/scripts/proxy/`. It strips malformed tool-call text from
  history, so the model does not copy it in later replies. Enabled with
  `SHADOW_OPENCODE_HERMES_TOOL_FORMAT=1`.
- `tool_limits.js` is a plugin that caps the timeout the model can pass to the
  bash tool (`SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS`). Without the cap, a GUI app
  launched in the foreground keeps the call open until the supervisor's idle
  timeout ends the episode.
