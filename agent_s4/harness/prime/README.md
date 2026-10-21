# Prime Agent harness

Builder (`builder.py`) and in-VM task agent (`evaluator.py`) built on the
official Prime Agent CLI.

## Upstream version

| | |
|---|---|
| Project | [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) |
| Commit | `a9b5d88b50db50860088772a408e867f8071b507` |
| CLI package | `prime-agent-0.8.0-beta.546.1.a9b5d88.tgz` (SHA-256 `830d2b10656a7e04bece4b0d895b66fa0aa86e04bfc2088d03b00c27ef015831`) |
| Node.js | 22.23.2 (official `linux-x64` build) |
| Python runtime | vendored from the same commit in `osworld_setup/s4/agent_harness/prime-agent-runtime/` (see its `PROVENANCE.md`) |

The pins live in `source.py` (`PINNED_PRIME_AGENT_COMMIT`,
`PRIME_AGENT_VERSION`, `PRIME_AGENT_ARTIFACT_SHA256`, `NODE_VERSION`), which
also checks them before any VM is launched: the vendored runtime's commit
marker and per-file digests, the CLI tarball hash, and the offline VM runtime
built by `osworld_setup/s4/scripts/assets/cache_prime_agent_assets.py`.

## What differs from upstream

The CLI and the vendored Python runtime are used unmodified on the host. The
agent loop, sessions, tools and automatic context compaction stay Prime's own.
Everything below is added around it, or applied to the VM copy only.

Added around the CLI:

- `runner.py` relays `prime-agent --mode json` output unchanged and appends
  the terminal `result` record the shared Builder supervisor expects.
- `step_runner.py` does the same in the VM and enforces a model-step ceiling.
- `compaction.py` and `context_window.py` translate an interface's
  `context_compaction.json` into the settings Prime reads.
- `openrouter.py` holds provider/model configuration.
- `temperature.py` is an opt-in `before_provider_request` extension
  (`SHADOW_PRIME_AGENT_TEMPERATURE`). Prime itself sends no temperature.

Patched into the installed runtime on each task VM (`evaluator.py` applies
both at install time):

- `overflow_patch.py` makes Prime's overflow detector recognise the vLLM
  "decoder prompt ... is longer than the maximum model length" error, so that
  error triggers Prime's own compact-and-retry instead of a generic retry.
- `ipython_timeout_patch.py` adds a retriable 60-second limit to each IPython
  cell. The overlong cell is interrupted and the model is told to retry with
  smaller work; the episode continues.
