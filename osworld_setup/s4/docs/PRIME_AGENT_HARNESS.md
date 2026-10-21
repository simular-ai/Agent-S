# Prime Agent harness

The `prime_agent` harness is available as both the host-side evolution builder
and the OSWorld/OSWorld V2 task agent. The integration deliberately launches
Prime Agent's official headless JSON CLI instead of reimplementing its agent
loop.

## Pinned upstream

- Repository: `PrimeIntellect-ai/prime-agent`
- Commit: `a9b5d88b50db50860088772a408e867f8071b507`
- Release: `0.8.0-beta.546.1.a9b5d88`
- Official release SHA256:
  `830d2b10656a7e04bece4b0d895b66fa0aa86e04bfc2088d03b00c27ef015831`

`agent_harness/prime-agent-runtime` directly contains the pinned Python RLM
runtime, including `src/rlm/harness.py`. It also contains byte-identical audit
copies of the upstream compaction/session/settings sources. The source
manifest records every required file digest, and preflight rejects a modified,
missing, or unexpected importable file.

The Python `harness.py` stores persistent memory, skills, subagent specs, and
refinement events. Automatic context compaction is implemented by the
TypeScript Prime Agent session engine. The runtime therefore uses the matching
official CLI artifact as well as the vendored Python kernel package.

## Auto-compaction contract

Both builder and evaluator:

- keep sessions enabled (they never pass `--no-session`);
- map interface `enabled`, `reserve_tokens`, and `keep_recent_tokens` into
  Prime's actual settings (defaults: enabled, 16,384 reserve, 20,000 recent);
- for self-hosted OpenAI-compatible and OpenRouter requests, raise the effective
  reserve to at least the 32,000 normal-turn output budget plus a 4,096-token
  estimation margin: **36,096 tokens**. Preserve the 32,000 output cap;
- append benchmark policy to the upstream Prime RLM system prompt rather than
  replacing it;
- preserve native `compaction_start` and `compaction_end` JSON events.

At the Parasail deployment's 262,144-token window, the resulting threshold is
226,048 tokens. Prime compares against the last successful assistant's **total
usage plus estimated trailing messages**, not input alone. The adapters reject
nonpositive token settings and a reserve plus recent-history budget that fills
the known window. The existing `SHADOW_PRIME_CONTEXT_WINDOW_SOURCE` contract
still controls the interface's window override. An override describes the
window Prime assumes; it cannot increase a provider's physical context limit.

The mini-SWE fields `min_compact_tokens`, `summary_max_tokens`, and
`summary_instructions` have no equivalent settings in the pinned Prime runtime;
they produce a warning and are not forwarded as if supported. Prime owns its
summary prompt and requests `floor(0.8 * reserveTokens)` summary tokens (28,876
at the new floor), plus `floor(0.5 * reserveTokens)` for a split-turn prefix
when needed. Raising reserve therefore also raises summary budgets. The margin
does not bound arbitrarily large tool results or summary serialization: retained
provider overflow errors and Prime's compact-and-retry path are still required.
Other provider adapters keep their existing reserve default because their
thinking-budget rules differ. These changes affect newly generated commands;
they do not alter saved experimental results or already running processes.

OSWorld V2 follow-up episodes use Prime Agent's official `--continue` path and
reuse the same session directory, so compacted history and persistent kernel
state survive an `ASK_USER:` round trip.

Prime Agent performs threshold compaction, overflow compaction-and-retry, and
post-compaction continuation itself. Its IPython kernel remains available
across compaction. The adapters only isolate configuration, relay trajectories,
and translate the terminal result into this repository's harness format.

## Host builder setup

The builder resolves `SHADOW_PRIME_AGENT_BIN`, then `prime-agent` on `PATH`,
and refuses a version other than the pin above. Install the exact official
artifact with Node 22.8 or newer, or point `SHADOW_PRIME_AGENT_BIN` at an
equivalent verified installation.

The builder stores Prime Agent configuration and sessions inside the island's
private `.builder_tmp` directory. OSWorld V2 Landlock grants read access to the
complete Prime Agent package root so its bundled skills and Python runtime
remain available.

## Offline VM runtime

Build the OSWorld runtime cache on Linux x86_64:

```bash
python3 scripts/assets/cache_prime_agent_assets.py
```

The builder downloads and verifies the pinned Prime Agent artifact and the
pinned official Node.js 22.23.2 standalone archive. It copies a complete
uv-managed CPython 3.11.16 distribution, installs the vendored runtime, the
locked kernel environment in `agent_harness/prime-agent-kernel`, and every
Python skill bundled by the pinned CLI artifact. It then writes:

- `vm_assets/prime_agent_runtime.tar.gz`
- `vm_assets/prime_agent_runtime.commit`
- `vm_assets/prime_agent_runtime.sha256`

For a cache-only build, supply both already-downloaded artifacts:

```bash
python3 scripts/assets/cache_prime_agent_assets.py --offline \
  --artifact /path/to/prime-agent.tgz \
  --node-artifact /path/to/node-v22.23.2-linux-x64.tar.xz
```

Evaluation VMs never install from the network. Before provisioning, the
orchestrator validates the vendored source, cache fingerprint, archive digest,
and safe layout. After extraction it probes the exact commit, fingerprint,
Prime Agent version, Python interpreter, the full default kernel dependency
set, `rlm.harness`, and all bundled Python skill imports. The native JSONL must
also parse cleanly and contain both `agent_start` and `agent_end`; a CLI exit 0
cannot hide a kernel bootstrap failure.

## OSWorld behavior

The evaluator runs `prime-agent --mode json --offline` with task-local config
and session directories. Native JSON events are saved to
`/tmp/claude_task/trajectory.jsonl`, so OSWorld V2's `ASK_USER:` bridge remains
functional. The appended system policy teaches Prime Agent to operate the
desktop through Python subprocess calls and to load screenshots with the
bundled `attach_image` skill.

Provider credentials are staged in a mode-0600, short-lived env file before
launch. They do not appear in the long-lived shell/Prime Agent command line,
and the setup server's echoed command is never written to evaluator logs.

## Live validation

On 2026-08-25, AWS Task 17 was run with `claude-sonnet-5`, an `Owner`
instance tag, and the offline cache. Prime Agent exited cleanly and OSWorld V2
scored the task `0.5714285714` (4 of 7 route edges). Its native trajectory had
1,761 valid JSON events, 102 completed tool executions, a terminal
`agent_end`, repeated GUI screenshots, and successful `attach_image` calls.
The configured API key was absent from the saved trajectory. No compaction
event was expected in that run because the model context threshold was not
crossed; compaction remained enabled in the isolated settings and session.
