# OpenCode harness

`opencode` is a second harness for both the Builder and the in-VM task agent,
alongside the default `prime_agent`. It runs the pinned OpenCode CLI
(**1.18.34**) under a small Python supervisor. OpenCode uses its native tools
(bash, read, edit, ...) rather than Prime's persistent IPython tool, so its
scores are a separate experimental condition: give an OpenCode run its own
namespace and never resume a Prime run with it.

## Files

| file | role |
|---|---|
| `agent_s4/harness/opencode/source.py` | version / release-archive pin, host binary and VM-cache validation |
| `agent_s4/harness/opencode/config.py` | builds `opencode.json` + runner settings, shared by Builder and task agent |
| `agent_s4/harness/opencode/runner.py` | standalone supervisor copied next to the CLI on the host or VM |
| `agent_s4/harness/opencode/hermes_sanitizer.js`, `tool_limits.js` | OpenCode plugins (see the settings below) |
| `agent_s4/harness/opencode/builder.py` | host-side Builder, inside the same Landlock sandbox as Prime |
| `agent_s4/harness/opencode/evaluator.py` | VM installer, agent command and artifact validation |
| `scripts/assets/prepare_opencode_runtime.py` | builds the host binary and `vm_assets/opencode_runtime.tar.gz` |

### Settings

All new environment settings start with `SHADOW_OPENCODE_`; Prime settings are
not inherited. Explicit environment overrides take precedence over the interface
policy. Configuration and runner settings are saved with each Builder log and
Evaluator episode's artifacts, and session continuation rejects config drift.

| Setting | OpenCode mapping / default |
|---|---|
| `SHADOW_OPENCODE_TEMPERATURE` | Native agent `temperature`, including compaction; optional, finite 0–2 |
| `SHADOW_OPENCODE_TOP_P` | Native agent `top_p`, including compaction; optional, finite 0–1 |
| `SHADOW_OPENCODE_CONTEXT_WINDOW` | Model `limit.context` and `limit.input`; interface `context_window_tokens`, then 131072 |
| `SHADOW_OPENCODE_MAX_OUTPUT_TOKENS` | Model `limit.output` and runtime output ceiling; 32000 |
| `SHADOW_OPENCODE_RESERVE_TOKENS` | `compaction.reserved`; interface `reserve_tokens`, then 16384, raised to at least output + 4096 |
| `SHADOW_OPENCODE_KEEP_RECENT_TOKENS` | `compaction.preserve_recent_tokens`; interface `keep_recent_tokens`, then 20000 |
| Interface `enabled` | `compaction.auto`; default true. Disabling compaction does not remove the context limit |
| Interface `summary_instructions` | Compaction agent prompt |
| Interface `summary_max_tokens` | Separate native compaction model alias, same upstream model ID, lower output cap; positive and no larger than normal output |
| Interface `min_compact_tokens` | No equivalent; warns explicitly rather than claiming it is applied |
| `SHADOW_OPENCODE_SUMMARY_MAX_TOKENS` | Overrides interface `summary_max_tokens` for the compaction alias. Prime never applies that field |
| `SHADOW_OPENCODE_HERMES_TOOL_FORMAT` | `1` for prompt-emulated (Hermes `<tool_call>`) endpoints: appends a tool-call format note to every system prompt and loads `opencode_hermes_sanitizer.js`, which strips leaked malformed tool-call text from the history sent to the model (stored trajectories keep it). Default off |
| `SHADOW_OPENCODE_DISABLED_TOOLS` | Comma list of native tools withheld from the agent (`write`, `edit`, `todowrite`, `webfetch`, `skill`, `glob`, `grep`). `write` and `edit` share one native permission, so naming either withholds both. Under Hermes emulation `write` broke JSON on ~9% of calls vs ~0.7% for bash |
| `SHADOW_OPENCODE_TOOLCALL_RECOVERIES` | Hermes endpoints: when the proxy gives up on an unparseable tool call (HTTP 422 "No valid tool call"), resume the same session with a corrective prompt up to this many times per episode, sharing the wall/step budget. Recorded as `opencode_toolcall_error` / `opencode_toolcall_recovery` events. Default 0 (422 stays fatal, as in Prime) |
| `SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS` | Caps the `timeout` the model may pass to the bash tool (plugin `tool_limits.js`, `tool.execute.before`). The pinned CLI otherwise honours any value and is silent while a tool runs, so a GUI launched with stdout attached held a call open until the supervisor's idle watchdog ended the episode. At most 1800000. Default unset (uncapped) |
| `SHADOW_OPENCODE_REQUEST_TIMEOUT` | Provider total/header/chunk timeout in seconds; 1800 |
| `SHADOW_OPENCODE_IDLE_TIMEOUT` | Supervisor inactivity deadline; request timeout + 120, must exceed request timeout |
| `SHADOW_OPENCODE_LENGTH_CONTINUATIONS` | Explicit same-session retry after `finish_reason=length`; default 0, opt in to change the protocol |
| `SHADOW_OPENCODE_BUILDER_MAX_STEPS` | Builder completed-model-step budget; 200 |
| Evaluator `--max-steps` | Evaluator completed-model-step budget; 200 if unspecified |
| `SHADOW_OPENCODE_BASE_URL` | Optional endpoint override. `openai/<id>` otherwise requires `OPENAI_BASE_URL` |
| `SHADOW_OPENCODE_VM_BASE_URL` | VM-only endpoint override; takes precedence over the shared OpenCode endpoint |
| `SHADOW_OPENCODE_BIN` | Explicit host binary; exact pinned version is required |

For VM evaluation with `openai/<id>`, endpoint selection is
`SHADOW_OPENCODE_VM_BASE_URL`, then `SHADOW_OPENCODE_BASE_URL`, then the existing
`SHADOW_EVOL_VM_OPENAI_BASE_URL`, then `OPENAI_BASE_URL`. This allows a VPC model
address inside AWS while the host Builder continues to use the public address.
The legacy VM OpenAI setting does not redirect OpenRouter or Parasail.

Unspecified sampling settings keep OpenCode/provider defaults; they do not
establish numerical equivalence to Prime's provider defaults. Set temperature
and top-p explicitly for a controlled comparison. Top-k is not wired into this
Chat Completions integration.

The pinned runtime computes its compaction threshold from `limit.input - reserved`
when input is present. With context 131072 and output 32000, the minimum reserve
is 36096 and the threshold is 94976 tokens. Merely copying Prime's reserve field
without declaring `limit.input` would not implement that threshold. Native
compaction is usage-driven, not an absolute guarantee against an arbitrarily
large single tool result. Native pruning is explicitly disabled.

The JSON event stream does not expose a reliable model-versus-tool-start phase.
The watchdog therefore uses a conservative inactivity limit longer than a model
request, rather than guessing that a buffered response is a hung tool. Wall-clock
and completed-step budgets remain independently enforced. Completed-step counts
include any compaction steps surfaced by the native CLI.

### Failure and process handling

- Task text goes from a file to stdin. Neither the Python supervisor nor the
  OpenCode CLI receives it in argv. System prompts live in the effective config.
- `PWD` and `--dir` are set explicitly. OpenCode consults inherited `PWD`; merely
  setting `Popen(cwd=...)` can make its tools operate in the wrong directory.
- A fresh process group owns the CLI and ordinary tool descendants. Timeout,
  termination, and normal cleanup signal that group by ID; no global `pkill` is
  used. Deliberately detached processes are outside this cleanup boundary.
- CLI nonzero exits, provider errors, malformed JSONL, empty output, and missing
  completion become failures. A reply cut off at the output cap is resumed up to
  `SHADOW_OPENCODE_LENGTH_CONTINUATIONS` times per episode; after that the episode
  ends as `length_limit`, which is scoreable, as Prime ends such an episode and
  lets the grader score the VM state. Scoreable endings (`completed`, `step_limit`,
  `length_limit`, wall timeout 124) still require evidence of completed model steps. The terminal result must agree with the exit marker.
- Follow-ups select the exact session ID, never the latest session. Length
  continuations share the original wall/step budget; benchmark follow-ups preserve
  the shared step count. Existing V2 wall budgeting continues to govern phases.
- API keys use `AgentCommand.secret_env`, not interpolated shell commands or
  persistent key fallback files. Config uses environment references.
- Project config, external skills, unrelated user config, automatic updates,
  model catalog refreshes, snapshots, and interactive permission questions are
  disabled/isolated for batch runs. Subagent creation is disabled so it cannot
  bypass the top-level step budget. The existing Builder filesystem sandbox stays.

This prevents accidental matching of task/application names; it does not make
same-user processes immune to an explicit `kill <agent-pid>` or `pkill python`.
It also does not turn tool execution into a new process/network security sandbox.

### Provider compatibility

`openai/<deployment>` is mapped internally to `shadow-openai/<deployment>` using
`@ai-sdk/openai-compatible`. The pinned OpenCode built-in `openai` loader forces
the Responses API; overriding its npm package alone fails before any request.
The independent provider uses Chat Completions and preserves the upstream ID.
`openrouter/<slug>` and `parasail/<slug>` are also supported, with explicit model
capabilities and their corresponding API-key environment variables. Local runtime
smoke tests cover all three prefixes against a local Chat Completions server.
The AWS smoke procedure below additionally uses the configured real model.

## Prepare and select the harness

```bash
python scripts/assets/prepare_opencode_runtime.py
```

The script downloads the official release archive, checks its pinned SHA256,
extracts the host binary to `vm_assets/opencode/1.18.34/opencode`, and builds the
VM tarball with version/checksum metadata. Task VMs install
`/opt/shadow-opencode/1.18.34/opencode` from that tarball and never download it.

Both launch scripts select the harness by environment variable; the default is
`prime_agent`:

```bash
# evolution: Builder and task agent both run OpenCode
SHADOW_EVOL_HARNESS=opencode bash scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh \
    --namespace <new-namespace> --yes

# held-out evaluation of a variant (results go to a separate _opencode tree)
SHADOW_EVOL_TEST_HARNESS=opencode NAMESPACE=<ns> VARIANT=<variant> \
    bash scripts/eval/run_osworld_v2_test_split_variant.sh --yes
```

Directly, the CLI takes `--builder-harness opencode` and an island with
`evaluator_harness=opencode`; the eval worker takes
`--harness opencode --model openai/<deployment>` with `--max-steps` and
optionally `--no-time-limit`.

Unspecified sampling settings keep OpenCode/provider defaults. For a controlled
comparison set them explicitly, e.g.:

```bash
export SHADOW_OPENCODE_CONTEXT_WINDOW=131072
export SHADOW_OPENCODE_MAX_OUTPUT_TOKENS=32000
export SHADOW_OPENCODE_TEMPERATURE=0.6
export SHADOW_OPENCODE_TOP_P=0.95
```

## Validation

Unit tests cover configuration validation, dispatch, credential separation,
artifact validation, malformed/empty/error exits, signal/timeout cleanup, exact
session continuation, and rejection of changed settings on resume. The opt-in
integration tests run the **real pinned binary** against a local HTTP/SSE fake
provider, with no external model calls:

```bash
OPENCODE_TEST_BINARY="$PWD/vm_assets/opencode/1.18.34/opencode" \
    python -m pytest -q tests/unit/test_opencode_live_runtime.py
```

Upstream references: [configuration](https://opencode.ai/docs/config/),
[agent sampling](https://opencode.ai/docs/agents/),
[pinned CLI stdin and directory handling](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/opencode/src/cli/cmd/run.ts),
[pinned compaction threshold](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/opencode/src/session/overflow.ts),
[pinned config fields](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/core/src/v1/config/config.ts).
