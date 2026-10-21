# Agent S4: Shadow Interface Evolution

An evolutionary loop that evolves Python **shadow interfaces** — small tool
libraries a computer-use agent imports to operate a desktop — and scores each
variant on the **OSWorld-V2** benchmark running on parallel AWS VMs.
ScienceBoard has its own workspace and README in
[`scienceboard_setup/s4/`](../../scienceboard_setup/s4/README.md).
The code still accepts `--benchmark osworld` (the original OSWorld, which V2's
runner builds on), but no v1 task splits ship; pass `--train-set`/`--val-set`
explicitly to use it.

Each step a **Builder** agent edits the current interface on the host; the
**task agent** then runs the benchmark tasks on fresh VMs with that interface
installed, and the score becomes the variant's fitness. By default both agents
are [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent), pinned and
run offline on the task VMs (see [docs/PRIME_AGENT_HARNESS.md](docs/PRIME_AGENT_HARNESS.md)).
[OpenCode](https://opencode.ai) is available as an alternative harness for both
(see [OpenCode harness](#opencode-harness)).

One orchestrator process drives everything:
`python -m agent_s4.cli run` (see `agent_s4/cli.py`). The launch
scripts under `scripts/eval/` wrap it with the exact settings of the published
runs.

The code is the `agent_s4` package; everything else — scripts, prompts,
seeds, task splits, data, and every run artefact — lives in this directory,
`osworld_setup/s4/`, which is the working directory for all commands below
(`agent_s4/paths.py`; override with `SHADOW_EVOL_PROJECT_ROOT`).

## Layout

```
agent_s4/                            # the package, at the Agent-S repo root;
                                     # its own distribution (pyproject.toml)
  cli.py driver.py orchestrator.py   # orchestrator, driver, DB, scoring
  operators/ tournament.py
  harness/prime/                     # Prime Agent: builder.py (host-side Builder),
                                     # evaluator.py (in-VM task agent), runner, runtime
  harness/opencode/                  # OpenCode: builder.py, evaluator.py, runner, config
  harness/ubuntu_vm.py               # eval subprocess wrapper (all benchmarks)
  harness/builder_common.py          # Builder subprocess + Landlock sandbox, shared
  eval/osworld.py                    # shared eval worker (episodes, results)
  eval/osworld_v2.py                 # OSWorld-V2 task loading / scoring
  eval/scienceboard_prime.py         # ScienceBoard VMs / scoring
  osworld_v2_preflight.py            # offline gate run before every V2 run
  paths.py                           # package / shared-root / project-root anchors
osworld_setup/s4/                    # project root (this directory), also the
                                     # shared root ScienceBoard runs use
  agent_harness/prime-agent-runtime/ # vendored, pinned Prime Python runtime
  seeds/combined/                    # seed interface every run starts from
  prompts/                           # Builder prompts (+ addenda)
  eval_sets/                         # task splits (train / val / test)
  data/README.md                     # where to download the 30 synthetic training tasks
  config/osworld_v2_release_pin.json # pinned V2 release, task digest, splits
  scripts/eval/                      # launch + held-out eval scripts
  scripts/assets/cache_prime_agent_assets.py  # builds the offline VM runtime
  scripts/assets/prepare_opencode_runtime.py  # same, for the OpenCode harness
  scripts/proxy/                     # LiteLLM tool-calling proxy (optional)
  scripts/ops/, scripts/vm/          # VM cleanup, infra-failure reruns, scratch VMs
  tests/                             # the whole S4 suite, ScienceBoard included
scienceboard_setup/s4/               # ScienceBoard workspace: scripts, prompts, splits, data
```

## Setup

Linux host with `git`, `rsync`, `sqlite3`, Python 3.12 and
[`uv`](https://docs.astral.sh/uv/). Builder isolation uses Landlock, so the
kernel must support it (the preflight checks).

```bash
git clone https://github.com/simular-ai/Agent-S.git
cd Agent-S/osworld_setup/s4

# The launch scripts use the interpreter at osworld_setup/s4/.venv.
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ../../agent_s4          # agent_s4 (separate from gui-agents)
uv pip install -r requirements.txt
```

The launch scripts put the Agent-S root on `PYTHONPATH` themselves, so the
editable install only matters for running `python -m agent_s4...` by hand.

### Benchmarks

**OSWorld-V2** — two release-pinned clones inside `osworld_setup/s4/`:

```bash
git clone https://github.com/xlang-ai/OSWorld-V2.git OSWorldV2_Base
git -C OSWorldV2_Base checkout v2026.08.08
git clone https://github.com/xlang-ai/OSWorld-V2.git OSWorldV2_Evol
git -C OSWorldV2_Evol checkout -b evol v2026.08.08
uv pip install -e OSWorldV2_Base         # provides desktop_env
```

OSWorld-V2 evals import the island clone's own `desktop_env` (they run with the
clone as their working directory); host-side tooling uses the installed one.

The 108 V2-native task files and their assets are gated on Hugging Face
(`xlangai/osworld_v2_tasks`, `xlangai/osworld_v2_assets_gated`). Accept both
gates, `hf auth login`, then use V2's downloaders:

```bash
REL=$PWD/OSWorldV2_Base/benchmark_releases/osworld-v2-2026.08.08.json
(cd OSWorldV2_Base && python scripts/tools/download_osworld_v2_tasks.py \
     --benchmark-release "$REL" --target-dir "$PWD/evaluation_examples/task_class")
(cd OSWorldV2_Base && python scripts/tools/download_osworld_v2_assets.py \
     --benchmark-release "$REL")     # -> OSWorldV2_Base/cache/osworld_v2_assets
```

Put the task files in `OSWorldV2_Base/` **only**, never in `OSWorldV2_Evol/`:
the Builder works inside clones of `OSWorldV2_Evol`, and task files there
would expose evaluator logic. The preflight rejects that layout.

### Prime Agent runtime

```bash
python3 scripts/assets/cache_prime_agent_assets.py
```

builds `vm_assets/prime_agent_runtime.tar.gz` (Node, the pinned Prime Agent
package, and the vendored Python runtime) for the task VMs, which never reach
npm. The Builder runs the same pinned CLI on the host: set
`SHADOW_PRIME_AGENT_BIN`, or put `prime-agent` on `PATH`. Both are validated
before any VM is launched. Details:
[docs/PRIME_AGENT_HARNESS.md](docs/PRIME_AGENT_HARNESS.md).

### OpenCode runtime (optional)

```bash
python scripts/assets/prepare_opencode_runtime.py
```

downloads the pinned OpenCode 1.18.34 release, checks its SHA256, and builds
the host binary under `vm_assets/opencode/` plus `vm_assets/opencode_runtime.tar.gz`
for the task VMs. Only needed for `SHADOW_EVOL_HARNESS=opencode`.

### Credentials

Copy [`.env.example`](.env.example) to `.env`. It covers the model endpoint
(any OpenAI-compatible endpoint, OpenRouter or Parasail), the OSWorld-V2
grader / user simulator, V2 website and GitLab settings, and AWS resource
IDs. AWS credentials come from your usual AWS config/SSO, not from `.env`.

If your vLLM deployment was started without tool calling, front it with the
LiteLLM proxy in `scripts/proxy/`: set `HERMES_UPSTREAM_MODEL`,
`HERMES_UPSTREAM_BASE_URL`, `HERMES_UPSTREAM_API_KEY` and `LITELLM_MASTER_KEY`,
run `scripts/proxy/run_litellm.sh` (it needs LiteLLM in `.venv-litellm/`), and
point `OPENAI_BASE_URL` at it with `SHADOW_PRIME_MODEL_REQUEST_TIMEOUT=1800`.

## Running

### OSWorld-V2 evolution

Fitness is 15 official validation tasks. The Builder learns from the
trajectories of 30 ungraded synthetic training tasks, published on
[Hugging Face](https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks). Download them into `data/` as described in
[data/README.md](data/README.md); the launcher unpacks the archive on first
use. Nine of them open the OSWorld-V2 mock websites, so point the package at
your deployment once before the first run (`WEBSITE_HOST_SUFFIX` gives
`<scheme>://<app>.<suffix>`; `--origin` overrides one app):

```bash
curl -L -o data/osworld_synthetic_30.tar.xz \
    https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks/resolve/main/osworld_synthetic.tar.xz
tar -xJf data/osworld_synthetic_30.tar.xz -C data
python3 scripts/eval/prepare_osworld_synthetic.py [--scheme http] [--origin APP=URL ...]
```

Then:

```bash
bash scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh --local-check     # offline checks
bash scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh --preflight-only  # + pinned V2 gate
bash scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh --yes --model <provider/model>
```

`--help` lists the options; `SHADOW_EVOL_*` environment variables override
step count, concurrency, timeouts, scheduler and parent selection. The offline
gate on its own is `python -m agent_s4.osworld_v2_preflight --help`.

### Held-out evaluation of a variant

Each variant is a git tag `evol/<namespace>/<variant>` in the island clone;
the scripts materialize it from the tag into an isolated directory.

```bash
# default: the 68-task held-out manifest; any eval_sets/ file works
NAMESPACE=<ns> VARIANT=if-011 bash scripts/eval/run_osworld_v2_test_split_variant.sh \
    --tasks-file eval_sets/osworld_v2_full_minus_val_93.json --yes
```

`run_osworld_v2_test_split_variant.sh` also has a no-interface mode
(`SHADOW_EVOL_NO_TOOL=1`) that runs Prime Agent with nothing deployed.

### OpenCode harness

Both launch scripts run Prime Agent unless told otherwise:

```bash
SHADOW_EVOL_HARNESS=opencode bash scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh \
    --namespace <new-namespace> --yes
SHADOW_EVOL_TEST_HARNESS=opencode NAMESPACE=<ns> VARIANT=if-011 \
    bash scripts/eval/run_osworld_v2_test_split_variant.sh --yes
```

OpenCode uses its native tools instead of Prime's IPython tool, so it is a
separate experimental condition: use a new namespace, never resume a Prime run
with it. `SHADOW_OPENCODE_*` variables set sampling, context and timeouts; see
[docs/OPENCODE_HARNESS.md](docs/OPENCODE_HARNESS.md).

### Stopping, resuming, cleanup

- Resume: re-run the same command; bootstrap is idempotent and reads the
  namespace's SQLite DB (`shadow_evolution_<benchmark>_<ns>.db`).
- Stop: `bash scripts/eval/stop_evolution.sh <namespace> [--terminate-vms]`.
  Killing only the driver leaves the eval subprocess and its VMs running.
- Orphaned VMs: `python scripts/ops/list_orphan_scratch_vms.py`.
- Infrastructure zeros: `python scripts/ops/rerun_infra_failures.py --help`.

## What a run produces

| path | contents |
|------|----------|
| `shadow_evolution_<benchmark>_<ns>.db` | variants, scores, per-task results, costs |
| `OSWorldV2_Evol_<ns>/` | island clone; each variant is a git tag |
| `eval_cache_<benchmark>_<ns>/<variant>/` | per-task trajectories, results, summaries |
| `builder_logs_<benchmark>_<ns>/` | Builder session logs |

Trajectories record the VM environment, which can include API keys. Redact
them before sharing.

## Tests

```bash
pytest -q          # from osworld_setup/s4
```

No test contacts AWS or a model provider.
