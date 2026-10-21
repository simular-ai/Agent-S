# Agent S4 on ScienceBoard

The same shadow-interface evolution as [`osworld_setup/s4`](../../osworld_setup/s4/README.md),
scored on [ScienceBoard](https://github.com/OS-Copilot/ScienceBoard): six
scientific desktop apps (ChimeraX, GrassGIS, KAlgebra, Lean, TeXstudio,
Celestia), each official task graded by ScienceBoard's own programmatic
evaluator. Prime Agent is the Builder and the task agent; the OpenCode harness
is OSWorld-only.

This directory is the run workspace: ScienceBoard checkouts, prompts, task
splits, the synthetic training bank and every run artefact live here. What
both benchmarks share — the venv, the vendored Prime runtime, the VM asset
cache and the seed interface — stays in `osworld_setup/s4/`. The launchers
set `SHADOW_EVOL_PROJECT_ROOT` to this directory and
`SHADOW_EVOL_SHARED_ROOT` to `osworld_setup/s4` (see `agent_s4/paths.py`).

## Layout

```
scienceboard_setup/s4/
  scripts/eval/run_scienceboard_qwen_synthetic_50_15.sh  # evolution launcher
  scripts/eval/run_scienceboard_full_variant.sh          # all-169 eval of one variant
  scripts/eval/create_scienceboard_eval_sets.py          # builds the official splits
  prompts/                       # SB Builder prompts and addenda
  eval_sets/                     # scienceboard_* splits (val 15, test, full 169, ...)
  data/README.md                 # where to download the 50 synthetic training tasks
agent_s4/eval/scienceboard_prime.py   # SB VMs / scoring (package code)
```

## Setup

1. Do the OSWorld-side setup first ([`osworld_setup/s4/README.md`](../../osworld_setup/s4/README.md)):
   the venv at `osworld_setup/s4/.venv` and the Prime runtime
   (`scripts/assets/cache_prime_agent_assets.py`) are used from there.
2. In this directory, a reference ScienceBoard checkout with the official
   tasks, and a clone template for the Builder on branch `evol` (tagged at
   HEAD — the template provenance needs a tag):

   ```bash
   cd scienceboard_setup/s4
   git clone <scienceboard-repo> ScienceBoard_Base
   git clone <scienceboard-repo> ScienceBoard_Evol
   git -C ScienceBoard_Evol checkout -b evol && git -C ScienceBoard_Evol tag <tag>
   uv pip install --python ../../osworld_setup/s4/.venv/bin/python \
       -r ScienceBoard_Base/requirements.txt
   ```

   The Builder's clones never receive the official task tree (`tasks/`) or
   its git history; the eval reads tasks from `ScienceBoard_Base` only.
3. `cp .env.example .env` and fill it in.

### What upstream ScienceBoard does not provide

Tasks run on one fresh EC2 instance each, launched from `SB_AMI_ID`. Two
requirements go beyond upstream
[OS-Copilot/ScienceBoard](https://github.com/OS-Copilot/ScienceBoard), whose
`VManager` supports only the VMware and Docker providers:

- the checkout's `sci` package must accept `provider_name="aws"` (with
  `region` / `ami_id`) in `VManager` and wait for the guest daemon on revert;
- `SB_AMI_ID` must be an AMI of the ScienceBoard VM image (the six scientific
  apps installed, guest daemon on port 5000, user password
  `osworld-public-evaluation`, overridable with `SB_CLIENT_PASSWORD`).

Without both, the launchers' offline checks pass but no task can run.

## Running

48 of the 50 synthetic training tasks (ungraded) and 15 official validation
tasks as the fitness signal. The synthetic tasks are on
[Hugging Face](https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks); download them first (see
[data/README.md](data/README.md)):

```bash
curl -L -o data/scienceboard_synthetic_50.tar.xz \
    https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks/resolve/main/scienceboard_synthetic.tar.xz
bash scripts/eval/run_scienceboard_qwen_synthetic_50_15.sh --dry-run   # offline checks
bash scripts/eval/run_scienceboard_qwen_synthetic_50_15.sh --yes
```

`SHADOW_EVOL_EVAL_TIMEOUT` sets the per-task wall clock (600 s default); the
matching Builder prompt (`prompts/scienceboard_synthetic_trajectories*.md`) is
selected from it. `--help` lists the other settings.

Evaluate one variant on all 169 official tasks:

```bash
bash scripts/eval/run_scienceboard_full_variant.sh --namespace <ns> --variant if-014 --yes
```

## What a run produces

| path | contents |
|------|----------|
| `shadow_evolution_scienceboard_<ns>.db` | variants, scores, per-task results, costs |
| `ScienceBoard_Evol_<ns>/` | island clone; each variant is a git tag |
| `eval_cache_scienceboard_<ns>/<variant>/` | per-task trajectories, results, summaries |
| `builder_logs_scienceboard_<ns>/` | Builder session logs |

Trajectories record the VM environment, which can include API keys. Redact
them before sharing.

## Tests

The ScienceBoard tests run with the rest of the S4 suite:
`cd osworld_setup/s4 && pytest -q`.
