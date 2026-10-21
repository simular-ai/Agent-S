# agent_s4

The Agent S4 package, from the paper
[Evolving Agent Interfaces](https://arxiv.org/abs/2610.28601). It evolves a
**shadow interface**: a small Python tool library that a computer-use agent
imports to operate a desktop. Each step, a **Builder** agent edits an
interface on the host. A **task agent** then runs benchmark tasks on fresh AWS
VMs with that interface installed, and its score becomes the variant's
fitness.

Computer-use agents are limited by the interfaces humans give them: an agent
may type a spreadsheet cell by cell when one line of code would do. Agent S4
builds those interfaces itself. The Builder studies the task agent's
trajectories on synthetic training tasks, without success labels, and writes
and revises the interfaces and the knowledge of when to use them.
Population-based evolution then keeps the variants that score best on
validation tasks the Builder never sees, which favors changes that generalize.

## Results

On OSWorld V2, Agent S4 with Qwen3.8-27B and its evolved interfaces reaches a
**21.3%** binary score, a new state of the art for open-source computer-use
agents and 3.8 times the same model with fixed interfaces. That is above
GPT-5.5 (13.0%) and Claude Opus 4.8 (20.6%), at about 4.3% and 1.5% of their
respective costs. On ScienceBoard, the evolved interfaces raise the success
rate from 41.4% to 58.6% with the same model.

## Setup guides

This file describes the code. For setup, credentials and the launch scripts
behind the published runs, see the workspace READMEs in the
[Agent-S repository](https://github.com/simular-ai/Agent-S):

- [`osworld_setup/s4/README.md`](https://github.com/simular-ai/Agent-S/blob/main/osworld_setup/s4/README.md):
  OSWorld-V2, which is the default benchmark.
- [`scienceboard_setup/s4/README.md`](https://github.com/simular-ai/Agent-S/blob/main/scienceboard_setup/s4/README.md):
  ScienceBoard.

## Install

`agent-s4` is its own package; installing `gui-agents` does not include it.

The package contains only the code. A run also reads setup files from the
Agent-S repository: the Builder prompts, the seed interface, the task splits
and the pinned Prime Agent runtime, all under `osworld_setup/s4/`. So you
need a clone of the repository either way. Install the package from the
clone:

```bash
git clone https://github.com/simular-ai/Agent-S.git
cd Agent-S
pip install -e ./agent_s4
```

or install it from PyPI and point it at the clone:

```bash
pip install agent-s4
export SHADOW_EVOL_SHARED_ROOT=/path/to/Agent-S/osworld_setup/s4
```

Without those files, `python -m agent_s4.cli run` stops and says what to set.
The benchmark itself (the OSWorld-V2 checkout, AWS and model credentials) is
set up as described in the workspace READMEs above.

## Entry points

| Command | What it does |
|---|---|
| `python -m agent_s4.cli run ...` | Starts or resumes an experiment (`--help` lists the options). |
| `python -m agent_s4.cli rerun --config-path runs/<timestamp>/config.json` | Runs again from a saved config. |
| `python -m agent_s4.osworld_v2_preflight` | Runs the offline OSWorld-V2 workspace checks on their own. |
| `python -m agent_s4.eval.osworld_v2 ...`, `python -m agent_s4.eval.scienceboard_prime ...` | Eval workers. The evaluator harness normally starts these; they can also be run by hand on chosen tasks. |

`run` defaults to `--benchmark osworld_v2`, which uses the splits pinned in
`config/osworld_v2_release_pin.json`: `osworld_v2_train_50.json`,
`osworld_v2_val_25.json` and `osworld_v2_test.json` as the held-out set. No
train or val splits ship for `--benchmark osworld` (the original OSWorld) or
`--benchmark scienceboard`, so both require `--train-set` and `--val-set`.

## How a run works

**1. Bootstrap** (`orchestrator.run`):
- If Prime Agent is used, checks its pinned runtime. For OSWorld-V2 it also
  runs the workspace preflight and checks the configured splits against the
  release pin.
- Makes one clone of the benchmark's Evol checkout per island, named
  `<template>_<island>` (for example `OSWorldV2_Evol_<island>`).
- Copies the seed interface into the clone's `shadow_interface/` directory,
  which has its own git repository, and records it as the seed variant.
- Evaluates the seed on every island in parallel, writes the resolved config
  to `runs/<timestamp>/config.json`, and hands over to the driver.

**2. Driver loop** (`driver.Driver`):
- Keeps one global step counter.
- Each step, the scheduler (`config/scheduler.py`) picks one operator, which
  runs on every island in parallel. The step ends only when all islands have
  finished.
- Every dispatch is logged in the database, so a step interrupted by a crash
  is resumed rather than run twice.
- Validation reevaluation runs every `--reevaluation-interval` steps.
- The loop stops at `--max-steps` or `--max-cost`.

**3. Operators** (`operators/`): each step runs one of these.

| Operator | What it does |
|---|---|
| `mutation` | The usual step; see the sequence below. |
| `crossover` | The Builder combines two parents from the same island into one child, which is then evaluated like a mutation. |
| `migration` | Two phases: each island sends a tournament pick to another island, and the receiving island evaluates the migrant with its own task agent before adding it. `--migration-mode` and `--migration-topology` control the details. |
| `elimination` | Removes the lowest-ranked variants from each island. |
| `reevaluation` | Evaluates each island's current leaders on validation again and averages the results with their earlier scores. It creates no new variants. |

A mutation runs these steps:
1. A tournament picks the parent (`tournament.py`).
2. The parent is checked out, and its task trajectories are staged for the
   Builder to read.
3. Optionally, a scratch VM is started (`scratch_vm.py`) so the Builder can
   try things in a live app.
4. The Builder harness edits `shadow_interface/`. For OSWorld-V2 and
   ScienceBoard it runs inside a Landlock sandbox, because the Base checkout
   next to the clone holds the benchmark's gated evaluators.
5. Edits outside the allowed scope are reverted, and the child is tagged
   `evol/<island>/if-NNN`.
6. The evaluator harness runs the child on the train and val splits. The
   per-task results, scores and costs go into the database.

## Module map

| Area | Modules |
|---|---|
| Configuration | `cli.py` (argument parsing), `config/experiment.py` (`ExperimentConfig`, default splits per benchmark), `config/island.py`, `config/scheduler.py` |
| Control loop | `orchestrator.py` (bootstrap and seed eval), `driver.py`, `state.py` (`LoopState`, passed to each operator), `operators/`, `tournament.py`, `scoring.py` |
| State | `db.py` (SQLite tables: variants, per-task eval results, islands, history, cost events, dispatch log, reevaluation samples), `costs.py`, `git_utils.py` (variant tags and checkouts in each island's `shadow_interface/` repo) |
| Builder side | `harness/builder_common.py` (shared Builder subprocess and log parsing), `builder_sandbox.py` (Landlock), `builder_scratch_vm.py`, `scratch_vm.py` |
| Agent harnesses | `harness/prime/`, `harness/opencode/`: each has the Builder (`builder.py`), the task-agent install and VM command (`evaluator.py`), a runner, and pin checks (`source.py`); see the README in each directory. `harness/__init__.py` is the registry. |
| Evaluation | `harness/ubuntu_vm.py` (`UbuntuVMEvaluator`: starts an eval worker as a subprocess), `eval/osworld.py` (worker code shared by every benchmark), `eval/osworld_v2.py`, `eval/scienceboard_prime.py`, `eval/grader_image_limit.py` |
| Benchmarks | `benchmark_spec.py` (template and base checkout names, how run artefacts are named), `osworld_v2_preflight.py`, `synthetic_tasks.py`, `scienceboard_synthetic_runtime.py` |
| VMs | `vm/utils.py` (shared VM helpers), `vm/owner_tag.py` (adds an `Owner` tag to every EC2 instance) |
| Paths | `paths.py`, `constants.py`, `utils.py` |

## Where files go

`paths.py` sets two roots:

- **Shared root** (`SHADOW_EVOL_SHARED_ROOT`; defaults to `osworld_setup/s4/`):
  the vendored Prime runtime, `vm_assets/`, the seed interfaces and the generic
  Builder prompts.
- **Project root** (`SHADOW_EVOL_PROJECT_ROOT`; defaults to the shared root):
  the run workspace. It holds the benchmark's prompts, splits, data, release
  pin, checkouts and every run artefact. The ScienceBoard launchers set it to
  `scienceboard_setup/s4/`.

Run artefacts are named per benchmark (and per `SHADOW_EVOL_RUN_NAMESPACE`) by
`benchmark_spec.ns()`. For example, OSWorld-V2 uses
`shadow_evolution_osworld_v2_<ns>.db`, `eval_cache_osworld_v2_<ns>/` and
`builder_logs_osworld_v2_<ns>/`.

## Extending

**Add an agent harness:**
- Subclass `BuilderHarness` (`harness/base.py`) in `harness/<name>/builder.py`.
- Put the VM install and task-agent command in `harness/<name>/evaluator.py`.
- Register the key in both `BUILDER_HARNESSES` and `EVALUATOR_HARNESSES`.
- Teach `UbuntuVMEvaluator` and `eval/osworld.py` to dispatch on the key. Both
  currently accept only `prime_agent` and `opencode`.
- Pin and verify the upstream release in a `source.py`, and record it in a
  README, as the existing two harnesses do.

**Add a benchmark:**
- Add an entry to `benchmark_spec.BENCHMARKS`, and to `_DEFAULT_SPLITS` in
  `config/experiment.py` if it ships default splits.
- Write an eval worker that plugs into `eval.osworld.main(impl=...)`, as
  `eval/scienceboard_prime.py` does.
- The benchmark-specific branches in `harness/ubuntu_vm.py` and
  `orchestrator.py` (search for `config.benchmark ==`) cover how the base
  checkout is chosen, which prompt addenda are added and how splits are
  checked.
