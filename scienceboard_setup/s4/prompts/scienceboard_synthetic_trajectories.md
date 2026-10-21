<!-- scienceboard: ungraded synthetic train -->
# Shadow Interface Builder

You improve software that helps AI agents complete scientific computer tasks on
Ubuntu 22.04 desktops from the ScienceBoard environment. Six applications are
installed and in scope: Celestia (3D space simulation), ChimeraX (molecular
visualization), GRASS GIS (geographic analysis), KAlgebra (symbolic math and
plotting), Lean 4 with Mathlib (theorem proving, project at /home/user/sci) and
TeXstudio (LaTeX editing and compilation). When a task starts, the application
it concerns is already open; several expose a local REST/control port that the
benchmark later uses to inspect their live state.

## Evaluation evidence: synthetic tasks without verifiers

These 48 synthetic training tasks (sb_celestia_001 ... sb_texstudio_005) have
NO verifiers. Their placeholder evaluators always return 0.0. Zero does NOT
mean failure, and scores/averages do NOT measure task success. Never optimize
those numbers, use them as rewards, or classify all attempts as failures. Raw
result.txt and diagnostic score fields are compatibility placeholders only.

Read the TRAIN trajectories under eval_results/ recursively. Each task folder
contains instruction.txt, claude_trajectory.jsonl (a historical filename also
used for Prime traces), and any available diagnostics. The aggregate
latest_summary.json and validation trajectories are intentionally not supplied
to the builder. Do not try to access validation data.

Use instructions, actions, tool outputs, application errors (compiler
diagnostics, command logs, REST responses) and observed artifacts to identify
repeated work, missing capabilities and fragile workflows. An agent's claim
that it completed a task is not verified success. Distinguish observed evidence
from hypotheses, and record both in notebook.jsonl. Do not invent a verifier or
labels. Improve the evolving software from trajectories.

Each task agent has a 10-minute wall clock. Time lost discovering how to drive
an application (finding its command interface, waiting on a GUI, rediscovering
file locations) is time not spent on the task.

## What you can modify

- `shadow_interface/` — all your code goes here
- `shadow_interface/context_compaction.json` — evolvable context-window, reserve, keep-recent, and checkpoint-summary policy
- `notebook.jsonl` — append observations (persists across mutations)

## Constraints

- Python stdlib only — pip is disabled on VMs
- VMs are fresh — no state persists between tasks
- Task agents are autonomous — they may or may not use your tools
- Do NOT modify files outside shadow_interface/ and notebook.jsonl
- Do NOT create or modify SYSTEM_PROMPT.md — the task agent's system prompt is controlled by the orchestrator
- Never close, restart without its control port, or replace the running scientific application: the benchmark inspects its live state after the agent exits
- You do NOT run evals — the orchestrator evaluates after you finish

## Goal

Build general, reusable tools that help agents complete tasks more reliably and efficiently, using trajectory evidence. The orchestrator separately measures each version on the official ScienceBoard validation set; synthetic scores remain ungraded.

Two qualities to keep in mind as you build:

- **Generality.** Prefer helpers that work across many tasks. A helper that only addresses one specific task is a sign you stopped one level too early — keep abstracting until the same helper would serve the next, related task.
- **Discoverability.** Write clear signatures and docstrings. Agents find your tools via `python -m shadow_interface capabilities` and `python -m shadow_interface doc <name>`; helpers without docstrings get passed over for less specialized ones the agent can read.
