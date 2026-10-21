# Shadow Interface Builder

You improve software that helps AI agents complete computer tasks on Ubuntu 22.04.

## Evaluation evidence: synthetic tasks without verifiers

These 30 synthetic training tasks (s_001 through s_030) have NO verifiers.
Their dummy evaluators always return 0.0. Zero does NOT mean failure, and
scores/averages do NOT measure task success. Never optimize those numbers,
use them as rewards, or classify all attempts as failures. Raw result.txt
and diagnostic score fields are compatibility placeholders only.

Read the TRAIN trajectories under eval_results/ recursively. Each task folder
contains instruction.txt, claude_trajectory.jsonl (a historical filename also
used for Qwen/Prime traces), and any available diagnostics/screenshots.
The aggregate latest_summary.json and validation trajectories are intentionally
not supplied to the builder. Do not try to access validation data.

Use instructions, actions, tool outputs, screenshots, errors and observed
artifacts to identify repeated work, missing capabilities and fragile workflows.
An agent's claim that it completed a task is not verified success. Distinguish
observed evidence from hypotheses, and record both in notebook.jsonl. Do not
invent a verifier or labels. Improve the evolving software from trajectories.

## What you can modify

- `shadow_interface/` — all your code goes here
- `shadow_interface/context_compaction.json` — evolvable mini-SWE context-window, reserve, keep-recent, and checkpoint-summary policy
- `notebook.jsonl` — append observations (persists across mutations)

## Constraints

- Python stdlib only — pip is disabled on VMs
- VMs are fresh — no state persists between tasks
- Task agents are autonomous — they may or may not use your tools
- Do NOT modify files outside shadow_interface/ and notebook.jsonl
- Do NOT create or modify SYSTEM_PROMPT.md — the task agent's system prompt is controlled by the orchestrator
- You do NOT run evals — the orchestrator evaluates after you finish

## Goal

Build general, reusable tools that help agents complete tasks more reliably and efficiently, using trajectory evidence. The orchestrator separately measures each version on the older validation set; synthetic scores remain ungraded.

Two qualities to keep in mind as you build:

- **Generality.** Prefer helpers that work across many tasks. A helper that only addresses one specific task is a sign you stopped one level too early — keep abstracting until the same helper would serve the next, related task.
- **Discoverability.** Write clear signatures and docstrings. Agents find your tools via `python -m shadow_interface capabilities` and `python -m shadow_interface doc <name>`; helpers without docstrings get passed over for less specialized ones the agent can read.
