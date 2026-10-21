## ScienceBoard eval result format

Each ScienceBoard task writes one extra file beyond the generic per-task
artefacts:

- `eval_results/shadow_eval/screenshot/{model}/{task_id}/scienceboard_evaluation.json`
  — eval-side diagnostic. **Read this before opening the trajectory.** Fields:
    - `score` — 0.0 or 1.0 (no partial credit for ScienceBoard).
    - `eval_items_in_config` — what the benchmark was checking, e.g.
      `[{"type":"states","key":"_state_containers...","value":"prot.cif"}]`.
    - `eval_exception` — full Python traceback if `task.eval()` itself raised.
      **The single most informative field.** Common cause: `AssertionError`
      from `manager.states_dump()` means ScienceBoard couldn't reach the
      app's REST endpoint on port 8000 — almost always because the agent
      launched the app without REST control enabled (e.g. plain `chimerax`
      instead of `chimerax --cmd "remotecontrol rest start json true port
      8000"`). The trajectory may show "task complete" but the benchmark
      cannot introspect.
    - `post_eval_state_error` — when `task.eval()` returned False (not
      raised), we re-probe `states_dump()`; this captures the error if that
      probe also fails. Same root cause as `eval_exception` in most cases.
    - `post_eval_state_keys` — first 80 sorted keys of the post-eval state
      dict (when reachable). Use to sanity-check the app left the right
      shape behind.
    - `stop_used` / `stop_args` — terminal signal we passed to the
      evaluator (`TIMEOUT` / `ANS` / `FAIL`).

When triaging a 0: open `scienceboard_evaluation.json` first. If
`eval_exception` is non-null the fix is in the interface's app-launch path
(REST mode); if it's null but `score=0.0`, the agent's claim disagrees
with `post_eval_state_keys` — dive into the trajectory.
