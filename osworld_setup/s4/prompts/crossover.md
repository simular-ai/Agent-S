# Shadow Interface Builder — Crossover

You have two interface implementations that took different approaches.

## Parents

- **Parent A** (`crossover_parent_a/`): eval results in `eval_results/parent_a/`
- **Parent B** (`crossover_parent_b/`): eval results in `eval_results/parent_b/`

## Your Job

Combine the best of both into `shadow_interface/` (starts with Parent B's code).

1. Capabilities in A but not B — port them
2. Capabilities in B but not A — keep them
3. Different approaches to same thing — pick what works better per eval results
4. Ensure merged code is coherent

## Constraints

- Python stdlib only
- Modify only shadow_interface/ and notebook.jsonl
- You do NOT run evals
