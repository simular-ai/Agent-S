# Synthetic ScienceBoard training tasks

The ScienceBoard evolution launcher trains the Builder on ungraded synthetic
ScienceBoard tasks (50 in the bank, 48 in `scienceboard_synthetic_train_48`).
They are not stored in git; download the archive from Hugging Face:

[simular-ai/evolving_agent_interfaces_synthetic_tasks](https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks)

| File | Size | SHA-256 |
|---|---|---|
| `scienceboard_synthetic.tar.xz` | 0.12 MB (0.4 MB unpacked) | `fe1dca1ecaa67e6e97e0c10cb585888fb910b114aadd37f43327040d8663d5cb` |

From `scienceboard_setup/s4/`, download it under the name the launcher looks for:

```bash
curl -L -o data/scienceboard_synthetic_50.tar.xz \
    https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks/resolve/main/scienceboard_synthetic.tar.xz
sha256sum data/scienceboard_synthetic_50.tar.xz   # must match the table above
```

```text
scienceboard_setup/s4/data/
  scienceboard_synthetic_50.tar.xz
```

`scripts/eval/run_scienceboard_qwen_synthetic_50_15.sh` unpacks it to
`data/scienceboard_synthetic_50/` on first use (or set
`SCIENCEBOARD_SYNTHETIC_ROOT` to an unpacked copy elsewhere).

The tasks are not scored: `eval()` deliberately raises, so they must not be
sent to the ordinary ScienceBoard runner. Fitness comes from the official
validation split.
