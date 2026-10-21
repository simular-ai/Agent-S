# Synthetic OSWorld-V2 training tasks

The evolution launcher trains the Builder on 30 ungraded synthetic OSWorld-V2
tasks (`s_001` … `s_030`). They are not stored in git; download the archive
from Hugging Face:

[simular-ai/evolving_agent_interfaces_synthetic_tasks](https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks)

| File | Size | SHA-256 |
|---|---|---|
| `osworld_synthetic.tar.xz` | 18.6 MB (54 MB unpacked) | `095f9b9a594039a9a24e1556b0d13faf66eb0c636b75ef601487083ac1b8c9f4` |

From `osworld_setup/s4/`, download it under the name the launcher looks for:

```bash
curl -L -o data/osworld_synthetic_30.tar.xz \
    https://huggingface.co/datasets/simular-ai/evolving_agent_interfaces_synthetic_tasks/resolve/main/osworld_synthetic.tar.xz
sha256sum data/osworld_synthetic_30.tar.xz   # must match the table above
```

```text
osworld_setup/s4/data/
  osworld_synthetic_30.tar.xz
```

`scripts/eval/run_osworld_v2_qwen_synthetic_30_15.sh` unpacks it to
`data/osworld_synthetic_30/` on first use (or set `OSWORLD_SYNTHETIC_ROOT` to
an unpacked copy elsewhere). Nine tasks open the OSWorld-V2 mock websites, so
configure the package for your deployment once before the first run:

```bash
tar -xJf data/osworld_synthetic_30.tar.xz -C data
python3 scripts/eval/prepare_osworld_synthetic.py [--scheme http] [--origin APP=URL ...]
```

Every task has a dummy evaluator that always returns `0.0`: the Builder learns
from the trajectories, and fitness comes from the official validation split.
