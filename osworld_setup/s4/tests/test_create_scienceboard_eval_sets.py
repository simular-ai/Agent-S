import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "scienceboard_setup" / "s4"
SCRIPT = PROJECT_ROOT / "scripts" / "eval" / "create_scienceboard_eval_sets.py"


def test_split_sizes_and_id_format(tmp_path):
    # fixture: 2 apps x 4 tasks each
    for app in ("ChimeraX", "Lean"):
        d = tmp_path / "tasks_VM" / app
        d.mkdir(parents=True)
        for n in ("A-01", "A-02", "B-01", "B-02"):
            (d / f"{n}.json").write_text(json.dumps({"type": app, "instruction": "x"}))
    out = tmp_path / "eval_sets"
    out.mkdir()
    subprocess.run([sys.executable, str(SCRIPT),
                    "--tasks-root", str(tmp_path / "tasks_VM"),
                    "--out-dir", str(out), "--train", "4", "--val", "2", "--seed", "42"],
                   check=True, cwd=str(PROJECT_ROOT))
    train = json.loads((out / "scienceboard_train_4.json").read_text())
    val = json.loads((out / "scienceboard_val_2.json").read_text())
    test = json.loads((out / "scienceboard_test.json").read_text())
    assert len(train) == 4
    assert len(val) == 2
    assert all(":" in tid for tid in train + val + test)  # App:stem ids
    # 3-way partition: splits are mutually disjoint and cover all 8 tasks.
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    assert set(train) | set(val) | set(test) == {
        f"{app}:{n}" for app in ("ChimeraX", "Lean")
        for n in ("A-01", "A-02", "B-01", "B-02")
    }
