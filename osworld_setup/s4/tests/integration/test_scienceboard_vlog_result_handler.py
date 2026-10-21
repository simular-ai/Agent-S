"""Contract test for the ScienceBoard eval result-logging fix.

ScienceBoard's `Task.eval` is decorated with `@Log.result_handler`, which writes
the verdict to `vlog.result_file_path` — a property that asserts an *open* log
`file_handler`. The task's `vlog` is a `VirtualLog` proxy whose `__getattr__`
returns a brand-new THROWAWAY `Log(disabled=True)` whenever its backing `_log`
is None. In ScienceBoard's native runner (`Tester`) the backing log is set and
opened (`new_task.vlog.set(self.log)` + `self.log.trigger(...)`); our driver
runs its own agent with a stub Community, so `_log` stays None.

`agent_s4/eval/scienceboard_prime.py::ScienceBoardEnv.score_official` therefore mirrors `Tester`: it
`set()`s a persistent backing `Log` on the task's vlog, then `trigger()`s it,
before calling `eval()`. This test pins both halves of why that's necessary:

  - a bare `trigger()` on an empty proxy does NOT persist (the trap an earlier
    fix attempt fell into), so `result_file_path` still asserts;
  - `set()` + `trigger()` opens the handler and `result_file_path` resolves into
    the given dir as `result.out` — NOT `result.txt`, so it can't clobber the
    `result.txt` our driver writes into the same per-task dir.

Skips when the vendored ScienceBoard checkout isn't present.
"""
import os
import sys
from pathlib import Path

import pytest

_SB = (
    Path(__file__).resolve().parents[4]
    / "scienceboard_setup"
    / "s4"
    / "ScienceBoard_Base"
)
if _SB.is_dir() and str(_SB) not in sys.path:
    sys.path.insert(0, str(_SB))

log_mod = pytest.importorskip(
    "sci.base.log", reason="ScienceBoard_Base checkout not present"
)


def test_bare_trigger_on_empty_proxy_still_asserts(tmp_path):
    """A VirtualLog with no backing log (like task_obj.vlog under a stub
    Community) makes a throwaway Log per access, so a bare trigger() is
    discarded and result_file_path still asserts — the exact failure mode the
    first fix attempt hit."""
    v = log_mod.VirtualLog()
    assert v._log is None
    v.trigger(str(tmp_path))  # opens a handler on an instantly-discarded Log
    with pytest.raises(AssertionError):
        _ = v.result_file_path


def test_set_then_trigger_opens_handler(tmp_path):
    """The fix run_one applies, mirroring ScienceBoard's Tester
    (new_task.vlog.set(self.log) + self.log.trigger(...)): a persistent backing
    Log + trigger opens the handler; result_file_path resolves into our dir."""
    v = log_mod.VirtualLog()
    v.set(log_mod.Log(disabled=True, global_vlog=False))
    v.trigger(str(tmp_path))

    assert v._log.file_handler is not None
    assert os.path.dirname(v.result_file_path) == str(tmp_path)
    # Must NOT be result.txt — that's our driver's file; a collision would let
    # ScienceBoard's "1"/"0" overwrite our "0.0"/"1.0".
    assert os.path.basename(v.result_file_path) == "result.out"
    assert os.path.basename(v.result_file_path) != "result.txt"

    # And the decorator's write actually succeeds now.
    with open(v.result_file_path, "w") as f:
        f.write(str(int(True)))
    assert open(v.result_file_path).read() == "1"
