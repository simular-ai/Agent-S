"""Integration test for UbuntuVMEvaluator.run_tasks — subprocess mocked."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from agent_s4.harness.base import EvalResults
from agent_s4.harness.ubuntu_vm import (
    EVAL_MODULE_BY_BENCHMARK,
    UbuntuVMEvaluator,
)
from agent_s4.harness import ubuntu_vm


def _subprocess_writes_summary(result_dir, payload, *, returncode=0,
                               stdout="", stderr=""):
    """Return a mock side effect that creates output during this run."""
    def run(*args, **kwargs):
        (result_dir / "latest_summary.json").write_text(json.dumps(payload))
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr,
        )
    return run


def _write_task_leaf(result_dir, model, task_id, score, failure=None):
    leaf = result_dir / "shadow_eval" / "screenshot" / model / task_id
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "result.txt").write_text(str(score))
    if failure is not None:
        (leaf / "failure.json").write_text(json.dumps(failure))
    return leaf


def test_run_tasks_happy_path(tmp_path):
    """Subprocess writes latest_summary.json; harness reads + returns EvalResults."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    summary = {
        "pass_rate": 0.5, "total": 2, "passed": 1, "failed": 1,
        "results": [
            {"task_id": "t1", "passed": 1},
            {"task_id": "t2", "passed": 0},
        ],
    }

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    shadow_dir = clone_dir / "shadow_interface"
    shadow_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(
            result_dir, summary, stdout="eval output\n",
        )
        harness = UbuntuVMEvaluator()
        out = harness.run_tasks(
            clone_dir=clone_dir,
            shadow_dir=shadow_dir,
            result_dir=result_dir,
            task_ids=["t1", "t2"],
            model="haiku",
            region="us-east-1",
            timeout=7200,
        )

    assert isinstance(out, EvalResults)
    assert out.pass_rate == 0.5
    assert out.total == 2
    assert out.passed == 1
    assert out.failed == 1
    assert out.timed_out is False

    # Verify argv shape
    argv = mock_run.call_args[0][0]
    # venv python binaries name themselves `python`, `python3`, `python3.N`
    assert any(argv[0].endswith(n) for n in ("python", "python3")) \
        or argv[0].rsplit("/", 1)[-1].startswith("python3.")
    assert "-m" in argv
    assert "agent_s4.eval.osworld" in argv
    assert "agent_s4.eval.osworld_v2" not in argv
    assert "--model" in argv and "haiku" in argv
    assert "--num_envs" in argv and "2" in argv
    assert "--task_ids" in argv
    assert "t1" in argv and "t2" in argv
    assert "--region" in argv and "us-east-1" in argv
    # --timeout is the per-task in-VM Claude cap. Regression: before the
    # wiring fix, this arg wasn't passed and eval/osworld.py fell back to
    # its argparse default (600), silently ignoring the configured value.
    assert "--timeout" in argv
    assert str(7200) in argv
    # --test_config_base_dir is the path the eval uses to look up task
    # JSON definitions. We point it at OSWorld_Base/evaluation_examples
    # rather than the clone's own evaluation_examples/, so the curated
    # train-only prune in OSWorld_Evol/ doesn't starve val/test eval of
    # task definitions.
    assert "--test_config_base_dir" in argv
    idx = argv.index("--test_config_base_dir")
    assert argv[idx + 1].endswith("OSWorld_Base/evaluation_examples")


def test_run_tasks_caps_vm_concurrency_from_env(tmp_path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    shadow_dir = clone_dir / "shadow_interface"
    shadow_dir.mkdir()
    summary = {
        "pass_rate": 0, "total": 2, "passed": 0, "failed": 2,
        "results": [
            {"task_id": "t1", "passed": 0},
            {"task_id": "t2", "passed": 0},
        ],
    }
    monkeypatch.setenv("SHADOW_EVOL_EVAL_NUM_ENVS", "1")

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir, shadow_dir=shadow_dir, result_dir=result_dir,
            task_ids=["t1", "t2"], model="haiku", region="us-east-1",
            timeout=600,
        )

    argv = mock_run.call_args[0][0]
    assert argv[argv.index("--num_envs") + 1] == "1"


def test_run_tasks_test_config_base_dir_resolves_from_clone_parent(tmp_path):
    """The flag's value should be <clone_dir>.parent/OSWorld_Base/evaluation_examples
    — i.e. peer of the clone, derived from project root."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    summary = {
        "pass_rate": 0, "total": 1, "passed": 0, "failed": 1,
        "results": [{"task_id": "t1", "passed": 0}],
    }
    clone_dir = tmp_path / "OSWorld_Evol_t"
    clone_dir.mkdir()
    shadow_dir = clone_dir / "shadow_interface"
    shadow_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir, shadow_dir=shadow_dir, result_dir=result_dir,
            task_ids=["t1"], model="haiku", region="us-east-1", timeout=600,
        )

    argv = mock_run.call_args[0][0]
    idx = argv.index("--test_config_base_dir")
    expected = str(tmp_path / "OSWorld_Base" / "evaluation_examples")
    assert argv[idx + 1] == expected


@pytest.mark.parametrize(
    "n_tasks,expected_outer_timeout",
    [
        # outer = timeout*2 + 400 * max(1, n // 25). per-task timeout=600.
        (1,   1200 + 400 * 1),   # 1600s — minimum (max(1, 0) clamp)
        (25,  1200 + 400 * 1),   # 1600s
        (100, 1200 + 400 * 4),   # 2800s — 47 min (was 40 min under 300/chunk)
        (150, 1200 + 400 * 6),   # 3600s — 60 min for the train+val combined call
    ],
)
def test_outer_timeout_scales_with_task_count(tmp_path, n_tasks, expected_outer_timeout):
    """Outer subprocess timeout grows with task count. Bumped per-chunk from
    300s → 400s in support of 150-task train+val combined eval calls (more
    concurrent VMs → more AWS API throttling + per-task setup variance)."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    task_ids = [f"t{i}" for i in range(n_tasks)]
    summary = {
        "pass_rate": 0, "total": n_tasks, "passed": 0, "failed": n_tasks,
        "results": [{"task_id": tid, "passed": 0} for tid in task_ids],
    }
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=task_ids,
            model="haiku",
            region="us-east-1",
            timeout=600,
        )

    assert mock_run.call_args.kwargs["timeout"] == expected_outer_timeout


def test_run_tasks_handles_timeout(tmp_path):
    """Subprocess TimeoutExpired → EvalResults with timed_out=True."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=1)
        harness = UbuntuVMEvaluator()
        out = harness.run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=["t1"],
            model="haiku",
            region="us-east-1",
            timeout=1,
        )

    assert out.timed_out is True
    assert out.pass_rate == 0
    assert out.total == 1
    assert out.passed == 0
    assert out.failed == 1


def test_v2_timeout_runs_tag_sweep_and_records_cleanup(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess",
               side_effect=subprocess.TimeoutExpired(cmd=[], timeout=1)), \
         patch("agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
               return_value=["i-leaked"]) as cleanup:
        out = UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir, task_ids=["t1"], model="haiku",
            region="us-east-1", timeout=1, benchmark="osworld_v2",
        )
    assert out.timed_out is True
    cleanup.assert_called_once()
    records = list(result_dir.glob("vm_cleanup.shadow-evol-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["cleanup_status"] == "complete"
    assert record["terminated_instance_ids"] == ["i-leaked"]


def test_subprocess_timeout_terminates_process_group(monkeypatch):
    class FakeProc:
        pid = 4321
        returncode = -15

        def __init__(self):
            self.waits = 0

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired(cmd=["eval"], timeout=timeout)

        def wait(self, timeout=None):
            self.waits += 1
            return self.returncode

    fake = FakeProc()
    signals = []
    popen_kwargs = {}

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return fake

    monkeypatch.setattr(ubuntu_vm.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        ubuntu_vm.os, "killpg", lambda pid, sig: signals.append((pid, sig)),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        ubuntu_vm._run_eval_subprocess(
            ["eval"], cwd=".", capture_output=True, text=True, timeout=1,
        )
    assert signals == [(4321, ubuntu_vm.signal.SIGTERM)]
    assert fake.waits == 1
    assert popen_kwargs["start_new_session"] is True


def test_subprocess_timeout_escalates_to_sigkill(monkeypatch):
    class StubbornProc:
        pid = 9876
        returncode = -9

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired(cmd=["eval"], timeout=timeout)

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd=["eval"], timeout=timeout)
            return self.returncode

    signals = []
    monkeypatch.setattr(
        ubuntu_vm.subprocess, "Popen", lambda *args, **kwargs: StubbornProc(),
    )
    monkeypatch.setattr(
        ubuntu_vm.os, "killpg", lambda pid, sig: signals.append((pid, sig)),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        ubuntu_vm._run_eval_subprocess(
            ["eval"], cwd=".", capture_output=True, text=True, timeout=1,
        )
    assert signals == [
        (9876, ubuntu_vm.signal.SIGTERM),
        (9876, ubuntu_vm.signal.SIGKILL),
    ]


def test_run_tasks_missing_summary_returns_zeroed_results(tmp_path):
    """Missing summary is infrastructure failure, never zero fitness."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="infra failure",
        )
        harness = UbuntuVMEvaluator()
        with pytest.raises(RuntimeError, match="without latest_summary"):
            harness.run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir,
                task_ids=["t1", "t2", "t3"],
                model="haiku",
                region="us-east-1",
                timeout=7200,
            )


def test_run_tasks_reuses_complete_summary_without_subprocess(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "latest_summary.json").write_text(json.dumps({
        "pass_rate": 1, "total": 1, "passed": 1, "failed": 0,
        "results": [{"task_id": "t1", "passed": 1}],
    }))
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        out = UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=["t1"], model="haiku", region="us-east-1",
            timeout=600,
        )

    mock_run.assert_not_called()
    assert out.total == 1
    assert out.passed == 1


def test_run_tasks_recovers_complete_per_task_cache_without_subprocess(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    model = "openrouter/qwen/test-model"
    _write_task_leaf(result_dir, model, "t1", 1.0)
    _write_task_leaf(result_dir, model, "t2", 0.0, {
        "task_id": "t2", "score": 0.0, "valid_for_fitness": True,
        "imputed_zero": True, "attempt": 3, "max_attempts": 3,
        "failure_type": "agent_execution_failure", "error": "daemon closed",
    })

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        out = UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=["t1", "t2"], model=model, region="us-east-1",
            timeout=600,
        )

    mock_run.assert_not_called()
    assert out.total == 2
    assert out.passed == 1
    assert out.failed == 1
    summary = json.loads((result_dir / "latest_summary.json").read_text())
    assert summary["recovered_from_task_results"] is True
    assert summary["results"][1]["imputed_zero"] is True


def test_v2_recovery_normalizes_float_dust_but_keeps_partial_nonbinary(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    model = "openrouter/qwen/test-model"
    _write_task_leaf(result_dir, model, "dust", 0.9999999999999999)
    _write_task_leaf(result_dir, model, "partial", 0.9979)

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run, \
         patch("agent_s4.harness.ubuntu_vm._recover_v2_cleanup_records"):
        out = UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=["dust", "partial"], model=model, region="us-east-1",
            timeout=600, benchmark="osworld_v2",
        )

    mock_run.assert_not_called()
    assert out.passed == 1
    assert out.failed == 1
    summary = json.loads((result_dir / "latest_summary.json").read_text())
    assert summary["results"][0]["score"] == 1.0
    assert summary["results"][1]["score"] == 0.9979


def test_run_tasks_does_not_recover_unresolved_per_task_failure(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    model = "haiku"
    _write_task_leaf(result_dir, model, "t1", 0.0, {
        "task_id": "t1", "score": 0.0, "valid_for_fitness": False,
        "failure_type": "vm_allocation_failure",
    })

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="new run crashed",
        )
        with pytest.raises(RuntimeError, match="without latest_summary"):
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir,
                task_ids=["t1"], model="haiku", region="us-east-1",
                timeout=600,
            )

    mock_run.assert_called_once()


def test_run_tasks_deletes_invalid_stale_summary_before_subprocess(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "latest_summary.json").write_text(json.dumps({
        "pass_rate": 1, "total": 1, "passed": 1, "failed": 0,
        "results": [{"task_id": "old-task", "passed": 1}],
    }))
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="new run crashed",
        )
        with pytest.raises(RuntimeError, match="without latest_summary"):
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir,
                task_ids=["t1"], model="haiku", region="us-east-1",
                timeout=600,
            )

    mock_run.assert_called_once()
    assert not (result_dir / "latest_summary.json").exists()


def test_run_tasks_rejects_same_count_from_different_tasks(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    summary = {
        "pass_rate": 0.5, "total": 2, "passed": 1, "failed": 1,
        "results": [
            {"task_id": "old-a", "passed": 1},
            {"task_id": "old-b", "passed": 0},
        ],
    }
    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        with pytest.raises(RuntimeError, match="task IDs do not match"):
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir,
                task_ids=["new-a", "new-b"], model="haiku",
                region="us-east-1", timeout=600,
            )


@pytest.mark.parametrize("summary,match", [
    ({
        "pass_rate": 0, "total": 1, "passed": 0, "failed": 0,
        "infrastructure_failed": 1,
        "results": [{"task_id": "t1", "valid_for_fitness": False,
                     "failure_type": "vm_allocation_failure"}],
    }, "infrastructure/config"),
    ({
        "pass_rate": 0, "total": 0, "passed": 0, "failed": 0,
        "results": [],
    }, "requested task"),
])
def test_run_tasks_rejects_infrastructure_and_partial_summaries(
        tmp_path, summary, match):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        with pytest.raises(RuntimeError, match=match):
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir, task_ids=["t1"], model="haiku",
                region="us-east-1", timeout=600,
            )


def test_run_tasks_rejects_nonzero_subprocess_with_complete_summary(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    summary = {
        "pass_rate": 1.0, "total": 1, "passed": 1, "failed": 0,
        "results": [{"task_id": "t1", "score": 1.0}],
    }
    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run:
        mock_run.side_effect = _subprocess_writes_summary(
            result_dir, summary, returncode=2, stderr="post-summary failure",
        )
        with pytest.raises(RuntimeError, match="despite writing a summary"):
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir, task_ids=["t1"], model="haiku",
                region="us-east-1", timeout=600,
            )


def test_summary_aggregate_validation_rejects_contradiction():
    with pytest.raises(RuntimeError, match="aggregates contradict"):
        ubuntu_vm._validate_summary_aggregates({
            "pass_rate": 1.0, "total": 1, "passed": 1, "failed": 0,
            "results": [{"task_id": "t1", "score": 0.0}],
        }, ["t1"])


@pytest.mark.parametrize("benchmark,expected_base", [
    ("osworld", "OSWorld_Base/evaluation_examples"),
    ("osworld_v2", "OSWorldV2_Base/evaluation_examples"),
])
def test_run_tasks_task_tree_follows_benchmark(tmp_path, benchmark, expected_base):
    """The *_Base task tree is selected by benchmark, not hardcoded. Passing
    osworld_v2 must point the eval at OSWorldV2_Base; the clone dir itself is
    irrelevant to that choice."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    summary = {
        "pass_rate": 0, "total": 1, "passed": 0, "failed": 1,
        "results": [{"task_id": "t1", "score": 0.0}],
    }
    clone_dir = tmp_path / f"clone_{benchmark}"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run, \
         patch("agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
               return_value=[]) as mock_cleanup:
        mock_run.side_effect = _subprocess_writes_summary(result_dir, summary)
        UbuntuVMEvaluator().run_tasks(
            clone_dir=clone_dir,
            shadow_dir=clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=["t1"],
            model="haiku",
            region="us-east-1",
            timeout=600,
            benchmark=benchmark,
        )

    argv = mock_run.call_args[0][0]
    idx = argv.index("--test_config_base_dir")
    assert argv[idx + 1].endswith(expected_base)
    # Each generation has its own eval module; dispatching a V2 run to the
    # v1 one would silently drop V2's task semantics.
    assert EVAL_MODULE_BY_BENCHMARK[benchmark] in argv
    if benchmark == "osworld_v2":
        assert "--benchmark" in argv and "osworld_v2" in argv
        assert "--enable-proxy" in argv
        child_env = mock_run.call_args.kwargs["env"]
        assert child_env["AWS_INSTANCE_NAME"].startswith("shadow-evol-")
        mock_cleanup.assert_called_once()
    else:
        assert "--enable-proxy" not in argv
        mock_cleanup.assert_not_called()


def test_empty_summary_raises_with_stderr_instead_of_keyerror(tmp_path):
    """An empty latest_summary.json means every worker died before its first
    task (e.g. V2's AWS manager raising at import on a missing AWS_SUBNET_ID).
    That must surface the subprocess's stderr, not a KeyError on 'pass_rate',
    and must not be recorded as a legitimate zero score."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()

    with patch("agent_s4.harness.ubuntu_vm._run_eval_subprocess") as mock_run, \
         patch("agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
               return_value=[]):
        mock_run.side_effect = _subprocess_writes_summary(
            result_dir, {}, returncode=1, stdout="Summary written\n",
            stderr="EnvironmentError: AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID "
                   "must be set in the environment variables.",
        )
        with pytest.raises(RuntimeError) as exc:
            UbuntuVMEvaluator().run_tasks(
                clone_dir=clone_dir,
                shadow_dir=clone_dir / "shadow_interface",
                result_dir=result_dir,
                task_ids=["t1", "t2"],
                model="haiku",
                region="us-east-1",
                timeout=600,
                benchmark="osworld_v2",
            )
    msg = str(exc.value)
    assert "AWS_SUBNET_ID" in msg, "stderr must reach the operator"
    assert "empty summary" in msg


def test_cleanup_v2_instances_terminates_only_tagged_active_instances():
    class EC2:
        def __init__(self):
            self.filters = None
            self.terminated = []
            self.first_page_calls = 0

        def describe_instances(self, Filters, NextToken=None):
            self.filters = Filters
            if NextToken == "page-2":
                return {"Reservations": [
                    {"Instances": [{"InstanceId": "i-b"}]},
                ]}
            self.first_page_calls += 1
            if self.first_page_calls == 1:
                return {
                    "Reservations": [{"Instances": [{"InstanceId": "i-a"}]}],
                    "NextToken": "page-2",
                }
            return {"Reservations": []}

        def terminate_instances(self, InstanceIds):
            self.terminated.extend(InstanceIds)

    ec2 = EC2()
    with patch("boto3.client", return_value=ec2) as client, \
         patch("agent_s4.harness.ubuntu_vm.time.sleep"):
        got = ubuntu_vm._cleanup_v2_instances("us-east-1", "shadow-evol-run")
    assert got == ["i-a", "i-b"]
    assert ec2.terminated == got
    assert {"Name": "tag:Name", "Values": ["shadow-evol-run"]} in ec2.filters
    client.assert_called_once_with("ec2", region_name="us-east-1")


def test_cleanup_v2_instances_filters_by_configured_owner(monkeypatch):
    class EC2:
        def __init__(self):
            self.filters = []

        def describe_instances(self, Filters, NextToken=None):
            self.filters = Filters
            return {"Reservations": []}

    ec2 = EC2()
    monkeypatch.setenv("SHADOW_EVOL_OWNER_TAG", "example-owner")
    with patch("boto3.client", return_value=ec2), \
         patch("agent_s4.harness.ubuntu_vm.time.sleep"):
        assert ubuntu_vm._cleanup_v2_instances("us-east-1", "run", attempts=1) == []
    assert {"Name": "tag:Owner", "Values": ["example-owner"]} in ec2.filters


def test_cleanup_v2_instances_retries_eventual_visibility():
    class EC2:
        def __init__(self):
            self.calls = 0
            self.terminated = []

        def describe_instances(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                return {"Reservations": [
                    {"Instances": [{"InstanceId": "i-late"}]},
                ]}
            return {"Reservations": []}

        def terminate_instances(self, InstanceIds):
            self.terminated.extend(InstanceIds)

    ec2 = EC2()
    with patch("boto3.client", return_value=ec2), \
         patch("agent_s4.harness.ubuntu_vm.time.sleep"):
        got = ubuntu_vm._cleanup_v2_instances("us-east-1", "run")
    assert got == ["i-late"]
    assert ec2.terminated == ["i-late"]


def test_cleanup_v2_instances_can_stop_for_restricted_scheduler_role(
        monkeypatch):
    class Waiter:
        def __init__(self, name, ec2):
            self.name = name
            self.ec2 = ec2

        def wait(self, **kwargs):
            self.ec2.waited.append((self.name, kwargs))
            if self.name == "instance_stopped":
                self.ec2.stopped = True

    class EC2:
        def __init__(self):
            self.stopped = False
            self.stop_calls = []
            self.waited = []

        def describe_instances(self, Filters, NextToken=None):
            states = next(
                item["Values"] for item in Filters
                if item["Name"] == "instance-state-name"
            )
            state = "stopped" if self.stopped else "running"
            if state not in states:
                return {"Reservations": []}
            return {"Reservations": [{"Instances": [{"InstanceId": "i-a"}]}]}

        def stop_instances(self, InstanceIds):
            self.stop_calls.append(InstanceIds)

        def get_waiter(self, name):
            return Waiter(name, self)

    ec2 = EC2()
    monkeypatch.setenv("SHADOW_EVOL_V2_CLEANUP_ACTION", "stop")
    with patch("boto3.client", return_value=ec2):
        got = ubuntu_vm._cleanup_v2_instances("us-east-1", "run")
    assert got == ["i-a"]
    assert ec2.stop_calls == [["i-a"]]
    assert ec2.waited[0][0] == "instance_stopped"


def test_recover_v2_cleanup_records_preserves_and_sweeps_prior_run(tmp_path):
    record_path = tmp_path / "vm_cleanup.shadow-evol-old.json"
    record_path.write_text(json.dumps({
        "run_name": "shadow-evol-old", "region": "us-east-1",
        "cleanup_status": "pending",
    }))
    with patch(
        "agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
        return_value=["i-old"],
    ) as cleanup:
        ubuntu_vm._recover_v2_cleanup_records(tmp_path, "us-east-1")
    cleanup.assert_called_once_with("us-east-1", "shadow-evol-old")
    recovered = json.loads(record_path.read_text())
    assert recovered["cleanup_status"] == "complete"
    assert recovered["recovered_on_next_run"] is True
    assert recovered["terminated_instance_ids"] == ["i-old"]


def test_recover_v2_cleanup_record_uses_run_tag_from_truncated_filename(tmp_path):
    record_path = tmp_path / "vm_cleanup.shadow-evol-old.json"
    record_path.write_text('{"run_name":')
    with patch(
        "agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
        return_value=["i-recovered"],
    ) as cleanup:
        ubuntu_vm._recover_v2_cleanup_records(tmp_path, "us-east-1")
    cleanup.assert_called_once_with("us-east-1", "shadow-evol-old")
    recovered = json.loads(record_path.read_text())
    assert recovered["cleanup_status"] == "complete"
    assert recovered["recovered_on_next_run"] is True
    assert recovered["terminated_instance_ids"] == ["i-recovered"]
