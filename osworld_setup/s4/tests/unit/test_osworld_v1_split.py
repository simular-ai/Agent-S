"""OSWorld v1 behaviour that the osworld/osworld_v2 module split must keep.

agent_s4.eval.osworld holds both the v1 implementation and the pieces both
generations share, so it is the module most at risk of quietly acquiring V2
semantics. These tests pin the v1 side of each thing that was split.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_s4.eval import osworld, osworld_v2


def _args(tmp_path, **overrides):
    values = dict(
        screen_width=1920, screen_height=1080, provider_name="docker",
        region="us-east-1", snapshot_name="init_state", enable_proxy=None,
        volume_size=None, cache_dir=None, result_dir=str(tmp_path),
        path_to_vm=None, headless=True, client_password="pw",
        api_key=None, openrouter_api_key=None, parasail_api_key=None,
        timeout=1, model="fake", or_provider=None,
        no_tool=True, prompt_addendum="", action_space="shadow_eval",
        observation_type="screenshot", benchmark="osworld",
        test_config_base_dir=str(tmp_path),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _V2ShapedTask(dict):
    platform = "windows"
    image = "custom-image"
    instance_type = "t3.2xlarge"
    volume_size = 100
    proxy = True
    disable_vnc = True
    disable_recording = True


def test_v1_env_spec_ignores_task_declared_vm_overrides(tmp_path):
    """v1 tasks carry no VM configuration; a task that looks like a V2 one
    must not start reconfiguring v1's VM."""
    spec = osworld.build_env_spec(_V2ShapedTask(id="x"), _args(tmp_path))
    assert spec.os_type == "Ubuntu"
    assert spec.instance_type is None
    assert spec.volume_size is None
    assert spec.enable_proxy is False
    assert spec.force_disable_vnc is False
    assert spec.force_disable_recording is False
    assert spec.is_v2 is False


def test_v1_env_kwargs_omit_the_v2_only_desktop_env_arguments(tmp_path):
    """The v1 fork's DesktopEnv would reject these as unexpected kwargs."""
    spec = osworld.build_env_spec({"id": "x"}, _args(tmp_path))
    kwargs = osworld.build_env_kwargs(spec, _args(tmp_path))
    for v2_only in ("instance_type", "volume_size",
                    "force_disable_vnc", "force_disable_recording"):
        assert v2_only not in kwargs


def test_v1_task_index_and_loader_ignore_the_gated_task_class_corpus(tmp_path):
    base = tmp_path / "evaluation_examples"
    (base / "examples" / "chrome").mkdir(parents=True)
    (base / "examples" / "chrome" / "abc-123.json").write_text(
        json.dumps({"id": "abc-123", "instruction": "do it"}))
    (base / "task_class").mkdir()
    (base / "task_class" / "task_001.py").write_text("TASK_CLASS = None\n")

    index = osworld.build_task_index(str(base))
    assert set(index) == {"abc-123"}
    assert osworld.load_task(index["abc-123"], str(base))["instruction"] == "do it"

    # The V2 module is the one that sees both corpora.
    assert set(osworld_v2.build_task_index(str(base))) == {"abc-123", "001"}


def test_v1_instruction_gets_the_plain_context_suffix(tmp_path, monkeypatch):
    """No benchmark task date, and the 'do not ask' rule stays in place."""
    captured = []

    class Controller:
        def get_file(self, path):
            return b"0" if path.endswith("exit_code.txt") else b"{}\n"

        def start_recording(self):
            pass

        def end_recording(self, path):
            pass

    class Env:
        controller = Controller()
        client_password = "pw"
        action_history: list = []
        is_environment_used = False

        def reset(self, task_config):
            pass

        def evaluate(self):
            return 1.0

    def build_command(**kwargs):
        captured.append(kwargs["instruction"])
        return "agent"

    model_cfg = {
        "binary": "fake", "model_flag": "fake",
        "installer": lambda controller, password: True,
        "build_command": build_command,
    }
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)

    task = {"id": "x", "instruction": "do it", "task_current_date": "2026-03-01"}
    result = osworld.run_single_task(
        Env(), task, "do it", str(tmp_path), _args(tmp_path),
        model_cfg, str(tmp_path),
    )
    assert result["score"] == 1.0
    assert "Do not ask user any follow-up questions." in captured[0]
    assert "2026-03-01" not in captured[0]


def test_v1_episode_runs_once_even_when_the_agent_emits_ask_user(tmp_path, monkeypatch):
    """ASK_USER is a V2 protocol. Without the V2 hook the shared episode
    runner must finish after one episode rather than loop or raise."""
    ask = json.dumps({"text": "ASK_USER: which one?"}).encode() + b"\n"

    class Controller:
        def get_file(self, path):
            if path.endswith("trajectory.jsonl"):
                return ask
            if path.endswith("exit_code.txt"):
                return b"0"
            return None

    launches = []
    monkeypatch.setattr(
        osworld, "run_agent_on_vm",
        lambda controller, command, timeout: launches.append(command) or True,
    )
    osworld._run_agent_episode(
        SimpleNamespace(controller=Controller()), _args(tmp_path),
        {"binary": "fake", "model_flag": "fake",
         "build_command": lambda **kwargs: kwargs["instruction"]},
        "do it", "prompt", str(tmp_path),
    )
    assert launches == ["do it"]


@pytest.mark.parametrize("module,wrong_benchmark,right_module", [
    (osworld, "osworld_v2", "agent_s4.eval.osworld_v2"),
    (osworld_v2, "osworld", "agent_s4.eval.osworld"),
])
def test_each_module_refuses_the_other_generations_benchmark(
        module, wrong_benchmark, right_module, monkeypatch, capsys):
    """Dispatching a run to the wrong module would silently change the task
    semantics it is scored under, so it must fail loudly at parse time."""
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--task_ids", "x", "--benchmark", wrong_benchmark],
    )
    with pytest.raises(SystemExit):
        module.config()
    assert right_module in capsys.readouterr().err
