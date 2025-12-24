import os
from gui_agents.s3 import cli_app
from PIL import Image


class FailAgent:
    def __init__(self, out_path):
        self.called = False
        self.out_path = out_path

    def predict(self, instruction: str, observation: dict):
        if not self.called:
            self.called = True
            code = f"with open(r'{self.out_path}', 'w', encoding='utf-8') as f: f.write('fail')"
            return {"reflection": "created"}, [code, "fail"]
        return {"reflection": "done"}, ["done"]


class DoneTokenAgent:
    def __init__(self, out_path):
        self.called = False
        self.out_path = out_path

    def predict(self, instruction: str, observation: dict):
        # Return a body token 'done' as the code_str, which should be treated as terminal token
        if not self.called:
            self.called = True
            return {"reflection": "stopping"}, ["done"]
        return {"reflection": "done"}, ["done"]


def _stub_screenshot():
    return Image.new("RGB", (16, 16), "white")


def test_final_status_fail_executes_file(tmp_path, monkeypatch):
    out = tmp_path / "fail_out.txt"
    agent = FailAgent(str(out))
    # Stub screenshot to avoid desktop dependency
    monkeypatch.setattr("pyautogui.screenshot", lambda: _stub_screenshot())

    # Run agent; require_exec_confirmation=False to avoid interactive prompt
    cli_app.run_agent(agent, "Create file", 16, 16, require_exec_confirmation=False)

    assert out.exists(), "Output file should exist after agent execution for fail status"
    assert out.read_text(encoding="utf-8") == "fail"


def test_done_token_does_not_execute_code(tmp_path, monkeypatch):
    out = tmp_path / "done_out.txt"
    agent = DoneTokenAgent(str(out))
    monkeypatch.setattr("pyautogui.screenshot", lambda: _stub_screenshot())

    cli_app.run_agent(agent, "No-op", 16, 16, require_exec_confirmation=False)

    assert not out.exists(), "No file should be created when agent returns code token 'done'"
