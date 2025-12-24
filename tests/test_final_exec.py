import os
from gui_agents.s3 import cli_app
from PIL import Image


class FakeAgent:
    def __init__(self, out_path):
        self.called = False
        self.out_path = out_path

    def predict(self, instruction: str, observation: dict):
        # On first call, return code that writes a small file and indicate final status
        if not self.called:
            self.called = True
            code = f"with open(r'{self.out_path}', 'w', encoding='utf-8') as f: f.write('ok')"
            return {"reflection": "created"}, [code, "done"]
        # Subsequent calls return done token
        return {"reflection": "done"}, ["done"]


def test_agent_final_exec_writes_file(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"

    agent = FakeAgent(str(out))

    # Stub screenshot to avoid desktop dependency
    monkeypatch.setattr("pyautogui.screenshot", lambda: Image.new("RGB", (16, 16), "white"))

    # Run agent; require_exec_confirmation=False to avoid interactive prompt
    cli_app.run_agent(agent, "Create file", 16, 16, require_exec_confirmation=False)

    # Check file was created and contains expected content
    assert out.exists(), "Output file should exist after agent execution"
    assert out.read_text(encoding='utf-8') == "ok"
