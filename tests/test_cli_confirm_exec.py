import builtins
import pytest


class OneStepAgent:
    def __init__(self, code):
        self.code = code
        self._called = False

    def reset(self):
        self._called = False

    def predict(self, instruction: str, observation: dict):
        if not self._called:
            self._called = True
            return {}, [self.code, "done"]
        return {}, ["done"]


def test_execute_with_gui_confirmation(monkeypatch):
    from gui_agents.s3 import cli_app

    agent = OneStepAgent("print('hello')")

    called = []

    def fake_execute(code_str):
        called.append(code_str)

    monkeypatch.setattr(cli_app, "execute_code", fake_execute)
    monkeypatch.setattr(cli_app, "show_permission_dialog", lambda code, desc: True)

    # Run with require_exec_confirmation=True
    cli_app.run_agent(agent, "instr", 100, 100, require_exec_confirmation=True)

    assert len(called) == 1
    assert "print('hello')" in called[0]


def test_execute_with_input_confirmation(monkeypatch):
    from gui_agents.s3 import cli_app

    agent = OneStepAgent("print('hi')")

    called = []

    def fake_execute(code_str):
        called.append(code_str)

    monkeypatch.setattr(cli_app, "execute_code", fake_execute)
    monkeypatch.setattr(cli_app, "show_permission_dialog", lambda code, desc: False)

    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")

    cli_app.run_agent(agent, "instr", 100, 100, require_exec_confirmation=True)

    assert len(called) == 1
    assert "print('hi')" in called[0]


def test_decline_execution(monkeypatch):
    from gui_agents.s3 import cli_app

    agent = OneStepAgent("print('no')")

    called = []

    def fake_execute(code_str):
        called.append(code_str)

    monkeypatch.setattr(cli_app, "execute_code", fake_execute)
    monkeypatch.setattr(cli_app, "show_permission_dialog", lambda code, desc: False)

    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")

    cli_app.run_agent(agent, "instr", 100, 100, require_exec_confirmation=True)

    assert len(called) == 0
