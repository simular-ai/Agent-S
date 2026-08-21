from gui_agents.s3.agents.code_agent import extract_code_block, execute_code


class DummyEnvController:
    def __init__(self):
        pass

    def run_python_script(self, code):
        # emulate running python code
        if "print(" in code:
            return {"status": "success", "output": "printed", "returncode": 0}
        return {"status": "success", "output": "ok", "returncode": 0}

    def run_bash_script(self, code, timeout=30):
        return {"status": "success", "output": code, "returncode": 0}


def test_extract_code_block():
    s = "Some text ```python\nprint(1)\n``` more"
    t, code = extract_code_block(s)
    assert t == "python"
    assert "print(1)" in code


def test_execute_code_python():
    controller = DummyEnvController()
    res = execute_code("python", "print(1)", controller)
    assert res["status"] == "success"
    assert "output" in res


def test_execute_code_bash():
    controller = DummyEnvController()
    res = execute_code("bash", "echo hi", controller)
    assert res["status"] == "success"
    assert res["output"] == "echo hi"
