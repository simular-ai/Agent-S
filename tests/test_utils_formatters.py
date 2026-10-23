from gui_agents.s3.utils.common_utils import (extract_agent_functions,
                                              parse_code_from_string)


def test_parse_code_from_string_normal():
    s = "Intro ```python\nagent.wait(1)\n``` end"
    code = parse_code_from_string(s)
    assert "agent.wait" in code


def test_extract_agent_functions():
    code = "agent.wait(1); agent.click('ok')"
    funcs = extract_agent_functions(code)
    assert any("agent.wait" in f for f in funcs)
    assert any("agent.click" in f for f in funcs)
