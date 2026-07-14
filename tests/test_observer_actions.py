import pytest

from gui_agents.s3.observer.actions import (
    ActionParseError,
    extract_action_code,
    parse_action_call,
)


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ('agent.click("The blue Continue button")', "click"),
        ('agent.type("The visible search field", "hello", enter=True)', "type"),
        ('agent.scroll("The main page content area", -3)', "scroll"),
        ("agent.hotkey(['ctrl', 'l'])", "hotkey"),
        ("agent.wait(1.5)", "wait"),
        ("agent.done()", "done"),
        ("agent.fail()", "fail"),
    ],
)
def test_parse_supported_literal_actions(source, kind):
    assert parse_action_call(source).kind == kind


@pytest.mark.parametrize(
    "source",
    [
        "agent.open('Terminal')",
        "agent.call_code_agent('do it')",
        "agent.set_cell_values({}, 'a', 'b')",
        "agent.click(__import__('os').system('false'))",
        "agent.click('The blue Continue button'); agent.done()",
        "[agent.done() for _ in range(1)]",
        "other.click('The blue Continue button')",
        "agent.click(*['The blue Continue button'])",
        "agent.click(**{'element_description': 'The blue Continue button'})",
        "agent.hotkey(['ctrl', 'alt', 'delete', 'x'])",
        "agent.wait(999)",
        "agent.scroll('The main page content area', 0)",
    ],
)
def test_rejects_non_literal_or_unsupported_actions(source):
    with pytest.raises(ActionParseError):
        parse_action_call(source)


def test_extract_requires_exactly_one_fence():
    assert extract_action_code("```python\nagent.done()\n```") == "agent.done()"
    with pytest.raises(ActionParseError):
        extract_action_code("agent.done()")
    with pytest.raises(ActionParseError):
        extract_action_code("```agent.done()``` and ```agent.fail()```")
