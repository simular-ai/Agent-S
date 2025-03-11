from unittest.mock import Mock, patch

import pytest

from gui_agents.s1.aci.ACI import ACI, _normalize_key


@pytest.fixture
def aci():
    return ACI(top_app_only=True, ocr=False)


def test_normalize_key():
    """Test key normalization"""
    assert _normalize_key("cmd") == "command"
    assert _normalize_key("ctrl") == "ctrl"
    assert _normalize_key("shift") == "shift"


def test_hotkey_cmd_normalization(aci):
    """Test cmd normalization in hotkey command"""
    command = aci.hotkey(["cmd", "c"])
    assert "command" in command
    assert "cmd" not in command


def test_click_with_cmd_key(aci):
    """Test cmd normalization in click command"""
    aci.nodes = [{"position": (100, 200), "size": (50, 50)}]
    command = aci.click(0, hold_keys=["cmd"])
    assert "command" in command
    assert "cmd" not in command


def test_type_with_overwrite(aci):
    """Test type command with overwrite"""
    aci.nodes = [{"position": (100, 200), "size": (50, 50)}]
    command = aci.type(0, "test", overwrite=True)
    assert "command" in command or "ctrl" in command
    assert "backspace" in command
