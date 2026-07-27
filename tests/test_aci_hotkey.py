import pytest
from gui_agents.s1.aci.WindowsOSACI import WindowsACI
from gui_agents.s1.aci.MacOSACI import MacOSACI


def test_windows_hotkey_single_string():
    aci = WindowsACI()
    command = aci.hotkey("enter")
    assert "pyautogui.hotkey('enter', interval=0.5)" in command


def test_windows_hotkey_list():
    aci = WindowsACI()
    command = aci.hotkey(["alt", "tab"])
    assert "pyautogui.hotkey('alt', 'tab', interval=0.5)" in command


def test_macos_hotkey_single_string():
    aci = MacOSACI()
    command = aci.hotkey("enter")
    assert "pyautogui.hotkey('enter', interval=1)" in command


def test_macos_hotkey_list():
    aci = MacOSACI()
    command = aci.hotkey(["cmd", "c"])
    assert "pyautogui.hotkey('command', 'c', interval=1)" in command
