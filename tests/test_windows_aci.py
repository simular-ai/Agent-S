import pytest
from gui_agents.aci.WindowsACI import WindowsACI, _normalize_key

@pytest.fixture
def windows_aci():
    """Create a WindowsACI instance for testing"""
    return WindowsACI(top_app_only=True, ocr=False)

def test_windows_normalize_key():
    """Test key normalization in Windows"""
    assert _normalize_key('ctrl') == 'control'
    assert _normalize_key('shift') == 'shift'
    assert _normalize_key('alt') == 'alt'

def test_windows_hotkey_ctrl_normalization(windows_aci):
    """Test ctrl normalization in hotkey command"""
    command = windows_aci.hotkey(['ctrl', 'c'])
    assert 'control' in command
    assert 'ctrl' not in command

def test_windows_click_with_modifier(windows_aci):
    """Test click with modifier keys in Windows"""
    windows_aci.nodes = [{'position': (100, 200), 'size': (50, 50)}]
    command = windows_aci.click(0, hold_keys=['ctrl'])
    assert 'control' in command
    assert 'ctrl' not in command

def test_windows_ocr_integration(windows_aci):
    """Test OCR integration error handling in Windows"""
    with pytest.raises(EnvironmentError, match="OCR SERVER ADDRESS NOT SET"):
        windows_aci.extract_elements_from_screenshot(b"dummy_screenshot")
