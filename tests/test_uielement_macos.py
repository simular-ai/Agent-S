from typing import Dict
from unittest.mock import Mock, patch

import pytest

from gui_agents.s1.aci.MacOSACI import UIElement


@pytest.fixture
def mock_ax_element():
    mock_element = Mock()
    mock_element.__repr__ = lambda x: "x:100 y:200"
    return mock_element


@pytest.fixture
def mock_size_element():
    mock_element = Mock()
    mock_element.__repr__ = lambda x: "w:300 h:400"
    return mock_element


@pytest.fixture
def ui_element(mock_ax_element):
    element = UIElement(mock_ax_element)
    return element


def test_position_parsing(ui_element, mock_ax_element):
    """Test position parsing from AX element"""
    with patch.object(ui_element, "attribute", return_value=mock_ax_element):
        pos = ui_element.position()
        assert pos == (100.0, 200.0)


def test_size_parsing(ui_element, mock_size_element):
    """Test size parsing from AX element"""
    with patch.object(ui_element, "attribute", return_value=mock_size_element):
        size = ui_element.size()
        assert size == (300.0, 400.0)


def test_get_current_applications(obs: Dict):
    """Test getting list of current applications"""
    with patch("AppKit.NSWorkspace") as mock_workspace:
        mock_app = Mock()
        mock_app.activationPolicy.return_value = 0
        mock_app.localizedName.return_value = "TestApp"
        mock_workspace.sharedWorkspace.return_value.runningApplications.return_value = [
            mock_app
        ]

        apps = UIElement.get_current_applications(obs)
        assert apps == ["TestApp"]
