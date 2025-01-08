from unittest.mock import Mock, patch

import pyatspi
import pytest

from gui_agents.aci.UIElementLinux import UIElement


@pytest.fixture
def mock_accessible():
    mock = Mock()
    mock.name = "Test Window"
    mock.getRole.return_value = pyatspi.ROLE_WINDOW
    mock.getState.return_value.contains.return_value = True
    return mock


@pytest.fixture
def ui_element(mock_accessible):
    return UIElement(mock_accessible)


def test_role(ui_element, mock_accessible):
    """Test role retrieval"""
    mock_accessible.getRoleName.return_value = "window"
    assert ui_element.role() == "window"


def test_position(ui_element, mock_accessible):
    """Test position retrieval"""
    mock_accessible.getPosition.return_value = (100, 200)
    assert ui_element.position() == (100, 200)


def test_size(ui_element, mock_accessible):
    """Test size retrieval"""
    mock_accessible.getSize.return_value = (300, 400)
    assert ui_element.size() == (300, 400)
