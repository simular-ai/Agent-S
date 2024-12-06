import pytest

from gui_agents.aci.UIElementBase import UIElementBase


def test_uielement_base_is_abstract():
    """Test that UIElementBase cannot be instantiated directly"""
    with pytest.raises(TypeError):
        UIElementBase()
