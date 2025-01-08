import xml.etree.ElementTree as ET

import pytest

from gui_agents.aci.UIElementOSWorld import UIElement


@pytest.fixture
def sample_xml():
    return """
    <root>
        <application name="TestApp">
            <window uri:deskat:state.at-spi.gnome.org:active="true">
                <button uri:deskat:component.at-spi.gnome.org:screencoord="(100,200)"
                        uri:deskat:component.at-spi.gnome.org:size="(300,400)">
                    Click me
                </button>
            </window>
        </application>
    </root>
    """


@pytest.fixture
def ui_element(sample_xml):
    tree = ET.ElementTree(ET.fromstring(sample_xml))
    return UIElement(tree.getroot())


def test_nodeFromTree(sample_xml):
    """Test creating UIElement from XML string"""
    element = UIElement.nodeFromTree(sample_xml)
    assert element is not None
    assert isinstance(element, UIElement)


def test_position(ui_element):
    """Test position extraction from XML"""
    button = ui_element.children()[0].children()[0]
    assert button.position() == (100, 200)


def test_size(ui_element):
    """Test size extraction from XML"""
    button = ui_element.children()[0].children()[0]
    assert button.size() == (300, 400)
