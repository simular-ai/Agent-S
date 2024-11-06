import logging
from typing import Dict, Optional, List
from xml.etree import ElementTree as ET

logger = logging.getLogger("openaci.agent")

# New XML namespaces for OSWorld
# state_ns = "https://accessibility.ubuntu.example.org/ns/state"
# component_ns = "https://accessibility.ubuntu.example.org/ns/component"
# attributes_ns = "https://accessibility.windows.example.org/ns/attributes"
# value_ns = "https://accessibility.ubuntu.example.org/ns/value"


# Old XML namespaces for OSWorld
state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"
attributes_ns = "uri:deskat:attributes.at-spi.gnome.org"
value_ns = "uri:deskat:value.at-spi.gnome.org"

class UIElement:
    def __init__(self, node: ET.ElementTree): 
        self.node = node

    @staticmethod 
    def systemWideElement(tree: ET.ElementTree) -> 'UIElement':
        """Find the active window in the ElementTree"""
        root = tree.getroot()
        for app in root:
            for window in app:
                if window.get(f"{{{state_ns}}}active") == "true":
                    return UIElement(app)
        return None
    
    @staticmethod
    def nodeFromObservation(tree: str) -> 'UIElement':
        """Create a UIElement from an XML string"""
        return UIElement(ET.ElementTree(ET.fromstring(tree)))

    @staticmethod
    def nodeFromTree(tree_data: str) -> Optional['UIElement']:
        """Create UIElement from XML string"""
        try:
            tree = ET.ElementTree(ET.fromstring(tree_data))
            return UIElement(tree.getroot())
        except ET.ParseError:
            logger.error("Failed to parse XML tree")
            return None

    @staticmethod
    def get_current_applications(obs: Dict) -> List[str]:
        """This should be implemented based on the available data in OSWorld"""
        return []  # Or implement based on available data
   
    @property
    def states(self) -> Dict[str, str]:
        """Get all state attributes with state namespace"""
        return {k: v for k, v in self.node.attrib.items() if state_ns in k}

    @property
    def attributes(self) -> Dict[str, str]:
        """Get all attributes with attributes namespace"""
        return {k: v for k, v in self.node.attrib.items() if attributes_ns in k}

    @property
    def component(self) -> Optional[Dict[str, str]]:
        """Get component attributes like position and size"""
        comp_attrs = {k: v for k, v in self.node.attrib.items() if component_ns in k}
        return comp_attrs if comp_attrs else None

    @property
    def value(self) -> Optional[Dict[str, str]]:
        """Get value attributes"""
        val = self.node.get(f"{{{value_ns}}}current")
        return {f"{{{value_ns}}}current": val} if val else None

    @property
    def text(self) -> str:
        """Get element text content"""
        return self.node.text or ''

    def role(self) -> str:
        """Get element role"""
        return self.node.tag

    def position(self) -> tuple:
        """Get element position from component attributes"""
        pos_str = self.node.get(f"{{{component_ns}}}screencoord", "(0,0)")
        return eval(pos_str)

    def size(self) -> tuple:
        """Get element size from component attributes"""
        size_str = self.node.get(f"{{{component_ns}}}size", "(0,0)")
        return eval(size_str)

    def isValid(self) -> bool:
        """Check if element has valid component info"""
        return bool(self.component)

    def parse(self) -> Dict:
        """Parse element into dictionary with key properties"""
        return {
            "position": self.position(),
            "size": self.size(),
            "title": self.node.get("name", ""),
            "text": self.text,
            "role": self.role(),
        }

    def children(self):
        """Get child elements"""
        return [UIElement(child) for child in self.node]

    def __repr__(self):
        return f"UIElement({self.node.tag})"

def traverse_and_print(node: UIElement):
    print(node.attributes)
    print(node.role())
    for child in node.children():
        traverse_and_print(child)

if __name__ == "__main__":
    # Example usage with ElementTree
    tree = ET.parse("example.xml")  # Load your accessibility tree XML
    active_node = UIElement.systemWideElement(tree)
    if active_node:
        traverse_and_print(active_node)

