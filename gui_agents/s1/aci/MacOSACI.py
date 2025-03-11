import base64
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import platform
from gui_agents.s1.utils.common_utils import box_iou

if platform.system() == "Darwin":
    from AppKit import *
    from ApplicationServices import (
        AXUIElementCopyAttributeNames,
        AXUIElementCopyAttributeValue,
        AXUIElementCreateSystemWide,
    )

from gui_agents.s1.aci.ACI import ACI, agent_action


def _normalize_key(key: str) -> str:
    """Convert 'cmd' to 'command' for pyautogui compatibility"""
    return "command" if key == "cmd" else key


def list_apps_in_directories(directories):
    apps = []
    for directory in directories:
        if os.path.exists(directory):
            directory_apps = [
                app for app in os.listdir(directory) if app.endswith(".app")
            ]
            apps.extend(directory_apps)
    return apps


class MacOSACI(ACI):
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        super().__init__(top_app_only=top_app_only, ocr=ocr)
        # Directories to search for applications in MacOS
        directories_to_search = ["/System/Applications", "/Applications"]
        self.all_apps = list_apps_in_directories(directories_to_search)

    def get_active_apps(self, obs: Dict) -> List[str]:
        return UIElement.get_current_applications(obs)

    def get_top_app(self, obs: Dict) -> str:
        return UIElement.get_top_app(obs)

    def preserve_nodes(self, tree, exclude_roles=None):
        if exclude_roles is None:
            exclude_roles = set()

        preserved_nodes = []

        # Inner function to recursively traverse the accessibility tree
        def traverse_and_preserve(element):
            role = element.attribute("AXRole")

            if role not in exclude_roles:
                # TODO: get coordinate values directly from interface
                position = element.attribute("AXPosition")
                size = element.attribute("AXSize")
                if position and size:
                    pos_parts = position.__repr__().split().copy()
                    # Find the parts containing 'x:' and 'y:'
                    x_part = next(part for part in pos_parts if part.startswith("x:"))
                    y_part = next(part for part in pos_parts if part.startswith("y:"))

                    # Extract the numerical values after 'x:' and 'y:'
                    x = float(x_part.split(":")[1])
                    y = float(y_part.split(":")[1])

                    size_parts = size.__repr__().split().copy()
                    # Find the parts containing 'Width:' and 'Height:'
                    width_part = next(
                        part for part in size_parts if part.startswith("w:")
                    )
                    height_part = next(
                        part for part in size_parts if part.startswith("h:")
                    )

                    # Extract the numerical values after 'Width:' and 'Height:'
                    w = float(width_part.split(":")[1])
                    h = float(height_part.split(":")[1])

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        preserved_nodes.append(
                            {
                                "position": (x, y),
                                "size": (w, h),
                                "title": str(element.attribute("AXTitle")),
                                "text": str(element.attribute("AXDescription"))
                                or str(element.attribute("AXValue")),
                                "role": str(element.attribute("AXRole")),
                            }
                        )

            children = element.children()
            if children:
                for child_ref in children:
                    child_element = UIElement(child_ref)
                    traverse_and_preserve(child_element)

        # Start traversing from the given element
        traverse_and_preserve(tree)

        return preserved_nodes

    def extract_elements_from_screenshot(self, screenshot: bytes) -> Dict[str, Any]:
        url = os.environ.get("OCR_SERVER_ADDRESS")
        if not url:
            raise EnvironmentError("OCR SERVER ADDRESS NOT SET")

        encoded_screenshot = base64.b64encode(screenshot).decode("utf-8")
        response = requests.post(url, json={"img_bytes": encoded_screenshot})

        if response.status_code != 200:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "results": [],
            }
        return response.json()

    def add_ocr_elements(
        self,
        screenshot,
        linearized_accessibility_tree: List[str],
        preserved_nodes: List[Dict],
    ) -> Tuple[List[str], List[Dict]]:
        """
        Add OCR-detected elements to the accessibility tree if they don't overlap with existing elements
        Uses optimized NumPy implementation
        """
        # Convert preserved nodes to numpy array of bounding boxes
        if preserved_nodes:
            tree_bboxes = np.array(
                [
                    [
                        node["position"][0],
                        node["position"][1],
                        node["position"][0] + node["size"][0],
                        node["position"][1] + node["size"][1],
                    ]
                    for node in preserved_nodes
                ],
                dtype=np.float32,
            )
        else:
            tree_bboxes = np.empty((0, 4), dtype=np.float32)

        try:
            ocr_bboxes = self.extract_elements_from_screenshot(screenshot)
        except Exception as e:
            print(f"Error: {e}")
            ocr_bboxes = []
        else:
            if ocr_bboxes:
                preserved_nodes_index = len(preserved_nodes)

                # Convert OCR boxes to numpy array
                ocr_boxes_array = np.array(
                    [
                        [
                            int(box.get("left", 0)),
                            int(box.get("top", 0)),
                            int(box.get("right", 0)),
                            int(box.get("bottom", 0)),
                        ]
                        for _, _, box in ocr_bboxes
                    ],
                    dtype=np.float32,
                )

                # Calculate max IOUs efficiently
                if len(tree_bboxes) > 0:
                    max_ious = box_iou(tree_bboxes, ocr_boxes_array).max(axis=0)
                else:
                    max_ious = np.zeros(len(ocr_boxes_array))

                # Process boxes with low IOU
                for idx, ((_, content, box), max_iou) in enumerate(
                    zip(ocr_bboxes, max_ious)
                ):
                    if max_iou < 0.1:
                        x1 = int(box.get("left", 0))
                        y1 = int(box.get("top", 0))
                        x2 = int(box.get("right", 0))
                        y2 = int(box.get("bottom", 0))

                        linearized_accessibility_tree.append(
                            f"{preserved_nodes_index}\tAXButton\t\t{content}\t\t"
                        )

                        node = {
                            "position": (x1, y1),
                            "size": (x2 - x1, y2 - y1),
                            "title": "",
                            "text": content,
                            "role": "AXButton",
                        }
                        preserved_nodes.append(node)
                        preserved_nodes_index += 1

        return linearized_accessibility_tree, preserved_nodes

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        accessibility_tree = obs["accessibility_tree"]
        screenshot = obs["screenshot"]
        self.top_app = (
            NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
        )
        tree = UIElement(accessibility_tree.attribute("AXFocusedApplication"))
        exclude_roles = ["AXGroup", "AXLayoutArea", "AXLayoutItem", "AXUnknown"]
        preserved_nodes = self.preserve_nodes(tree, exclude_roles).copy()
        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        if self.ocr:
            tree_elements, preserved_nodes = self.add_ocr_elements(
                screenshot, tree_elements, preserved_nodes, "AXButton"
            )

        self.nodes = preserved_nodes
        return "\n".join(tree_elements)

    def find_element(self, element_id: int) -> Dict:
        try:
            return self.nodes[element_id]
        except IndexError:
            print("The index of the selected element was out of range.")
            self.index_out_of_range_flag = True
            return self.nodes[0]

    @agent_action
    def open(self, app_or_file_name: str):
        """Open an application or file
        Args:
            app_or_file_name:str, the name of the application or file to open
        """
        return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"

    @agent_action
    def switch_applications(self, app_or_file_name):
        """Switch to a different an application. Utility function to use instead of command+tab
        Args:
            app_or_file_name:str, the name of the application or file to switch to
        """
        return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"

    @agent_action
    def click(
        self,
        element_id: int,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            element_id:int, ID of the element to click on
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        node = self.find_element(element_id)
        coordinates: Tuple[int, int] = node["position"]
        sizes: Tuple[int, int] = node["size"]

        # Calculate the center of the element
        x = coordinates[0] + sizes[0] // 2
        y = coordinates[1] + sizes[1] // 2

        command = "import pyautogui; "

        # Normalize any 'cmd' to 'command'
        hold_keys = [_normalize_key(k) for k in hold_keys]

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    @agent_action
    def type(
        self,
        element_id: int = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into the element
        Args:
            element_id:int ID of the element to type into. If not provided, typing will start at the current cursor location.
            text:str the text to type
            overwrite:bool Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool Assign it to True if the enter (return) key should be pressed after typing the text, otherwise assign it to False.
        """
        try:
            # Use the provided element_id or default to None
            node = self.find_element(element_id) if element_id is not None else None
        except:
            node = None

        if node is not None:
            # If a node is found, retrieve its coordinates and size
            coordinates = node["position"]
            sizes = node["size"]

            # Calculate the center of the element
            x = coordinates[0] + sizes[0] // 2
            y = coordinates[1] + sizes[1] // 2

            # Start typing at the center of the element
            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "

            if overwrite:
                # Use 'command' instead of 'cmd'
                command += f"pyautogui.hotkey('command', 'a', interval=1); pyautogui.press('backspace'); "

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # If no element is found, start typing at the current cursor location
            command = "import pyautogui; "

            if overwrite:
                # Use 'command' instead of 'cmd'
                command += f"pyautogui.hotkey('command', 'a', interval=1); pyautogui.press('backspace'); "

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "

        return command

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save facts, elements, texts, etc. to a long-term knowledge for reuse during this task. Can be used for copy-pasting text, saving elements, etc. Use this instead of ctrl+c, ctrl+v.
        Args:
            text:List[str] the text to save to the knowledge
        """
        self.notes.extend(text)
        return """WAIT"""

    @agent_action
    def drag_and_drop(self, drag_from_id: int, drop_on_id: int, hold_keys: List = []):
        """Drag element1 and drop it on element2.
        Args:
            drag_from_id:int ID of element to drag
            drop_on_id:int ID of element to drop on
            hold_keys:List list of keys to hold while dragging
        """
        node1 = self.find_element(drag_from_id)
        node2 = self.find_element(drop_on_id)
        coordinates1 = node1["position"]
        sizes1 = node1["size"]

        coordinates2 = node2["position"]
        sizes2 = node2["size"]

        # Calculate the center of the element
        x1 = coordinates1[0] + sizes1[0] // 2
        y1 = coordinates1[1] + sizes1[1] // 2

        x2 = coordinates2[0] + sizes2[0] // 2
        y2 = coordinates2[1] + sizes2[1] // 2

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    @agent_action
    def scroll(self, element_id: int, clicks: int):
        """Scroll in the specified direction inside the specified element
        Args:
            element_id:int ID of the element to scroll in
            clicks:int the number of clicks to scroll can be positive (up) or negative (down).
        """
        try:
            node = self.find_element(element_id)
        except:
            node = self.find_element(0)
        # print(node.attrib)
        coordinates = node["position"]
        sizes = node["size"]

        # Calculate the center of the element
        x = coordinates[0] + sizes[0] // 2
        y = coordinates[1] + sizes[1] // 2
        return (
            f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
        )

    @agent_action
    def hotkey(self, keys: List):
        """Press a hotkey combination
        Args:
            keys:List the keys to press in combination in a list format (e.g. ['shift', 'c'])
        """
        # Normalize any 'cmd' to 'command'
        keys = [_normalize_key(k) for k in keys]
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)}, interval=1)"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """
        # Normalize any 'cmd' to 'command' in both lists
        hold_keys = [_normalize_key(k) for k in hold_keys]
        press_keys = [_normalize_key(k) for k in press_keys]

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"""import time; time.sleep({time})"""

    @agent_action
    def done(self):
        """End the current task with a success"""
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure"""
        return """FAIL"""


class UIElement(object):

    def __init__(self, ref=None):
        self.ref = ref

    def getAttributeNames(self):
        error_code, attributeNames = AXUIElementCopyAttributeNames(self.ref, None)
        return list(attributeNames)

    def attribute(self, key: str):
        error, value = AXUIElementCopyAttributeValue(self.ref, key, None)
        return value

    def children(self):
        return self.attribute("AXChildren")

    def systemWideElement():
        ref = AXUIElementCreateSystemWide()
        return UIElement(ref)

    def role(self):
        return self.attribute("AXRole")

    def position(self):
        pos = self.attribute("AXPosition")
        if pos is None:
            return None
        pos_parts = pos.__repr__().split().copy()
        # Find the parts containing 'x:' and 'y:'
        x_part = next(part for part in pos_parts if part.startswith("x:"))
        y_part = next(part for part in pos_parts if part.startswith("y:"))

        # Extract the numerical values after 'x:' and 'y:'
        x = float(x_part.split(":")[1])
        y = float(y_part.split(":")[1])

        return (x, y)

    def size(self):
        size = self.attribute("AXSize")
        if size is None:
            return None
        size_parts = size.__repr__().split().copy()
        # Find the parts containing 'Width:' and 'Height:'
        width_part = next(part for part in size_parts if part.startswith("w:"))
        height_part = next(part for part in size_parts if part.startswith("h:"))

        # Extract the numerical values after 'Width:' and 'Height:'
        w = float(width_part.split(":")[1])
        h = float(height_part.split(":")[1])
        return (w, h)

    def isValid(self):
        if self.position() is not None and self.size() is not None:
            return True

    def parse(self, element):
        position = element.position(element)
        size = element.size(element)
        return {
            "position": position,
            "size": size,
            "title": str(element.attribute("AXTitle")),
            "text": str(element.attribute("AXDescription"))
            or str(element.attribute("AXValue")),
            "role": str(element.attribute("AXRole")),
        }

    @staticmethod
    def get_current_applications(obs: Dict):
        # Get the shared workspace instance
        workspace = NSWorkspace.sharedWorkspace()

        # Get a list of running applications
        running_apps = workspace.runningApplications()

        # Iterate through the list and print each application's name
        current_apps = []
        for app in running_apps:
            if app.activationPolicy() == 0:
                app_name = app.localizedName()
                current_apps.append(app_name)

        return current_apps

    @staticmethod
    def list_apps_in_directories():
        directories_to_search = ["/System/Applications", "/Applications"]
        apps = []
        for directory in directories_to_search:
            if os.path.exists(directory):
                directory_apps = [
                    app for app in os.listdir(directory) if app.endswith(".app")
                ]
                apps.extend(directory_apps)
        return apps

    @staticmethod
    def get_top_app(obs: Dict):
        return NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()

    def __repr__(self):
        return "UIElement%s" % (self.ref)
