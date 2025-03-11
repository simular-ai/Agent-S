import base64
import os
import platform
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import requests
from gui_agents.s1.utils.common_utils import box_iou

if platform.system() == "Windows":
    import pywinauto
    from pywinauto import Desktop
    import win32gui
    import win32process

from gui_agents.s1.aci.ACI import ACI, agent_action


# Helper functions
def _normalize_key(key: str) -> str:
    """Convert 'ctrl' to 'control' for pyautogui compatibility"""
    return "ctrl" if key == "control" else key


def list_apps_in_directories():
    directories_to_search = [
        os.environ.get("PROGRAMFILES", "C:\\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"),
    ]
    apps = []
    for directory in directories_to_search:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".exe"):
                        apps.append(file)
    return apps


# WindowsACI Class
class WindowsACI(ACI):
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        super().__init__(top_app_only=top_app_only, ocr=ocr)
        self.nodes = []
        self.all_apps = list_apps_in_directories()

    def get_active_apps(self, obs: Dict) -> List[str]:
        return UIElement.get_current_applications(obs)

    def get_top_app(self, obs: Dict) -> str:
        return UIElement.get_top_app(obs)

    def preserve_nodes(self, tree, exclude_roles=None):
        if exclude_roles is None:
            exclude_roles = set()

        preserved_nodes = []

        def traverse_and_preserve(element):
            role = element.role()

            if role not in exclude_roles:
                position = element.position()
                size = element.size()
                if position and size:
                    x, y = position
                    w, h = size

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        preserved_nodes.append(
                            {
                                "position": (x, y),
                                "size": (w, h),
                                "title": element.title(),
                                "text": element.text(),
                                "role": role,
                            }
                        )

            children = element.children()
            if children:
                for child_element in children:
                    traverse_and_preserve(child_element)

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
                        for _, _, box in ocr_bboxes["results"]
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
                    zip(ocr_bboxes["results"], max_ious)
                ):
                    if max_iou < 0.1:
                        x1 = int(box.get("left", 0))
                        y1 = int(box.get("top", 0))
                        x2 = int(box.get("right", 0))
                        y2 = int(box.get("bottom", 0))

                        linearized_accessibility_tree.append(
                            f"{preserved_nodes_index}\tButton\t\t{content}\t\t"
                        )

                        node = {
                            "position": (x1, y1),
                            "size": (x2 - x1, y2 - y1),
                            "title": "",
                            "text": content,
                            "role": "Button",
                        }
                        preserved_nodes.append(node)
                        preserved_nodes_index += 1

        return linearized_accessibility_tree, preserved_nodes

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        desktop = Desktop(backend="uia")
        try:
            tree = desktop.window(
                handle=win32gui.GetForegroundWindow()
            ).wrapper_object()
        except Exception as e:
            print(f"Error accessing foreground window: {e}")
            self.nodes = []
            return ""

        exclude_roles = ["Pane", "Group", "Unknown"]
        preserved_nodes = self.preserve_nodes(UIElement(tree), exclude_roles).copy()

        if not preserved_nodes and show_all_elements:
            preserved_nodes = self.preserve_nodes(
                UIElement(tree), exclude_roles=[]
            ).copy()

        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        if self.ocr:
            screenshot = obs.get("screenshot", None)
            if screenshot is not None:
                # return tree_elements, preserved_nodes
                tree_elements, preserved_nodes = self.add_ocr_elements(
                    screenshot, tree_elements, preserved_nodes
                )

        self.nodes = preserved_nodes
        return "\n".join(tree_elements)

    def find_element(self, element_id: int) -> Dict:
        if not self.nodes:
            print("No elements found in the accessibility tree.")
            raise IndexError("No elements to select.")
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
        command = f"import pyautogui; import time; pyautogui.hotkey('win', 'r', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"
        return command

    @agent_action
    def switch_applications(self, app_or_file_name):
        """Switch to a different application. Utility function to use instead of alt+tab
        Args:
            app_or_file_name:str, the name of the application or file to switch to
        """
        command = f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"
        return command

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
        x = int(coordinates[0] + sizes[0] // 2)
        y = int(coordinates[1] + sizes[1] // 2)

        command = "import pyautogui; "

        # Normalize any 'ctrl' to 'control'
        hold_keys = [_normalize_key(k) for k in hold_keys]

        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
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
            enter:bool Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        try:
            node = self.find_element(element_id) if element_id is not None else None
        except:
            node = None

        if node is not None:
            coordinates = node["position"]
            sizes = node["size"]

            x = int(coordinates[0] + sizes[0] // 2)
            y = int(coordinates[1] + sizes[1] // 2)

            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "

            if overwrite:
                command += f"pyautogui.hotkey('ctrl', 'a', interval=0.5); pyautogui.press('backspace'); "

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            command = "import pyautogui; "

            if overwrite:
                command += f"pyautogui.hotkey('ctrl', 'a', interval=0.5); pyautogui.press('backspace'); "

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

        x1 = int(coordinates1[0] + sizes1[0] // 2)
        y1 = int(coordinates1[1] + sizes1[1] // 2)

        x2 = int(coordinates2[0] + sizes2[0] // 2)
        y2 = int(coordinates2[1] + sizes2[1] // 2)

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.0); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

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

        coordinates = node["position"]
        sizes = node["size"]

        x = int(coordinates[0] + sizes[0] // 2)
        y = int(coordinates[1] + sizes[1] // 2)
        command = (
            f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
        )
        return command

    @agent_action
    def hotkey(self, keys: List[str]):
        """Press a hotkey combination
        Args:
            keys:List[str] the keys to press in combination in a list format (e.g. ['shift', 'c'])
        """
        keys = [_normalize_key(k) for k in keys]
        keys = [f"'{key}'" for key in keys]
        command = f"import pyautogui; pyautogui.hotkey({', '.join(keys)}, interval=0.5)"
        return command

    @agent_action
    def hold_and_press(self, hold_keys: List[str], press_keys: List[str]):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List[str], list of keys to hold
            press_keys:List[str], list of keys to press in a sequence
        """
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
        command = f"import time; time.sleep({time})"
        return command

    @agent_action
    def done(self):
        """End the current task with a success"""
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure"""
        return """FAIL"""


# UIElement Class
class UIElement:
    def __init__(self, element=None):
        if isinstance(element, pywinauto.application.WindowSpecification):
            self.element = element.wrapper_object()
        else:
            self.element = element  # This should be a control wrapper

    def get_attribute_names(self):
        return list(self.element.element_info.get_properties().keys())

    def attribute(self, key: str):
        props = self.element.element_info.get_properties()
        return props.get(key, None)

    def children(self):
        try:
            return [UIElement(child) for child in self.element.children()]
        except Exception as e:
            print(f"Error accessing children: {e}")
            return []

    def role(self):
        return self.element.element_info.control_type

    def position(self):
        rect = self.element.rectangle()
        return (rect.left, rect.top)

    def size(self):
        rect = self.element.rectangle()
        return (rect.width(), rect.height())

    def title(self):
        return self.element.element_info.name

    def text(self):
        return self.element.window_text()

    def isValid(self):
        return self.position() is not None and self.size() is not None

    def parse(self):
        position = self.position()
        size = self.size()
        return {
            "position": position,
            "size": size,
            "title": self.title(),
            "text": self.text(),
            "role": self.role(),
        }

    @staticmethod
    def get_current_applications(obs: Dict):
        apps = []
        for proc in psutil.process_iter(["pid", "name"]):
            apps.append(proc.info["name"])
        return apps

    @staticmethod
    def get_top_app(obs: Dict):
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["pid"] == pid:
                return proc.info["name"]
        return None

    @staticmethod
    def list_apps_in_directories():
        return list_apps_in_directories()

    @staticmethod
    def systemWideElement():
        desktop = Desktop(backend="uia")
        return UIElement(desktop)

    def __repr__(self):
        return f"UIElement({self.element})"
