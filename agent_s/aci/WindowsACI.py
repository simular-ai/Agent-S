# windowsOSACI.py

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import requests
import base64
import psutil
import pyautogui
import time
import pywinauto
from pywinauto import Desktop, Application
import win32gui
import win32process

from .ACI import ACI, agent_action

# Helper functions
def _normalize_key(key: str) -> str:
    """Convert 'control' to 'ctrl' for pyautogui compatibility"""
    return 'control' if key == 'ctrl' else key

def list_apps_in_directories():
    directories_to_search = [
        os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
        os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
    ]
    apps = []
    for directory in directories_to_search:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.exe'):
                        apps.append(file)
    return apps

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Fast vectorized IOU implementation using only NumPy
    boxes1: [N, 4] array of boxes
    boxes2: [M, 4] array of boxes
    Returns: [N, M] array of IOU values
    """
    # Calculate areas of boxes1
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])

    # Calculate areas of boxes2
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Get intersections using broadcasting
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]

    # Calculate intersection areas
    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Calculate union areas
    union = area1[:, None] + area2[None, :] - intersection

    # Calculate IOU
    iou = np.where(union > 0, intersection / union, 0)
    return iou

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
        for proc in psutil.process_iter(['pid', 'name']):
            apps.append(proc.info['name'])
        return apps

    @staticmethod
    def get_top_app(obs: Dict):
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['pid'] == pid:
                return proc.info['name']
        return None

    @staticmethod
    def list_apps_in_directories():
        return list_apps_in_directories()

    def __repr__(self):
        return f"UIElement({self.element})"

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
                        preserved_nodes.append({
                            'position': (x, y),
                            'size': (w, h),
                            'title': element.title(),
                            'text': element.text(),
                            'role': role
                        })

            children = element.children()
            if children:
                for child_element in children:
                    traverse_and_preserve(child_element)

        traverse_and_preserve(tree)
        return preserved_nodes

    def linearize_and_annotate_tree(self, obs: Dict, show_all_elements: bool = False) -> str:
        desktop = Desktop(backend="uia")
        try:
            tree = desktop.window(handle=win32gui.GetForegroundWindow()).wrapper_object()
        except Exception as e:
            print(f"Error accessing foreground window: {e}")
            self.nodes = []
            return ""

        exclude_roles = ["Pane", "Group", "Unknown"]
        preserved_nodes = self.preserve_nodes(UIElement(tree), exclude_roles).copy()

        if not preserved_nodes and show_all_elements:
            preserved_nodes = self.preserve_nodes(UIElement(tree), exclude_roles=[]).copy()

        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
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
            raise

    @agent_action
    def click(self, element_id: int, num_clicks: int = 1, button_type: str = "left"):
        node = self.find_element(element_id)
        coordinates = node["position"]
        sizes = node["size"]

        x = int(coordinates[0] + sizes[0] // 2)
        y = int(coordinates[1] + sizes[1] // 2)

        command = f"import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)})"
        return command

    @agent_action
    def type(self, text: str, overwrite: bool = False, enter: bool = False):
        command = "import pyautogui; "
        if overwrite:
            command += f"pyautogui.hotkey('ctrl', 'a', interval=0.5); pyautogui.press('backspace'); "
        command += f"pyautogui.write({repr(text)}); "
        if enter:
            command += "pyautogui.press('enter'); "
        return command

    @agent_action
    def scroll(self, element_id: int, clicks: int):
        node = self.find_element(element_id)
        coordinates = node["position"]
        sizes = node["size"]

        x = int(coordinates[0] + sizes[0] // 2)
        y = int(coordinates[1] + sizes[1] // 2)
        command = f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
        return command

    @agent_action
    def hotkey(self, keys: List[str]):
        keys = [_normalize_key(k) for k in keys]
        keys = [f"'{key}'" for key in keys]
        command = f"import pyautogui; pyautogui.hotkey({', '.join(keys)}, interval=0.5)"
        return command

    @agent_action
    def drag_and_drop(self, drag_from_id: int, drop_on_id: int):
        node1 = self.find_element(drag_from_id)
        node2 = self.find_element(drop_on_id)

        coordinates1 = node1["position"]
        sizes1 = node1["size"]
        x1 = int(coordinates1[0] + sizes1[0] // 2)
        y1 = int(coordinates1[1] + sizes1[1] // 2)

        coordinates2 = node2["position"]
        sizes2 = node2["size"]
        x2 = int(coordinates2[0] + sizes2[0] // 2)
        y2 = int(coordinates2[1] + sizes2[1] // 2)

        command = f"import pyautogui; pyautogui.moveTo({x1}, {y1}); pyautogui.mouseDown(); pyautogui.moveTo({x2}, {y2}, duration=1.0); pyautogui.mouseUp()"
        return command

    @agent_action
    def open(self, app_or_file_name: str):
        command = f"import pyautogui; import time; pyautogui.hotkey('win', 'r', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"
        return command

    @agent_action
    def switch_applications(self, app_or_file_name):
        command = f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter'); time.sleep(1.0)"
        return command

    @agent_action
    def wait(self, time_in_seconds: float):
        command = f"import time; time.sleep({time_in_seconds})"
        return command
