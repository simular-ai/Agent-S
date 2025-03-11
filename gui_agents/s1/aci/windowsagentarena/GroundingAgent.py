import base64
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import numpy as np
import requests
from gui_agents.s1.utils.common_utils import box_iou

logger = logging.getLogger("desktopenv.agent")


state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


class GroundingAgent:
    def __init__(self, vm_version: str, top_app=None, top_app_only=True, ocr=True):
        self.active_apps = set()
        self.top_app = top_app
        self.top_app_only = (
            top_app_only  # Only include top app in the accessibility tree
        )
        self.ocr = ocr
        self.index_out_of_range_flag = False
        self.app_setup_code = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""

        self.top_active_app = None
        self.notes = []
        self.clipboard = ""

        # TODO: this is terrible, fix this
        # global state_ns, component_ns, attributes_ns, value_ns
        # if vm_version == "old":
        #     state_ns = "uri:deskat:state.at-spi.gnome.org"
        #     component_ns = "uri:deskat:component.at-spi.gnome.org"
        # elif vm_version == 'win':
        #     state_ns = "uri:deskat:state.at-spi.gnome.org"
        #     component_ns = "uri:deskat:component.at-spi.gnome.org"
        # else:
        #     attributes_ns = "https://accessibility.windows.example.org/ns/attributes"
        #     state_ns = "https://accessibility.ubuntu.example.org/ns/state"
        #     component_ns = "https://accessibility.ubuntu.example.org/ns/component"
        #     value_ns = "https://accessibility.ubuntu.example.org/ns/value"

    def get_current_applications(self, obs):
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        apps = []
        root = tree.getroot()
        for item in root:
            apps.append(item.get("name", "").replace("\\", ""))
        return apps

    def check_new_apps(self, old_apps, new_apps):
        return new_apps - old_apps

    def find_active_applications(self, tree):
        # names of applications to keep TODO: soffice is a single application with all the isntances like impress, calc etc. being frames this will need to be dealt with separately
        to_keep = ["Program Manager"]
        apps_with_active_tag = []
        for application in list(tree.getroot()):
            app_name = application.get("name")
            for frame in application:
                is_active = frame.attrib.get("{{{:}}}active".format(state_ns), "false")
                if is_active == "true":
                    apps_with_active_tag.append(app_name)
        print(apps_with_active_tag)
        if apps_with_active_tag:
            to_keep.append(apps_with_active_tag[-1])
        return to_keep

    def filter_active_app(self, tree):
        for application in list(tree.getroot()):
            app_name = application.attrib.get("name")
            for frame in application:
                is_active = frame.attrib.get("{{{:}}}active".format(state_ns), "false")
                if is_active == "true":
                    return app_name
        return None

    def filter_nodes(self, tree, show_all=False):
        # created and populate a preserved nodes list which filters out unnecessary elements and keeps only those elements which are currently showing on the screen
        # TODO: include offscreen elements and then scroll to them before clicking
        preserved_nodes = []
        exclude_tags = ["panel", "window", "filler", "frame", "separator", "scroll-bar"]

        for node in tree.iter():
            if node.tag not in exclude_tags:
                if show_all:
                    if node.attrib.get(f"{{{state_ns}}}enabled") == "true":
                        coords: Tuple[int, int] = eval(
                            node.get(
                                "{{{:}}}screencoord".format(component_ns), "(-1, -1)"
                            )
                        )
                        if coords[0] >= 0 and coords[1] >= 0:
                            preserved_nodes.append(node)
                # if show_all is false, only show elements that are currently showing on screen
                else:
                    if node.attrib.get(f"{{{state_ns}}}visible") == "true":
                        coords: Tuple[int, int] = eval(
                            node.get(
                                "{{{:}}}screencoord".format(component_ns), "(-1, -1)"
                            )
                        )

                        if coords[0] >= 0 and coords[1] >= 0:
                            preserved_nodes.append(node)
        return preserved_nodes

    def linearize_tree(self, preserved_nodes):
        # TODO: Run an ablation to check if class and desc
        # linearized_accessibility_tree = ["id\ttag\tname\ttext\tclass\tdescription"]
        linearized_accessibility_tree = ["id\ttag\tname\ttext"]
        for idx, node in enumerate(preserved_nodes):
            if node.text:
                text = (
                    node.text
                    if '"' not in node.text
                    else '"{:}"'.format(node.text.replace('"', '""'))
                )
            else:
                text = '""'

            linearized_accessibility_tree.append(
                "{:}\t{:}\t{:}\t{:}".format(
                    idx,
                    node.tag,
                    node.get("name", ""),
                    text,
                    # node.get("{{{:}}}class".format(attributes_ns), ""),
                    # node.get("{{{:}}}description".format(attributes_ns), ""),
                )
            )

        # returning list of linearized elements
        return linearized_accessibility_tree

    def extract_elements_from_screenshot(self, screenshot) -> Dict:
        """Uses paddle-ocr to extract elements with text from the screenshot. The elements will be added to the linearized accessibility tree downstream"""

        # Convert screenshot to PIL image
        def send_image_to_ocr(screenshot) -> Dict:

            # url = os.environ.get("OCR_SERVER_ADDRESS", "")
            url = "http://127.0.0.1:8083/ocr/"
            if url == "":
                raise Exception("OCR SERVER ADDRESS NOT SET")
            encoded_screenshot = base64.b64encode(screenshot).decode("utf-8")
            data = {"img_bytes": encoded_screenshot}
            response = requests.post(url, json=data)

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Request failed with status code {response.status_code}",
                    "results": [],
                }

        return send_image_to_ocr(screenshot)["results"]

    def add_ocr_elements(
        self, screenshot, linearized_accessibility_tree, preserved_nodes
    ):
        # Get the bounding boxes of the elements in the linearized accessibility tree
        tree_bboxes = []
        for node in preserved_nodes:
            coordinates: Tuple[int, int] = eval(
                node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
            )
            sizes: Tuple[int, int] = eval(
                node.get("{{{:}}}size".format(component_ns), "(-1, -1)")
            )
            tree_bboxes.append(
                [
                    coordinates[0],
                    coordinates[1],
                    coordinates[0] + sizes[0],
                    coordinates[1] + sizes[1],
                ]
            )

        # Use OCR to found boxes that might be missing from the accessibility tree
        try:
            ocr_bboxes = self.extract_elements_from_screenshot(screenshot)
        except Exception as e:
            print(f"Error: {e}")
            ocr_bboxes = []
        else:
            # Check for intersection over union between the existing atree bounding boxes and the ocr bounding boxes, if ocr bounding boxes are new add them to the linearized accesibility tree
            if (
                len(ocr_bboxes) > 0
            ):  # Only check IOUs and add if there are any bounding boxes returned by the ocr module
                preserved_nodes_index = len(preserved_nodes)
                for ind, (i, content, box) in enumerate(ocr_bboxes):
                    # x1, y1, x2, y2 = int(box.get('left', 0)), int(box['top']), int(), int(box['bottom'])
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = (
                        int(box.get("left", 0)),
                        int(box.get("top", 0)),
                        int(box.get("right", 0)),
                        int(box.get("bottom", 0)),
                    )
                    iou = box_iou(
                        np.array(tree_bboxes, dtype=np.float32),
                        np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    ).flatten()

                    if max(iou) < 0.1:
                        # Add the element to the linearized accessibility tree
                        # TODO: ocr detected elements should be classified for their tag, currently set to push button for the agent to think they are interactable
                        linearized_accessibility_tree.append(
                            f"{preserved_nodes_index}\tpush-button\t\t{content}\t\t"
                        )

                        # add to preserved node with the component_ns prefix node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)"
                        node = ET.Element(
                            "ocr_node",
                            attrib={
                                "text": content,
                                "{{{}}}screencoord".format(
                                    component_ns
                                ): "({},{})".format(x1, y1),
                                "{{{}}}size".format(component_ns): "({},{})".format(
                                    x2 - x1, y2 - y1
                                ),
                            },
                        )
                        preserved_nodes.append(node)
                        preserved_nodes_index += 1

        return linearized_accessibility_tree, preserved_nodes

    def linearize_and_annotate_tree(self, obs, show_all=False):
        accessibility_tree = obs["accessibility_tree"]
        screenshot = obs["screenshot"]

        # convert the accessibility tree from a string representation to an xml tree
        tree = ET.ElementTree(ET.fromstring(accessibility_tree))

        # Get the applications to keep based on the active applications
        to_keep = self.find_active_applications(tree)
        self.top_app = to_keep[-1]

        # Remove applications which are not included in the to_keep list
        if not show_all:
            for application in list(tree.getroot()):
                if application.attrib.get("name", "") not in to_keep:
                    tree.getroot().remove(application)

        # Save tree for debugging
        # from datetime import datetime
        # with open(f"tree_raw_{datetime.now()}.xml", "wb") as file:
        #     tree.write(file, encoding="utf-8", xml_declaration=True)

        # Filter out filler elements and overlapping elements
        preserved_nodes = self.filter_nodes(tree, show_all)

        assert len(preserved_nodes) > 0

        # Linearize the tree as tsv
        linearized_accessibility_tree = self.linearize_tree(preserved_nodes)

        # Add OCR elements to the linearized accessibility tree to account for elements that are not in the accessibility tree
        if self.ocr:
            linearized_accessibility_tree, preserved_nodes = self.add_ocr_elements(
                screenshot, linearized_accessibility_tree, preserved_nodes
            )

        # Convert accessibility tree to a string
        linearized_accessibility_tree = "\n".join(linearized_accessibility_tree)

        # TODO: side-effect, set in separate functions
        self.nodes = preserved_nodes

        return linearized_accessibility_tree

    def find_element(self, element_id):
        try:
            selected_element = self.nodes[int(element_id)]
        except:
            print("The index of the selected element was out of range.")
            selected_element = self.nodes[0]
            self.index_out_of_range_flag = True
        return selected_element

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
        coordinates: Tuple[int, int] = eval(
            node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        )
        sizes: Tuple[int, int] = eval(
            node.get("{{{:}}}size".format(component_ns), "(-1, -1)")
        )

        # Calculate the center of the element
        x = coordinates[0] + sizes[0] // 2
        y = coordinates[1] + sizes[1] // 2

        command = "import pyautogui; "

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    @agent_action
    def switch_window(self):
        """Switch to a different application that is already open"""
        # return self.app_setup_code.replace("APP_NAME", app_code)
        return f"import pyautogui; pyautogui.hotkey('alt', 'tab');"

    @agent_action
    def type(
        self,
        text: str,
        element_id: int = None,
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into the element
        Args:
            text:str the text to type
            element_id:int ID of the element to type into. If not provided, typing will start at the current cursor location.
            overwrite:bool Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        try:
            # Use the provided element_id or default to None
            node = self.find_element(element_id) if element_id is not None else None
        except:
            node = None

        if node is not None:
            # If a node is found, retrieve its coordinates and size
            coordinates = eval(
                node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
            )
            sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

            # Calculate the center of the element
            x = coordinates[0] + sizes[0] // 2
            y = coordinates[1] + sizes[1] // 2

            # Start typing at the center of the element
            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # If no element is found, start typing at the current cursor location
            command = "import pyautogui; "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "

        return command

        # if overwrite:
        #     return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.hotkey("ctrl", "a"); pyautogui.press("backspace"); pyautogui.typewrite({repr(text)})"""
        # else:
        #     return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.hotkey("ctrl", "a"); pyautogui.press("backspace"); pyautogui.typewrite("{text}")"""

    # @agent_action
    # def type_and_enter(self, element_id:int, text:str, overwrite: bool = True):
    #     '''Type text into the element and press enter
    #     Args:
    #         element_id:int ID of the element to type into
    #         text:str the text to type into the element
    #     '''
    #     try:
    #         node = self.find_element(element_id)
    #     except:
    #         node = self.find_element(0)
    #     # print(node.attrib)
    #     coordinates = eval(
    #         node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)"))
    #     sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

    #     # Calculate the center of the element
    #     x = coordinates[0] + sizes[0] // 2
    #     y = coordinates[1] + sizes[1] // 2

    #     # Return pyautoguicode to type into the element
    #     if overwrite:
    #         return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.hotkey("ctrl", "a"); pyautogui.press("backspace"); pyautogui.typewrite({repr(text)}); pyautogui.press("enter")"""
    #     else:
    #         return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.typewrite({repr(text)}); pyautogui.press("enter")"""

    # @agent_action
    # def copy_text(self, element_id:int):
    #     '''Copy the selected text, use instead of ctrl+c
    #     Args:
    #         element_id:int ID of the element to copy text from
    #     '''
    #     try:
    #         node = self.find_element(element_id)
    #     except:
    #         node = self.find_element(0)

    #     self.clipboard = node.text

    # @agent_action
    # def paste_text(self, element_id:int, overwrite: bool = True):
    #     '''Paste text from the clipboard into the element, use instead of ctrl+v
    #     Args:
    #         element_id:int ID of the element to copy text from
    #         overwrite:bool a boolean value to determine if the text should be pasted over the existing text or appended to it
    #     '''
    #     try:
    #         node = self.find_element(element_id)
    #     except:
    #         node = self.find_element(0)

    #     coordinates = eval(
    #         node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)"))
    #     sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

    #     # Calculate the center of the element
    #     x = coordinates[0] + sizes[0] // 2
    #     y = coordinates[1] + sizes[1] // 2

    #     # Return pyautoguicode to paste into the element
    #     if overwrite:
    #         return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.typewrite("{self.clipboard}");"""
    #     else:
    #         return f"""import pyautogui; pyautogui.click({x}, {y}); pyautogui.hotkey("ctrl", "a"); pyautogui.press("backspace"); pyautogui.typewrite("{self.clipboard}");"""

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save facts, elements, texts, etc. to a long-term knowledge bank for reuse during this task. Can be used for copy-pasting text, saving elements, etc.
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
        coordinates1 = eval(
            node1.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        )
        sizes1 = eval(node1.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

        coordinates2 = eval(
            node2.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        )
        sizes2 = eval(node2.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

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
        """Scroll the element in the specified direction
        Args:
            element_id:int ID of the element to scroll in
            clicks:int the number of clicks to scroll can be positive (up) or negative (down).
        """
        try:
            node = self.find_element(element_id)
        except:
            node = self.find_element(0)
        # print(node.attrib)
        coordinates = eval(
            node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        )
        sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

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
            keys:List the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
        """
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """

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
