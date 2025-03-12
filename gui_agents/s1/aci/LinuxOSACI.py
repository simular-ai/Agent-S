import base64
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
import requests

from gui_agents.s1.aci.ACI import ACI
from gui_agents.s1.utils.common_utils import box_iou

import platform

if platform.system() == "Linux":
    import pyatspi
    from pyatspi import Accessible, StateType, STATE_SHOWING
    from pyatspi import Action as ATAction
    from pyatspi import Component  # , Document
    from pyatspi import Text as ATText
    from pyatspi import Value as ATValue

    from pyatspi import Accessible, StateType
    from lxml.etree import _Element
    from typing import Optional, Dict, Any, List

    import lxml.etree
    import concurrent.futures

_accessibility_ns_map_ubuntu = {
    "st": "https://accessibility.ubuntu.example.org/ns/state",
    "attr": "https://accessibility.ubuntu.example.org/ns/attributes",
    "cp": "https://accessibility.ubuntu.example.org/ns/component",
    "doc": "https://accessibility.ubuntu.example.org/ns/document",
    "docattr": "https://accessibility.ubuntu.example.org/ns/document/attributes",
    "txt": "https://accessibility.ubuntu.example.org/ns/text",
    "val": "https://accessibility.ubuntu.example.org/ns/value",
    "act": "https://accessibility.ubuntu.example.org/ns/action",
}

MAX_DEPTH = 50
MAX_WIDTH = 1024

logger = logging.getLogger("desktopenv.agent")


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


class LinuxACI(ACI):
    def __init__(self, top_app=None, vm_version="new", top_app_only=True, ocr=True):
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
        global state_ns, component_ns, attributes_ns, value_ns
        if vm_version == "old":

            state_ns = "uri:deskat:state.at-spi.gnome.org"
            component_ns = "uri:deskat:component.at-spi.gnome.org"
        else:
            attributes_ns = "https://accessibility.windows.example.org/ns/attributes"
            state_ns = "https://accessibility.ubuntu.example.org/ns/state"
            component_ns = "https://accessibility.ubuntu.example.org/ns/component"
            value_ns = "https://accessibility.ubuntu.example.org/ns/value"

    def get_active_apps(self, obs: Dict) -> List[str]:
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        apps = []
        exclude_list = ["gjs", "gnome-shell"]
        for node in tree.iter():
            # Keep applications and only those which have children
            if (
                node.tag.endswith("application")
                and list(node)
                and node.attrib.get("name", "") not in exclude_list
            ):
                apps.append(node.attrib.get("name", "").replace("\\", ""))
        return apps

    def check_new_apps(self, old_apps, new_apps):
        return new_apps - old_apps

    def get_top_app(self, obs):
        return self.top_app

    def find_active_applications(self, tree):
        # names of applications to keep TODO: soffice is a single application with all the isntances like impress, calc etc. being frames this will need to be dealt with separately
        to_keep = ["gnome-shell"]
        apps_with_active_tag = []
        for application in list(tree.getroot()):
            app_name = application.attrib.get("name")
            for frame in application:
                is_active = frame.attrib.get("{{{:}}}active".format(state_ns), "false")
                if is_active == "true":
                    apps_with_active_tag.append(app_name)
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
                    if node.attrib.get(f"{{{state_ns}}}visible") == "true":
                        coords: Tuple[int, int] = eval(
                            node.get(
                                "{{{:}}}screencoord".format(component_ns), "(-1, -1)"
                            )
                        )
                        if coords[0] >= 0 and coords[1] >= 0:
                            preserved_nodes.append(node)
                # if show_all is false, only show elements that are currently showing on screen
                else:
                    if node.attrib.get(f"{{{state_ns}}}showing") == "true":
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

            url = os.environ.get("OCR_SERVER_ADDRESS", "")
            if url == "":
                raise Exception("OCR SERVER ADDRESS NOT SET")
            encoded_screenshot = base64.b64encode(screenshot).decode("utf-8")
            data = {"img_bytes": encoded_screenshot}
            print("Getting OCR response")
            ocr_start = time.time()
            response = requests.post(url, json=data)
            print("Got OCR response in", time.time() - ocr_start)

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
        with open("tree_raw.xml", "wb") as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)

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
    def switch_applications(self, app_code):
        """Switch to a different application that is already open
        Args:
            app_code:str the code name of the application to switch to from the provided list of open applications
        """
        return self.app_setup_code.replace("APP_NAME", app_code)

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


def _create_atspi_node(
    node: Accessible, depth: int = 0, flag: Optional[str] = None
) -> _Element:
    node_name = node.name
    attribute_dict: Dict[str, Any] = {"name": node_name}

    #  States
    states: List[StateType] = node.getState().get_states()
    for st in states:
        state_name: str = StateType._enum_lookup[st]
        state_name: str = state_name.split("_", maxsplit=1)[1].lower()
        if len(state_name) == 0:
            continue
        attribute_dict[
            "{{{:}}}{:}".format(_accessibility_ns_map_ubuntu["st"], state_name)
        ] = "true"

    #  Attributes
    attributes: Dict[str, str] = node.get_attributes()
    for attribute_name, attribute_value in attributes.items():
        if len(attribute_name) == 0:
            continue
        attribute_dict[
            "{{{:}}}{:}".format(_accessibility_ns_map_ubuntu["attr"], attribute_name)
        ] = attribute_value

    #  Component
    if (
        attribute_dict.get(
            "{{{:}}}visible".format(_accessibility_ns_map_ubuntu["st"]), "false"
        )
        == "true"
        and attribute_dict.get(
            "{{{:}}}showing".format(_accessibility_ns_map_ubuntu["st"]), "false"
        )
        == "true"
    ):
        try:
            component: Component = node.queryComponent()
        except NotImplementedError:
            pass
        else:
            bbox: Sequence[int] = component.getExtents(pyatspi.XY_SCREEN)
            attribute_dict[
                "{{{:}}}screencoord".format(_accessibility_ns_map_ubuntu["cp"])
            ] = str(tuple(bbox[0:2]))
            attribute_dict["{{{:}}}size".format(_accessibility_ns_map_ubuntu["cp"])] = (
                str(tuple(bbox[2:]))
            )

    text = ""
    #  Text
    try:
        text_obj: ATText = node.queryText()
        # only text shown on current screen is available
        # attribute_dict["txt:text"] = text_obj.getText(0, text_obj.characterCount)
        text: str = text_obj.getText(0, text_obj.characterCount)
        # if flag=="thunderbird":
        # appeared in thunderbird (uFFFC) (not only in thunderbird), "Object
        # Replacement Character" in Unicode, "used as placeholder in text for
        # an otherwise unspecified object; uFFFD is another "Replacement
        # Character", just in case
        text = text.replace("\ufffc", "").replace("\ufffd", "")
    except NotImplementedError:
        pass

    #  Image, Selection, Value, Action
    try:
        node.queryImage()
        attribute_dict["image"] = "true"
    except NotImplementedError:
        pass

    try:
        node.querySelection()
        attribute_dict["selection"] = "true"
    except NotImplementedError:
        pass

    try:
        value: ATValue = node.queryValue()
        value_key = f"{{{_accessibility_ns_map_ubuntu['val']}}}"

        for attr_name, attr_func in [
            ("value", lambda: value.currentValue),
            ("min", lambda: value.minimumValue),
            ("max", lambda: value.maximumValue),
            ("step", lambda: value.minimumIncrement),
        ]:
            try:
                attribute_dict[f"{value_key}{attr_name}"] = str(attr_func())
            except:
                pass
    except NotImplementedError:
        pass

    try:
        action: ATAction = node.queryAction()
        for i in range(action.nActions):
            action_name: str = action.getName(i).replace(" ", "-")
            attribute_dict[
                "{{{:}}}{:}_desc".format(
                    _accessibility_ns_map_ubuntu["act"], action_name
                )
            ] = action.getDescription(i)
            attribute_dict[
                "{{{:}}}{:}_kb".format(_accessibility_ns_map_ubuntu["act"], action_name)
            ] = action.getKeyBinding(i)
    except NotImplementedError:
        pass

    # Add from here if we need more attributes in the future...

    raw_role_name: str = node.getRoleName().strip()
    node_role_name = (raw_role_name or "unknown").replace(" ", "-")

    if not flag:
        if raw_role_name == "document spreadsheet":
            flag = "calc"
        if raw_role_name == "application" and node.name == "Thunderbird":
            flag = "thunderbird"

    xml_node = lxml.etree.Element(
        node_role_name, attrib=attribute_dict, nsmap=_accessibility_ns_map_ubuntu
    )

    if len(text) > 0:
        xml_node.text = text

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached")
        return xml_node

    if flag == "calc" and node_role_name == "table":
        # Maximum column: 1024 if ver<=7.3 else 16384
        # Maximum row: 104 8576
        # Maximun sheet: 1 0000

        global libreoffice_version_tuple
        MAXIMUN_COLUMN = 1024 if libreoffice_version_tuple < (7, 4) else 16384
        MAX_ROW = 104_8576

        index_base = 0
        first_showing = False
        column_base = None
        for r in range(MAX_ROW):
            for clm in range(column_base or 0, MAXIMUN_COLUMN):
                child_node: Accessible = node[index_base + clm]
                showing: bool = child_node.getState().contains(STATE_SHOWING)
                if showing:
                    child_node: _Element = _create_atspi_node(
                        child_node, depth + 1, flag
                    )
                    if not first_showing:
                        column_base = clm
                        first_showing = True
                    xml_node.append(child_node)
                elif first_showing and column_base is not None or clm >= 500:
                    break
            if first_showing and clm == column_base or not first_showing and r >= 500:
                break
            index_base += MAXIMUN_COLUMN
        return xml_node
    else:
        try:
            for i, ch in enumerate(node):
                if i == MAX_WIDTH:
                    logger.warning("Max width reached")
                    break
                xml_node.append(_create_atspi_node(ch, depth + 1, flag))
        except:
            logger.warning(
                "Error occurred during children traversing. Has Ignored. Node: %s",
                lxml.etree.tostring(xml_node, encoding="unicode"),
            )
        return xml_node


class UIElement(object):
    def __init__(self, node):
        self.node = node

    def getAttributeNames(self):
        attributes = self.node.getAttributes()

    @staticmethod
    def systemWideElement():
        # desktop = pyatspi.Registry.getDesktop(0)
        # for app in desktop:
        #     for window in app:
        #         if window.getState().contains(pyatspi.STATE_ACTIVE):
        #             active_node = app
        # return UIElement(active_node)
        desktop: Accessible = pyatspi.Registry.getDesktop(0)
        xml_node = lxml.etree.Element(
            "desktop-frame", nsmap=_accessibility_ns_map_ubuntu
        )
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_create_atspi_node, app_node, 1) for app_node in desktop
            ]
            for future in concurrent.futures.as_completed(futures):
                xml_tree = future.result()
                xml_node.append(xml_tree)
        return lxml.etree.tostring(xml_node, encoding="unicode")

    @property
    def states(self):
        state_names = []
        states: List[StateType] = self.node.getState().get_states()
        for st in states:
            state_name: str = StateType._enum_lookup[st]
            state_names.append(state_name)
        return state_names

    @property
    def attributes(self):
        try:
            attributes: List[str] = self.node.getAttributes()
            attribute_dict = {}
            for attrbt in attributes:
                attribute_name: str
                attribute_value: str
                attribute_name, attribute_value = attrbt.split(":", maxsplit=1)
                attribute_dict[attribute_name] = attribute_value
            return attribute_dict
        except NotImplementedError:
            return None

    @property
    def component(self):
        try:
            component: Component = self.node.queryComponent()
            return component
        except NotImplementedError:
            return None

    @property
    def value(self):
        try:
            value: ATValue = self.node.queryValue()
            return value
        except NotImplementedError:
            return None

    @property
    def text(self):
        try:
            text_obj: ATText = self.node.queryText()
        except NotImplementedError:
            return ""
        else:
            text: str = text_obj.getText(0, text_obj.characterCount)
            text = text.replace("\ufffc", "").replace("\ufffd", "")
            return text

    @property
    def role(self):
        return self.node.getRoleName()

    def children(self):
        """Return list of children of the current node"""
        return list(self.node)

    def __repr__(self):
        return "UIElement%s" % (self.node)
