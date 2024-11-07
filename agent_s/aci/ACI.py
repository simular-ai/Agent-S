from typing import Dict, List, Tuple, Any
import logging
import os
from .OCRHelper import OCRHelper

logger = logging.getLogger("desktopenv.agent")

def agent_action(func):
    func.is_agent_action = True
    return func



class ACI:
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        self.top_app_only = top_app_only
        self.ocr = ocr
        self.index_out_of_range_flag = False
        self.notes: List[str] = []
        self.clipboard = ""
        self.nodes: List[Any] = []

    def get_active_apps(self, obs: Dict) -> List[str]:
        pass 

    def preserve_nodes(self, tree: Any, exclude_roles: set = None) -> List[Dict]:
        pass

    def linearize_and_annotate_tree(self, obs: Dict, show_all_elements: bool = False) -> str:
        screenshot = obs["screenshot"]
        
        # Handle different tree formats based on environment
        if self.is_osworld:
            # OSWorld XML string
            tree = self.UIElement.nodeFromTree(obs["accessibility_tree"])
        else:
            # Direct accessibility object
            tree = obs["accessibility_tree"]
            
        if not tree:
            logger.error("Failed to get accessibility tree")
            return "id\trole\ttitle\ttext"

        # TODO: write this function for each UIElement type 
        self.top_app = self.UIElement.get_top_app(obs)
        # self.top_app = UIElement.get_current_applications(obs: Dict)[0] if UIElement.get_current_applications(obs: Dict) else None
        
        preserved_nodes = self.preserve_nodes(tree)
        
        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        if self.ocr:
            tree_elements, preserved_nodes = OCRHelper.add_ocr_elements(
                screenshot, tree_elements, preserved_nodes
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
    def open_app_or_file(self, app_or_file_name):
        """Open an application
        Args:
            app_or_file_name:str, the name of the application or file to open
        """
        if self.platform == 'macos':
            return f"import pyautogui; pyautogui.hotkey('command', 'space', interval=1); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter')"
        elif self.platform == "ubuntu":
            return f"import pyautogui; pyautogui.hotkey('win', interval=1); pyautogui.typewrite({repr(app_or_file_name)}); pyautogui.press('enter')"

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
            enter:bool Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
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
                if self.platform == "macos":
                    command += f"pyautogui.hotkey('command', 'a', interval=1); pyautogui.press('backspace'); "
                else:
                    command += f"pyautogui.hotkey('ctrl', 'a', interval=1); pyautogui.press('backspace'); "

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # If no element is found, start typing at the current cursor location
            command = "import pyautogui; "

            if overwrite:
                # Use 'command' instead of 'cmd'
                if self.platform == "macos":
                    command += f"pyautogui.hotkey('command', 'a', interval=1); pyautogui.press('backspace'); "
                else:
                    command += f"pyautogui.hotkey('ctrl', 'a', interval=1); pyautogui.press('backspace'); "

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
