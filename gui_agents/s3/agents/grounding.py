import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output

from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.utils.common_utils import call_llm_safe
from gui_agents.s3.agents.code_agent import CodeAgent
import logging

logger = logging.getLogger("desktopenv.agent")


class ACI:
    def __init__(self):
        self.notes: List[str] = []


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


UBUNTU_APP_SETUP = f"""import subprocess;
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


SET_CELL_VALUES_CMD = """import uno
import subprocess
import unicodedata, json

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def _norm_name(s: str | None) -> str | None:
    if s is None:
        return None
    if "\\\\u" in s or "\\\\U" in s or "\\\\x" in s:
        try:
            # json.loads handles all the escape forms safely
            s = json.loads(f"{{s}}")
        except Exception:
            # fallback: best-effort
            try:
                s = s.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
    # Normalize (NFC works well across platforms)
    return unicodedata.normalize("NFC", s)

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    app_name  = _norm_name(app_name)
    sheet_name = _norm_name(sheet_name)

    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"osworld-public-evaluation\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""


# ACI primitives are parameterized by description, and coordinate generation uses a pretrained grounding model
class OSWorldACI(ACI):
    def __init__(
        self,
        env,
        platform: str,
        engine_params_for_generation: Dict,
        engine_params_for_grounding: Dict,
        width: int = 1920,
        height: int = 1080,
        code_agent_budget: int = 20,
        code_agent_engine_params: Dict = None,
    ):
        super().__init__()

        self.env = env
        self.platform = (
            platform  # Dictates how the switch_applications agent action works.
        )

        # Configure scaling
        self.width = width
        self.height = height

        # Maintain state for save_to_knowledge
        self.notes = []

        # Screenshot used during ACI execution
        self.obs = None

        # Configure the visual grounding model responsible for coordinate generation
        self.grounding_model = LMMAgent(engine_params_for_grounding)
        self.engine_params_for_grounding = engine_params_for_grounding

        # Configure text grounding agent
        self.text_span_agent = LMMAgent(
            engine_params=engine_params_for_generation,
            system_prompt=PROCEDURAL_MEMORY.PHRASE_TO_WORD_COORDS_PROMPT,
        )

        # Configure code agent
        code_agent_engine_params = (
            code_agent_engine_params or engine_params_for_generation
        )
        self.code_agent = CodeAgent(code_agent_engine_params, code_agent_budget)

        # Store task instruction for code agent
        self.current_task_instruction = None
        self.last_code_agent_result = None

    # Given the state and worker's referring expression, use the grounding model to generate (x,y)
    def generate_coords(self, ref_expr: str, obs: Dict) -> List[int]:

        # Reset the grounding model state
        self.grounding_model.reset()

        # Configure the context, UI-TARS demo does not use system prompt
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )

        # Generate and parse coordinates
        response = call_llm_safe(self.grounding_model)
        print("RAW GROUNDING MODEL RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        assert len(numericals) >= 2
        return [int(numericals[0]), int(numericals[1])]

    # Calls pytesseract to generate word level bounding boxes for text grounding
    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        image = Image.open(BytesIO(b64_image_data))
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Clean text by removing leading and trailing spaces and non-alphabetical characters, but keeping punctuation
        for i, word in enumerate(image_data["text"]):
            image_data["text"][i] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word
            )

        ocr_elements = []
        ocr_table = "Text Table:\nWord id\tText\n"
        # Obtain the <id, text, group number, word number> for each valid element
        grouping_map = defaultdict(list)
        ocr_id = 0
        for i in range(len(image_data["text"])):
            block_num = image_data["block_num"][i]
            if image_data["text"][i]:
                grouping_map[block_num].append(image_data["text"][i])
                ocr_table += f"{ocr_id}\t{image_data['text'][i]}\n"
                ocr_elements.append(
                    {
                        "id": ocr_id,
                        "text": image_data["text"][i],
                        "group_num": block_num,
                        "word_num": len(grouping_map[block_num]),
                        "left": image_data["left"][i],
                        "top": image_data["top"][i],
                        "width": image_data["width"][i],
                        "height": image_data["height"][i],
                    }
                )
                ocr_id += 1

        return ocr_table, ocr_elements

    # Given the state and worker's text phrase, generate the coords of the first/last word in the phrase
    def generate_text_coords(
        self, phrase: str, obs: Dict, alignment: str = ""
    ) -> List[int]:

        ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])

        alignment_prompt = ""
        if alignment == "start":
            alignment_prompt = "**Important**: Output the word id of the FIRST word in the provided phrase.\n"
        elif alignment == "end":
            alignment_prompt = "**Important**: Output the word id of the LAST word in the provided phrase.\n"

        # Load LLM prompt
        self.text_span_agent.reset()
        self.text_span_agent.add_message(
            alignment_prompt + "Phrase: " + phrase + "\n" + ocr_table, role="user"
        )
        self.text_span_agent.add_message(
            "Screenshot:\n", image_content=obs["screenshot"], role="user"
        )

        # Obtain the target element
        response = call_llm_safe(self.text_span_agent)
        print("TEXT SPAN AGENT RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        if len(numericals) > 0:
            text_id = int(numericals[-1])
        else:
            text_id = 0
        elem = ocr_elements[text_id]

        # Compute the element coordinates
        if alignment == "start":
            coords = [elem["left"], elem["top"] + (elem["height"] // 2)]
        elif alignment == "end":
            coords = [elem["left"] + elem["width"], elem["top"] + (elem["height"] // 2)]
        else:
            coords = [
                elem["left"] + (elem["width"] // 2),
                elem["top"] + (elem["height"] // 2),
            ]
        return coords

    def assign_screenshot(self, obs: Dict):
        self.obs = obs

    def set_task_instruction(self, task_instruction: str):
        """Set the current task instruction for the code agent."""
        self.current_task_instruction = task_instruction

    # Resize from grounding model dim into OSWorld dim (1920 * 1080)
    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        grounding_width = self.engine_params_for_grounding["grounding_width"]
        grounding_height = self.engine_params_for_grounding["grounding_height"]

        return [
            round(coordinates[0] * self.width / grounding_width),
            round(coordinates[1] * self.height / grounding_height),
        ]

    @agent_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            element_description:str, a detailed descriptions of which element to click on. This description should be at least a full sentence.
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        coords1 = self.generate_coords(element_description, self.obs)
        x, y = self.resize_coordinates(coords1)
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
        if self.platform == "darwin":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "linux":
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_code)
        elif self.platform == "windows":
            return f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        else:
            assert (
                False
            ), f"Unsupported platform: {self.platform}. Supported platforms are: darwin, linux, windows."

    @agent_action
    def open(self, app_or_filename: str):
        """Open any application or file with name app_or_filename. Use this action to open applications or files on the desktop, do not open manually.
        Args:
            app_or_filename:str, the name of the application or filename to open
        """
        if self.platform == "linux":
            return f"import pyautogui; pyautogui.hotkey('win'); time.sleep(0.5); pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); pyautogui.hotkey('enter'); time.sleep(0.5)"
        elif self.platform == "darwin":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_or_filename)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "windows":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('win'); time.sleep(0.5); "
                f"pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); "
                "pyautogui.press('enter'); time.sleep(0.5)"
            )
        else:
            assert (
                False
            ), f"Unsupported platform: {self.platform}. Supported platforms are: darwin, linux, windows."

    @agent_action
    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text/unicode into a specific element
        Args:
            element_description:str, a detailed description of which element to enter text in. This description should be at least a full sentence.
            text:str, the text to type
            overwrite:bool, Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool, Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        command = "import pyautogui; "
        command += (
            "\ntry:\n"
            "    import pyperclip\n"
            "except ImportError:\n"
            "    import subprocess\n"
            "    subprocess.run('echo \"osworld-public-evaluation\" | sudo -S apt-get install -y xclip xsel', shell=True, check=True)\n"
            "    subprocess.check_call([subprocess.sys.executable, '-m', 'pip', 'install', 'pyperclip'])\n"
            "    import pyperclip\n\n"
        )

        if element_description is not None:
            coords1 = self.generate_coords(element_description, self.obs)
            x, y = self.resize_coordinates(coords1)
            command += f"pyautogui.click({x}, {y}); "

        if overwrite:
            command += (
                f"pyautogui.hotkey({repr('command' if self.platform == 'darwin' else 'ctrl')}, 'a'); "
                "pyautogui.press('backspace'); "
            )

        # Check if text contains Unicode characters that pyautogui.write() can't handle
        has_unicode = any(ord(char) > 127 for char in text)

        if has_unicode:
            # Use clipboard method for Unicode characters
            command += f"pyperclip.copy({repr(text)}); "
            command += f"pyautogui.hotkey({repr('command' if self.platform == 'darwin' else 'ctrl')}, 'v'); "
        else:
            # Use regular pyautogui.write() for ASCII text
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
    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """Drag from the starting description to the ending description
        Args:
            starting_description:str, a very detailed description of where to start the drag action. This description should be at least a full sentence.
            ending_description:str, a very detailed description of where to end the drag action. This description should be at least a full sentence.
            hold_keys:List list of keys to hold while dragging
        """
        coords1 = self.generate_coords(starting_description, self.obs)
        coords2 = self.generate_coords(ending_description, self.obs)
        x1, y1 = self.resize_coordinates(coords1)
        x2, y2 = self.resize_coordinates(coords2)

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='left'); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    @agent_action
    def highlight_text_span(
        self, starting_phrase: str, ending_phrase: str, button: str = "left"
    ):
        """Highlight a text span between a provided starting phrase and ending phrase. Use this to highlight words, lines, and paragraphs.
        Args:
            starting_phrase:str, the phrase that denotes the start of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
            ending_phrase:str, the phrase that denotes the end of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
            button:str, the button to use to highlight the text span. Defaults to "left". Can be "left", "right", or "middle".
        """
        coords1 = self.generate_text_coords(
            starting_phrase, self.obs, alignment="start"
        )
        coords2 = self.generate_text_coords(ending_phrase, self.obs, alignment="end")
        x1, y1 = coords1
        x2, y2 = coords2

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='{button}'); pyautogui.mouseUp(); "

        # Return pyautoguicode to drag and drop the elements
        return command

    @agent_action
    def set_cell_values(
        self, cell_values: Dict[str, Any], app_name: str, sheet_name: str
    ):
        """Use this to set individual cell values in a spreadsheet. For example, setting A2 to "hello" would be done by passing {"A2": "hello"} as cell_values. The sheet must be opened before this command can be used.
        Args:
            cell_values: Dict[str, Any], A dictionary of cell values to set in the spreadsheet. The keys are the cell coordinates in the format "A1", "B2", etc.
                Supported value types include: float, int, string, bool, formulas.
            app_name: str, The name of the spreadsheet application. For example, "Some_sheet.xlsx".
            sheet_name: str, The name of the sheet in the spreadsheet. For example, "Sheet1".
        """
        return SET_CELL_VALUES_CMD.format(
            cell_values=cell_values, app_name=app_name, sheet_name=sheet_name
        )

    @agent_action
    def call_code_agent(self, task: str = None):
        """Call the code agent to execute code for tasks or subtasks that can be completed solely with coding.

        Args:
            task: str, the task or subtask to execute. If None, uses the current full task instruction.

        **🚨 CRITICAL GUIDELINES:**
        - **ONLY pass a task parameter for SPECIFIC subtasks** (e.g., "Calculate sum of column B", "Filter data by date")
        - **NEVER pass a task parameter for full tasks** - let it default to the original task instruction
        - **NEVER rephrase or modify the original task** - this prevents hallucination corruption
        - **If unsure, omit the task parameter entirely** to use the original task instruction

        Use this for tasks that can be fully accomplished through code execution, particularly for:
        - Spreadsheet applications (LibreOffice Calc, Excel): data processing, filtering, sorting, calculations, formulas, data analysis
        - Document editors (LibreOffice Writer, Word): text processing, content editing, formatting, document manipulation
        - Code editors (VS Code, text editors): code editing, file processing, text manipulation, configuration
        - Data analysis tools: statistical analysis, data transformation, reporting
        - File management: bulk operations, file processing, content extraction
        - System utilities: configuration, setup, automation
        """
        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Code Agent")
        logger.info("=" * 50)

        # **CRITICAL**: Only use provided task for specific subtasks, otherwise use original task instruction
        if task is not None:
            # This is a subtask - use the provided task
            task_to_execute = task
            logger.info(f"Executing SUBTASK: {task_to_execute}")
        else:
            # This is a full task - use the original task instruction to prevent hallucination
            task_to_execute = self.current_task_instruction
            logger.info(f"Executing FULL TASK: {task_to_execute}")

        if task_to_execute:
            print("obs keys: ", self.obs.keys())
            screenshot = self.obs.get("screenshot", "") if self.obs else ""
            logger.info(f"Screenshot available: {'Yes' if screenshot else 'No'}")

            logger.info("Executing code agent...")
            result = self.code_agent.execute(
                task_to_execute, screenshot, self.env.controller
            )

            # Store the result for the worker to access
            self.last_code_agent_result = result

            logger.info("Code agent execution completed")
            logger.info(f"Result - Completion reason: {result['completion_reason']}")
            logger.info(f"Steps executed: {result['steps_executed']}")
            logger.info(f"Summary: {result['summary']}")

            logger.info("=" * 50)
            logger.info("GROUNDING AGENT: Code Agent Call Finished")
            logger.info("=" * 50)

            # Return code to be executed in the environment
            return "import time; time.sleep(2.222)"
        else:
            logger.warning("No task instruction available for code agent call")
            return "import time; time.sleep(1.111)"

    @agent_action
    def scroll(self, element_description: str, clicks: int, shift: bool = False):
        """Scroll the element in the specified direction
        Args:
            element_description:str, a very detailed description of which element to enter scroll in. This description should be at least a full sentence.
            clicks:int, the number of clicks to scroll can be positive (up) or negative (down).
            shift:bool, whether to use shift+scroll for horizontal scrolling
        """
        coords1 = self.generate_coords(element_description, self.obs)
        x, y = self.resize_coordinates(coords1)

        if shift:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.hscroll({clicks})"
        else:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.vscroll({clicks})"

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
    def done(
        self,
    ):
        """End the current task with a success. Use this when you believe the entire task has been fully completed."""
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure. Use this when you believe the entire task is impossible to complete."""
        return """FAIL"""
