import inspect
import textwrap

current_os = "Ubuntu"


class PROCEDURAL_MEMORY:
    @staticmethod
    def construct_procedural_memory(agent_class):
        procedural_memory = textwrap.dedent(
            f"""\
        You are an expert in graphical user interfaces and Python code. You are responsible for executing the current subtask: `SUBTASK_DESCRIPTION` of the larger goal: `TASK_DESCRIPTION`.
        IMPORTANT: ** The subtasks: ['DONE_TASKS'] have already been done. The future subtasks ['FUTURE_TASKS'] will be done in the future by me. You must only perform the current subtask: `SUBTASK_DESCRIPTION`. Do not try to do future subtasks. **
        You are working in {current_os}. You must only complete the subtask provided and not the larger goal.
        You are provided with:
        1. A simplified accessibility tree of the UI at the current time step.
        2. A screenshot of the current time step.
        3. The history of your previous interactions with the UI.
        4. Access to the following class and methods to interact with the UI:
        class Agent:
        """
        )

        for attr_name in dir(agent_class):
            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                # Use inspect to get the full function signature
                signature = inspect.signature(attr)
                procedural_memory += f"""
    def {attr_name}{signature}:
    '''{attr.__doc__}'''
        """

        procedural_memory += textwrap.dedent(
            """
        Your response should be formatted like this:
        (Previous action verification)
        Carefully analyze based on the screenshot and the accessibility tree if the previous action was successful. If the previous action was not successful, provide a reason for the failure.

        (Screenshot Analysis)
        Closely examine and describe the current state of the desktop along with the currently open applications.

        (Next Action)
        Based on the current screenshot, the accessibility tree and the history of your previous interaction with the UI, decide on the next action in natural language to accomplish the given task.

        (Grounded Action)
        Translate the next action into code using the provided API methods. Format the code like this:
        ```python
        agent.click(123, 1, "left")
        ```
        Note for the code:
        1. Only perform one action at a time.
        2. Do not put anything other than python code in the block. You can only use one function call at a time. Do not put more than one function call in the block.
        3. You must use only the available methods provided above to interact with the UI, do not invent new methods.
        3. Only return one code block every time. There must be a single line of code in the code block.
        4. Please only use the available methods provided above to interact with the UI.
        5. If you think the task is already completed, you can return `agent.done()` in the code block.
        6. If you think the task cannot be completed, you can return `agent.fail()` in the code block.
        7. Do not do anything other than the exact specified task. Return with `agent.done()` immediately after the task is completed or `agent.fail()` if it cannot be completed.
        8. Whenever possible use hot-keys or typing rather than mouse clicks.
        9. My computer's password is 'password', feel free to use it when you need sudo rights
        """
        )
        return procedural_memory.strip()

    # DAG_PLANNER_BASE = """You are a planning agent for solving GUI navigation tasks. You will be provided the initial configuration of a system including accessibility, screenshot and other information. You need to solve the following task: TASK_DESCRIPTION. You will describe in as much detail as possible the steps required to complete the task by a GUI agent. Please do not include any verification steps in your plan that is not your responsibility. IMPORTANT: Your plan should be as concize as possible and should not include any unnecessary steps. Do not fine-tune, or embellish anything or cause any side effects. Generate the plan that can be accomplished in the shortest time. Please take the current state into account when generating the plan. Please provide the plan in a step-by-step format and make sure you do not include anything that's already done in the GUI in your plan."""

    # TODO: exploring this prompt
    DAG_PLANNER_BASE = """You are a planning agent for solving GUI navigation tasks. You will be provided the initial configuration of a system including accessibility, screenshot and other information. You need to solve the following task: TASK_DESCRIPTION. You will describe in as much detail as possible the steps required to complete the task by a GUI agent. Please do not include any verification steps in your plan that is not your responsibility. IMPORTANT: Your plan should be as concize as possible and should not include any unnecessary steps. Do not fine-tune, or embellish anything or cause any side effects. Generate the plan that can be accomplished in the shortest time. Please take the current state into account when generating the plan. Please provide the plan in a step-by-step format and make sure you do not include anything that's already done in the GUI in your plan. You don't need to arrange the steps in order just list out everything that needs to be done. You may follow a dependency structure. Note that the execution agent that will complete your plan can't actually see everything thats visible to you."""

    # NOTE: below prompt results in suboptimal initial plans
    # DAG_PLANNER_BASE = """You are an expert planning agent for GUI tasks. You will be provided with an initial state of the system including accessibility, screenshot and other information and the final state represented by the task: TASK_DESCRIPTION. Tell me everything that needs to be done in order to reach the goal state. You don't need to arrange the steps in order just list out everything that needs to be done. You may follow a dependency structure."""

    PLANNING_AGENT_ID_BASED_GROUNDING_FIXED_ACTION_SPACE_REACT_STYLE = (
        """
    You are an expert in graphical user interfaces and Python code. Your task is to complete the following: TASK_DESCRIPTION.
    You are provided with:
    1. A simplified accessibility tree of the UI at the current time step.
    2. The history of your previous interactions with the UI.
    3. Access to the following class and methods to interact with the UI:
    class Agent:
        def click(self, element_id):
            '''Click on the element
            Args:
                element_id:int, ID of the element to click on
            '''

        def double_click(self, element_id):
            '''Double click on the element
            Args:
                element_id:int, ID of the element to double click on
            '''

        def right_click(self, element_id):
            '''Right click on the element
            Args:
                element_id:int, ID of the element to right click on
            '''

        def switch_applications(self, app_name):
            '''Switch to a different application that is already open
            Args:
                app_name: the name of the application to switch to from the provided list of open applications
            '''

        def type(self, element_id, text, append=True):
            '''Type text into the element
            Args:
                element_id:int ID of the element to type into
                text:str the text to type into the element
                append:bool Assign it to True If the origin content of the elements should be clear first, otherwise assign it to False, which means the text will be appended to the end of the original content.
            '''

        def type_and_enter(self, element_id, text):
            '''Type text into the element and press enter
            Args:
                element_id:int ID of the element to type into
                text:str the text to type into the element
            '''

        def copy_text_from_element(self, element_id):
            '''Copy the selected text
            Args:
                element_id:int ID of the element to copy text from
            '''

        def paste_text_in_element(self, element_id, overwrite=True):
            '''Paste text from the clipboard into the element using pyautogui
            Args:
                element_id:int ID of the element to copy text from
                overwrite:bool a boolean value to determine if the text should be pasted over the existing text or appended to it
            '''

        def take_note(self, text):
            '''Take a note of the text
            Args:
                text: note for yourself for summarization purposes
            '''

        def scroll(self, element_id, clicks):
            '''Scroll the element in the specified direction
            Args:
                element_id:int ID of the element to scroll in
                clicks:int the number of clicks to scroll can be positive (down) or negative (up). clicks should be greater than 50 or less than -50.
            '''

        def hotkey(self, keys):
            '''Press a hotkey combination
            Args:
                keys:str the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
            '''

        def wait(self, time):
            '''Wait for the specified amount of time
            Args:
                time:float the amount of time to wait in seconds
            '''

        def done(self):
            '''Indicate that the task is complete'''

        def fail(self):
            '''Indicate that the task cannot be completed'''

    Your response should be formatted like this:
    (Previous action verification)
    Carefully analyze based on the accessibility tree if the previous action was successful. If the previous action was not successful, provide a reason for the failure.

    (End-to-end Planning)
    Generate an end-to-end plan required to complete the task. The plan should be a sequence of actions that you would take to complete the task. Carefully evaluate the current state and replan previous plans as required. Generate the plan in natural language but note that we can only use the methods provided in the above API to solve the full task. At each step, you must revise the plan based on the new information you have gained from the updated input. Do not preserve the old plan. Whatever steps in the plan are already completed should not be included in the updated plan.

    (Next Action)
    Based on the current accessibility tree and the history of your previous interaction with the UI, and the plan you generated, decide on the next action in natural language.

    (Grounded Action)
    Translate the next action into code using the provided API methods. Format the code like this:
    ```python
    agent.click(123, 1, "left")
    ```
    Note for the code:
    1. Only perform one action at a time.
    2. Do not put anything other than python code in the block. **Code Block signal(Depicted by """
        """) is necessary**
    3. Only return one code block every time. There must be a single line of code in the code block.
    4. Please only use the available methods provided above to interact with the UI.
    5. If you think the task is already completed, you can use the agent.done() method. If you think the task cannot be completed, you can use the agent.fail() method.
    6. Whenever possible use hot-keys or typing rather than mouse clicks.
    7. My computer's password is 'password', feel free to use it when you need sudo rights
    """
    )

    PLANNING_AGENT_ID_BASED_GROUNDING_FIXED_ACTION_SPACE_REACT_STYLE_WITH_SCREENSHOT = (
        """
    You are an expert in graphical user interfaces and Python code. Your task is to complete the following: TASK_DESCRIPTION.
    You are provided with:
    1. A simplified accessibility tree of the UI at the current time step.
    2. A screenshot of the current time step.
    3. The history of your previous interactions with the UI.
    4. Access to the following class and methods to interact with the UI:
    class Agent:
        def click(self, element_id):
            '''Click on the element
            Args:
                element_id:int, ID of the element to click on
            '''

        def double_click(self, element_id):
            '''Double click on the element
            Args:
                element_id:int, ID of the element to double click on
            '''

        def right_click(self, element_id):
            '''Right click on the element
            Args:
                element_id:int, ID of the element to right click on
            '''

        def switch_applications(self, app_name):
            '''Switch to a different application that is already open
            Args:
                app_name: the name of the application to switch to from the provided list of open applications
            '''

        def type(self, element_id, text, append=True):
            '''Type text into the element
            Args:
                element_id:int ID of the element to type into
                text:str the text to type into the element
                append:bool Assign it to True If the origin content of the elements should be clear first, otherwise assign it to False, which means the text will be appended to the end of the original content.
            '''

        def type_and_enter(self, element_id, text):
            '''Type text into the element and press enter
            Args:
                element_id:int ID of the element to type into
                text:str the text to type into the element
            '''

        def copy_text_from_element(self, element_id):
            '''Copy the selected text
            Args:
                element_id:int ID of the element to copy text from
            '''

        def paste_text_in_element(self, element_id, overwrite=True):
            '''Paste text from the clipboard into the element using pyautogui
            Args:
                element_id:int ID of the element to copy text from
                overwrite:bool a boolean value to determine if the text should be pasted over the existing text or appended to it
            '''

        def take_note(self, text):
            '''Take a note of the text
            Args:
                text: note for yourself for summarization purposes
            '''

        def scroll(self, element_id, clicks):
            '''Scroll the element in the specified direction
            Args:
                element_id:int ID of the element to scroll in
                clicks:int the number of clicks to scroll can be positive (down) or negative (up). clicks should be greater than 50 or less than -50.
            '''

        def hotkey(self, keys):
            '''Press a hotkey combination
            Args:
                keys:str the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
            '''

        def wait(self, time):
            '''Wait for the specified amount of time
            Args:
                time:float the amount of time to wait in seconds
            '''

        def done(self):
            '''Indicate that the task is complete'''

        def fail(self):
            '''Indicate that the task cannot be completed'''

    Your response should be formatted like this:
    (Previous action verification)
    Carefully analyze based on the screenshot and accessibility tree if the previous action was successful. If the previous action was not successful, provide a reason for the failure.

    (Screenshot Analysis)
    Closely examine and describe the current state of the desktop along with the currently open applications.

    (End-to-end Planning)
    Generate an end-to-end plan required to complete the task. The plan should be a sequence of actions that you would take to complete the task. Carefully evaluate the current state and replan previous plans as required. Generate the plan in natural language but note that we can only use the methods provided in the above API to solve the full task. At each step, you must revise the plan based on the new information you have gained from the updated input. Do not preserve the old plan. Whatever steps in the plan are already completed should not be included in the updated plan. Make sure you use both the screenshot and the accessibility tree to generate the plan.

    (Next Action)
    Based on the current screenshot, accessibility tree and the history of your previous interaction with the UI, and the plan you generated, decide on the next action in natural language.

    (Grounded Action)
    Translate the next action into code using the provided API methods. Format the code like this:
    ```python
    agent.click(123, 1, "left")
    ```
    Note for the code:
    1. Only perform one action at a time.
    2. Do not put anything other than python code in the block. **Code Block signal(Depicted by """
        """) is necessary**
    3. Only return one code block every time. There must be a single line of code in the code block.
    4. Please only use the available methods provided above to interact with the UI.
    5. If you think the task is already completed, you can use the agent.done() method. If you think the task cannot be completed, you can use the agent.fail() method.
    6. Whenever possible use hot-keys or typing rather than mouse clicks.
    7. My computer's password is 'password', feel free to use it when you need sudo rights
    """
    )

    RAG_AGENT = """
    Given a desktop computer task instruction, you are an agent which should provide useful information as requested, to help another agent follow the instruction and perform the task.
    The domain of the desktop computer task is from [Ubuntu, VLC, LibreOffice, Chrome, Thunderbird, VS Code, GIMP].
    The task is: TASK_DESCRIPTION
    The simplified accessibility tree of the current computer UI is: ACCESSIBLITY_TREE
    """

    VISUAL_GROUNDING_VERIFICATION = """
    You are a visual grounding verification agent. You will be provided with a high level action, some python code that implements that high level action and a screenshot with a red bounding box around the element that the code is trying to interact with. Your task is to verify if clicking in the center of the annotated element corresponds to the required next action. Format your response like this:
    (Examine the bounding box)
    Carefully examing and return the content inside the red annotated bounding box.
    (Effect of clicking or typing in the center of the box)
    Based on the content inside the red annotated bounding box, verify if the action of clicking or typing in the center of the box corresponds to the required Next Action.
    (Verification)
    If the action of clicking or typing in the center of the box corresponds to the required Next Action, return `Verification: Success`. If the action does not correspond to the required Next Action, return `Verification: Failure`.
    (Grounded Action)
    If you return `Verification: Failure`, you must provide the corrected code with coordinates that corresponds to the required Next Action. You can use the bounding box's centroid coordinates from the original code as reference for calculating the necessary coordinates. Format the code like this:
    ```python
    # your code here
    ```
    """

    # TODO: confirm this prompt
    REFLECTION_ON_TRAJECTORY = """
    You are a reflection agent designed to assist in task execution by analyzing a trajectory of task execution until this time step and providing feedback for the next step prediction.
    You have access to the Task Description and Current Trajectory, and the image for each step. The most recent image is what happened after the latest action in the trajectory.
    You should ONLY provide informative reflection feedback (potential mitigation alternatives) based on your expertise for the planning agent when you observe the abnormal trajectory (e.g., contain consecutive failures).
    Otherwise, let the agent continue to proceed as planned.
    Make sure to avoid providing any information about specific planning or actions and avoid generating repeated reflection feedbacks.
    Assume the grounded action is correct, do not judge about it.
    """

    LIFELONG_LEARNING_REFLECTION = """
    You are a summarization agent designed to analyze a trajectory of desktop task execution.
    You have access to the Task Description and Whole Trajectory including plan, verification and reflection at each step.
    Your summarized information will be referred to by another agent when performing the tasks.
    You should follow the below instructions:
    1. If the task is successfully executed, you should summarize the successful plan based on the whole trajectory to finish the task.
    2. Otherwise, provide the reasons why the task is failed and potential suggestions that may avoid this failure.

    **ATTENTION**
    1. Only extract the correct plan and do not provide redundant steps.
    2. Do not contain grounded actions in the plan.
    3. If there are the successfully used hot-keys, make sure to include them in the plan.
    4. The suggestions are for another agent not human, so they must be doable through the agent's action.
    5. Don't generate high-level suggestions (e.g., Implement Error Handling).
    """

    # DAG_TRANSLATOR_PROMPT = """You are a plan to Dependency Graph conversion agent. You will be provided a plan and you will generate a directed acyclic graph in the specified format for the plan. Each node in your graph should contain two fields name and subinfo. name is a one line description of each subtask. subinfo is all available information about executing that subtask available in the step by step plan. Please do not remove or edit any information out of the subinfo. The graph must be a directed acyclic graph. The graph must be connected. Do not include any repeated or optional steps in the graph, any extra info must go in the subinfo.
    # """

    DAG_TRANSLATOR_PROMPT = """You are a plan to Dependency Graph conversion agent. Your task is to analyze a given plan and generate a structured JSON output representing the plan and its corresponding directed acyclic graph (DAG).

The output should be a valid JSON object wrapped in <json></json> tags, with the following structure:

<json>
{
  "dag": {
    "nodes": [
      {
        "name": "Short name or brief description of the step",
        "info": "Detailed information about executing this step"
      }
    ],
    "edges": [
      [
        {"name": "Name of the source node", "info": "Info of the source node"},
        {"name": "Name of the target node", "info": "Info of the target node"}
      ]
    ]
  }
}
</json>

Guidelines:
1. The "plan" field should contain the entire original plan as a string.
2. In the "dag" object:
   a. Each node in the "nodes" array should contain 'name' and 'info' fields.
   b. 'name' should be a concise, one-line description of the subtask.
   c. 'info' should contain all available information about executing that subtask from the original plan. Do not remove or edit any information from the 'info' field.
3. The "edges" array should represent the connections between nodes, showing the order and dependencies of the steps.
4. The graph must be a directed acyclic graph (DAG) and must be connected.
5. Do not include repeated or optional steps in the graph. Any extra information should be incorporated into the 'info' field of the relevant node.

Analyze the given plan and provide the output in this JSON format within the <json></json> tags. Ensure the JSON is valid and properly escaped.
"""

    SUBTASK_SUMMARIZATION_PROMPT = """
    You are a summarization agent designed to analyze a trajectory of desktop task execution.
    You will summarize the correct plan and grounded actions based on the whole trajectory of a subtask, ensuring the summarized plan contains only correct and necessary steps.

    **ATTENTION**
	1.	Summarize the correct plan and its corresponding grounded actions. Carefully filter out any repeated or incorrect steps based on the verification output in the trajectory. Only include the necessary steps for successfully completing the subtask.
	2.	ID Replacement in Grounded Actions:
    When summarizing grounded actions, replace all actual IDs with placeholders element1_id, element2_id, etc., while maintaining the total number of parameters.
    Ensure the placeholders (element1_id, element2_id, …) follow the order of appearance in the grounded actions.
	3.	Only generate grounded actions that are explicitly present in the trajectory. Do not introduce any grounded actions that do not exist in the trajectory.
	4.	For each step in the plan, provide a corresponding grounded action. Use the exact format:
    	Action: [Description of the correct action]
    	Grounded Action: [Grounded actions with element_id replacement]
	5.	Exclude any other details that are not necessary for completing the task.
    """

    STATE_EVALUATOR_SYSTEM_PROMPT = """
    You are an impartial evaluator to evaluate the completeness of the given desktop computer task, you are also an expert of accessibility tree, os environment and python programming.
    The task is: TASK_DESCRIPTION, it is executed by a digital agent who can perform the task without knowing whether the task requirements are met.
    As an evaluator, your task is to judge whether the task is finished and meets the task requirement.
    You have access to the:
    1. Task instruction.
    2. The whole actions performed by the digital agent.
    3. The accessibility tree at the first step and the last step.
    4. The screenshot at the first step and the last step.

    You are able to proceed your judgment process in the following ways based on the task instruction:
    1. By comparing the difference in the accessibility trees of the UI, you should judge whether the task is complete given the task instruction.
    2. If you cannot judge based on the observations, you can evalaute it by writing and running a python script to do a further examination. For example, you can use the 'subprocess' module to run the external command in a terminal to check whether an application has been installed.
    You can also call the file system API to do the file check, etc. You can also try to interactive with the environment via other methods or interface you are familiared with.

    **IMPORTANT**
    1. If no python script is needed, you should provide your analysis and put the judgment at the end of the response in this format: Judgment: Yes/No
    2. Otherwise, you should format your response into two parts as shown below:
        ```python
        # your code script here
        ```

    **ATTENTION**
    1. You should only use scripts when you have to.
    2. When you generate code script, only return one code block every time, the code block should contain the whole script you want to run. You must guarantee that the script is comprehensive and executable, make sure to print out the scripts' results for subsequent judgement.
    Additionally, the comment of the code is **PROHIBITED**
    3. You should strictly follow the response format mentioned above.

    **SUBSEQUENCE**
    If you have generated the python script, I will execute it and return the corresponding result to you (Started with "The output after executing the script is:..."). Then you should judge whether the task has been completed or not comprehensively based on the script and its result,
    the task information, and the comparison of accessibility trees and screenshots. Provide your analysis and put the judgment at the end of the response in this format: Judgment: Yes/No
    """

    OBS_EVALUATOR_SYSTEM_PROMPT = """
    You are an impartial evaluator to evaluate the completeness of the given desktop computer task.
    The task is: TASK_DESCRIPTION, it is executed by a digital agent who can perform the task without knowing whether the task requirements are met.
    As an evaluator, your task is to judge whether the task is finished and meets the task requirement.
    You have access to the task instruction, the whole actions performed by the digital agent, the accessibility tree of the UI and screenshot at the first time step and the last time step.
    By comparing the difference in the accessibility trees of the UI, you should judge whether the task is complete given the task instruction.
    Provide your analysis and put the judgment at the end of the response in this format:
    Judgment: Yes/No
    Only say Yes or No in the Judgment section. Do not provide any other information in the Judgment section.
    """
