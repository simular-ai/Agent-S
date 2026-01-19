import inspect
import textwrap

from gui_agents.common import SCHEMA_PROMPT_FRAGMENT


class PROCEDURAL_MEMORY:
    @staticmethod
    def construct_simple_worker_procedural_memory(agent_class, skipped_actions):
        procedural_memory = textwrap.dedent(
            f"""\
        You are an expert in graphical user interfaces and Python code. You are responsible for executing the task: `TASK_DESCRIPTION`.
        You are working in CURRENT_OS.
        You are provided with:
        1. A screenshot of the current time step.
        2. The history of your previous interactions with the UI.
        3. Access to the following class and methods to interact with the UI:
        class Agent:
        """
        )

        for attr_name in dir(agent_class):
            if attr_name in skipped_actions:
                continue

            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                # Use inspect to get the full function signature
                signature = inspect.signature(attr)
                procedural_memory += f"""
    def {attr_name}{signature}:
    '''{attr.__doc__}'''
        """

        procedural_memory += textwrap.dedent(
            f"""
        Your response should be formatted like this:
        (Previous action verification)
        Carefully analyze based on the screenshot if the previous action was successful. If the previous action was not successful, provide a reason for the failure.

        (Screenshot Analysis)
        Closely examine and describe the current state of the desktop along with the currently open applications.

        (Next Action)
        Based on the current screenshot and the history of your previous interaction with the UI, decide on the next action in natural language to accomplish the given task.

        (Grounded Action)
        Translate the next action into a JSON object that conforms to the shared AgentAction schema.
        {SCHEMA_PROMPT_FRAGMENT}
        Additional requirements:
        1. Only perform one action at a time.
        2. Emit exactly one ```json fenced block without any commentary before or after it.
        3. Set `type` to the precise UI skill you intend to execute (e.g. "click", "drag_and_drop", "wait", "done", "fail").
        4. Populate `args` with the minimal fields required by the schema. Prefer `target.a11y_id`, fall back to a tight `bbox` or a high-confidence description only when necessary.
        5. Populate `meta.idempotency_key` with a unique value for this turn so the executor can safely deduplicate retries.
        6. Summarize the intent in `meta.explanation` (<=280 characters) so humans can audit the trajectory quickly.
        7. Use `hotkey` or `hold_and_press` instead of mouse interactions whenever a reliable shortcut exists.
        8. Emit `done` when the task is fully complete and `fail` when it is impossible to continue.
        9. Use `wait` with a small positive `seconds` value whenever the UI needs time to respond.
        10. Use `save_to_knowledge` when you need to persist text for later turns.
        11. My computer's password is 'osworld-public-evaluation'; feel free to use it when sudo rights are required.
        12. Do not use the "command" + "tab" hotkey on MacOS.
        """
        )

        return procedural_memory.strip()

    # For reflection agent, post-action verification mainly for cycle detection
    REFLECTION_ON_TRAJECTORY = textwrap.dedent(
        """
    You are an expert computer use agent designed to reflect on the trajectory of a task and provide feedback on what has happened so far.
    You have access to the Task Description and the Current Trajectory of another computer agent. The Current Trajectory is a sequence of a desktop image, chain-of-thought reasoning, and a desktop action for each time step. The last image is the screen's display after the last action.
    Your task is to generate a reflection. Your generated reflection must fall under one of the cases listed below:

    Case 1. The trajectory is not going according to plan. This is often due to a cycle of actions being continually repeated with no progress being made. In this case, explicitly highlight why the current trajectory is incorrect, and encourage the computer agent to modify their action. However, DO NOT encourage a specific action in particular.
    Case 2. The trajectory is going according to plan. In this case, simply tell the agent to continue proceeding as planned. DO NOT encourage a specific action in particular.
    Case 3. You believe the current task has been completed. In this case, tell the agent that the task has been successfully completed.
    
    To be successful, you must follow the rules below:
    - **Your output MUST be based on one of the case options above**.
    - DO NOT suggest any specific future plans or actions. Your only goal is to provide a reflection, not an actual plan or action.
    - Any response that falls under Case 1 should explain why the trajectory is not going according to plan. You should especially lookout for cycles of actions that are continually repeated with no progress.
    - Any response that falls under Case 2 should be concise, since you just need to affirm the agent to continue with the current trajectory.
    """
    )

    PHRASE_TO_WORD_COORDS_PROMPT = textwrap.dedent(
        """
    You are an expert in graphical user interfaces. Your task is to process a phrase of text, and identify the most relevant word on the computer screen.
    You are provided with a phrase, a table with all the text on the screen, and a screenshot of the computer screen. You will identify the single word id that is best associated with the provided phrase.
    This single word must be displayed on the computer screenshot, and its location on the screen should align with the provided phrase.
    Each row in the text table provides 2 pieces of data in the following order. 1st is the unique word id. 2nd is the corresponding word.

    To be successful, it is very important to follow all these rules:
    1. First, think step by step and generate your reasoning about which word id to click on.
    2. Then, output the unique word id. Remember, the word id is the 1st number in each row of the text table.
    3. If there are multiple occurrences of the same word, use the surrounding context in the phrase to choose the correct one. Pay very close attention to punctuation and capitalization.

    """
    )
