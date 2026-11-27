import ast
import re
import time
from io import BytesIO
from PIL import Image

from typing import Tuple, Dict

from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY

import logging

logger = logging.getLogger("desktopenv.agent")


def dispatch_agent_action(agent, code: str):
    """Safely evaluate a single grounded ``agent.<action>(...)`` call.

    The model-produced grounded action is parsed with ``ast`` and dispatched to
    the corresponding grounding-agent method instead of being run through
    ``eval``. Only a single call on the ``agent`` object is allowed, the target
    must be a method flagged with ``is_agent_action``, and every argument must be
    a plain Python literal (``ast.literal_eval``). This preserves every valid
    grounded action while preventing arbitrary expressions/builtins (e.g.
    ``__import__``, tuple side effects, attribute walks) from executing during
    response validation or action conversion.

    Args:
        agent (ACI): The grounding agent that owns the action method.
        code (str): The grounded-action call string to evaluate.

    Returns:
        The return value of the grounding-agent action (the pyautogui code).

    Raises:
        ValueError: If ``code`` is not a single allowed ``agent.<action>(...)``
            call with literal-only arguments.
    """
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid grounded action syntax: {e}") from e

    call = tree.body
    if not isinstance(call, ast.Call):
        raise ValueError("Grounded action must be a single agent action call.")

    func = call.func
    if (
        not isinstance(func, ast.Attribute)
        or not isinstance(func.value, ast.Name)
        or func.value.id != "agent"
    ):
        raise ValueError("Grounded action must call a method on 'agent'.")

    method_name = func.attr
    method = getattr(agent, method_name, None)
    if method is None or not getattr(method, "is_agent_action", False):
        raise ValueError(f"Unknown or disallowed agent action: {method_name!r}.")

    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            "Grounded action arguments must be literals (no expressions/calls)."
        ) from e

    if any(kw.arg is None for kw in call.keywords):
        raise ValueError("Grounded action does not support **kwargs unpacking.")

    return method(*args, **kwargs)


def create_pyautogui_code(agent, code: str, obs: Dict) -> str:
    """
    Attempts to evaluate the code into a pyautogui code snippet with grounded actions using the observation screenshot.

    Args:
        agent (ACI): The grounding agent to use for evaluation.
        code (str): The code string to evaluate.
        obs (Dict): The current observation containing the screenshot.

    Returns:
        exec_code (str): The pyautogui code to execute the grounded action.

    Raises:
        Exception: If there is an error in evaluating the code.
    """
    agent.assign_screenshot(obs)  # Necessary for grounding
    exec_code = dispatch_agent_action(agent, code)
    return exec_code


def call_llm_safe(
    agent, temperature: float = 0.0, use_thinking: bool = False, **kwargs
) -> str:
    # Retry if fails
    max_retries = 3  # Set the maximum number of retries
    attempt = 0
    response = ""
    while attempt < max_retries:
        try:
            response = agent.get_response(
                temperature=temperature, use_thinking=use_thinking, **kwargs
            )
            assert response is not None, "Response from agent should not be None"
            print("Response success!")
            break  # If successful, break out of the loop
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
    return response if response is not None else ""


def call_llm_formatted(generator, format_checkers, **kwargs):
    """
    Calls the generator agent's LLM and ensures correct formatting.

    Args:
        generator (ACI): The generator agent to call.
        obs (Dict): The current observation containing the screenshot.
        format_checkers (Callable): Functions that take the response and return a tuple of (success, feedback).
        **kwargs: Additional keyword arguments for the LLM call.

    Returns:
        response (str): The formatted response from the generator agent.
    """
    max_retries = 3  # Set the maximum number of retries
    attempt = 0
    response = ""
    if kwargs.get("messages") is None:
        messages = (
            generator.messages.copy()
        )  # Copy messages to avoid modifying the original
    else:
        messages = kwargs["messages"]
        del kwargs["messages"]  # Remove messages from kwargs to avoid passing it twice
    while attempt < max_retries:
        response = call_llm_safe(generator, messages=messages, **kwargs)

        # Prepare feedback messages for incorrect formatting
        feedback_msgs = []
        for format_checker in format_checkers:
            success, feedback = format_checker(response)
            if not success:
                feedback_msgs.append(feedback)
        if not feedback_msgs:
            # logger.info(f"Response formatted correctly on attempt {attempt} for {generator.engine.model}")
            break
        logger.error(
            f"Response formatting error on attempt {attempt} for {generator.engine.model}. Response: {response} {', '.join(feedback_msgs)}"
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
        )
        logger.info(f"Bad response: {response}")
        delimiter = "\n- "
        formatting_feedback = f"- {delimiter.join(feedback_msgs)}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROCEDURAL_MEMORY.FORMATTING_FEEDBACK_PROMPT.replace(
                            "FORMATTING_FEEDBACK", formatting_feedback
                        ),
                    }
                ],
            }
        )
        logger.info("Feedback:\n%s", formatting_feedback)

        attempt += 1
        if attempt == max_retries:
            logger.error(
                "Max retries reached when formatting response. Handling failure."
            )
        time.sleep(1.0)
    return response


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    try:
        # Extract thoughts section
        thoughts = full_response.split("<thoughts>")[-1].split("</thoughts>")[0].strip()

        # Extract answer section
        answer = full_response.split("<answer>")[-1].split("</answer>")[0].strip()

        return answer, thoughts
    except Exception as e:
        return full_response, ""


def parse_code_from_string(input_string):
    """Parses a string to extract each line of code enclosed in triple backticks (```)

    Args:
        input_string (str): The input string containing code snippets.

    Returns:
        str: The last code snippet found in the input string, or an empty string if no code is found.
    """
    input_string = input_string.strip()

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"

    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)
    if len(matches) == 0:
        # return []
        return ""
    relevant_code = matches[
        -1
    ]  # We only care about the last match given it is the grounded action
    return relevant_code


def extract_agent_functions(code):
    """Extracts all agent function calls from the given code.

    Args:
        code (str): The code string to search for agent function calls.

    Returns:
        list: A list of all agent function calls found in the code.
    """
    pattern = r"(agent\.\w+\(\s*.*\))"  # Matches
    return re.findall(pattern, code)


def compress_image(image_bytes: bytes = None, image: Image = None) -> bytes:
    """Compresses an image represented as bytes.

    Compression involves resizing image into half its original size and saving to webp format.

    Args:
        image_bytes (bytes): The image data to compress.

    Returns:
        bytes: The compressed image data.
    """
    if not image:
        image = Image.open(BytesIO(image_bytes))
    output = BytesIO()
    image.save(output, format="WEBP")
    compressed_image_bytes = output.getvalue()
    return compressed_image_bytes
