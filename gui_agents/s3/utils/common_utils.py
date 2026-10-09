import re
import time
from io import BytesIO
from PIL import Image

from typing import Tuple, Dict, Optional

from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY

import logging

logger = logging.getLogger("desktopenv.agent")


def image_media_type(image_bytes: bytes) -> str:
    """Returns the MIME type of an encoded image, defaulting to image/png."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def compress_image_bytes(image_bytes: bytes, max_bytes: Optional[int]) -> bytes:
    """
    Shrinks an encoded image until it fits within ``max_bytes``.

    Self-hosted inference servers often cap the request body size (e.g. the
    HuggingFace text-generation-inference default is 2 MB), and a full
    resolution PNG screenshot easily exceeds that once base64 encoded. This
    helper first re-encodes the image as JPEG at decreasing quality, and only
    downscales the image once quality reduction alone is not enough.

    Args:
        image_bytes (bytes): The encoded image (PNG, JPEG, ...).
        max_bytes (Optional[int]): The size budget. ``None`` disables compression.

    Returns:
        bytes: ``image_bytes`` unchanged if it already fits, otherwise the
        smallest re-encoded image that fits, or the smallest attempt if the
        budget could not be reached.
    """
    if max_bytes is None or len(image_bytes) <= max_bytes:
        return image_bytes

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    best = image_bytes

    def encode(img: Image.Image, quality: int) -> bytes:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    qualities = [90, 80, 70, 60, 50]
    scale = 1.0
    min_scale = 0.3
    while True:
        if scale < 1.0:
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            candidate_image = image.resize(size, Image.LANCZOS)
        else:
            candidate_image = image

        for quality in qualities:
            encoded = encode(candidate_image, quality)
            if len(encoded) < len(best):
                best = encoded
            if len(encoded) <= max_bytes:
                logger.info(
                    "Compressed image from %d to %d bytes (scale=%.2f, quality=%d) to fit %d byte budget",
                    len(image_bytes),
                    len(encoded),
                    scale,
                    quality,
                    max_bytes,
                )
                return encoded

        scale *= 0.8
        if scale < min_scale:
            logger.warning(
                "Could not compress image under %d bytes; sending smallest attempt (%d bytes)",
                max_bytes,
                len(best),
            )
            return best


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
    exec_code = eval(code)
    return exec_code


PAYLOAD_TOO_LARGE_MARKERS = (
    "length limit exceeded",
    "payload too large",
    "request entity too large",
    "413",
)


def payload_too_large_hint(error: Exception) -> Optional[str]:
    """Returns guidance when an endpoint rejected the request for being too big."""
    message = str(error).lower()
    if any(marker in message for marker in PAYLOAD_TOO_LARGE_MARKERS):
        return (
            "The endpoint rejected the request body as too large. This usually means "
            "the screenshot exceeds the server's payload limit (HuggingFace TGI "
            "defaults to 2 MB). Re-run with --max_image_mb (e.g. --max_image_mb 1.0) "
            "or raise the server's payload limit."
        )
    return None


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
            hint = payload_too_large_hint(e)
            if hint:
                print(f"Hint: {hint}")
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
