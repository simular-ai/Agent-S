import json
import re
from typing import List
import time
import tiktoken

from typing import Tuple, List, Union, Dict

from pydantic import BaseModel, ValidationError

import pickle


class Node(BaseModel):
    name: str
    info: str


class Dag(BaseModel):
    nodes: List[Node]
    edges: List[List[Node]]


NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


async def call_llm_safe(agent) -> Union[str, Dag]:
    # Retry if fails
    max_retries = 3  # Set the maximum number of retries
    attempt = 0
    response = ""
    start_time = time.time()
    while attempt < max_retries:
        try:
            response = await agent.get_response()
            break  # If successful, break out of the loop
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
    end_time = time.time()
    print(f"CALL LLM TIME: {end_time - start_time}")
    return response


def calculate_tokens(messages, num_image_token=NUM_IMAGE_TOKEN) -> Tuple[int, int]:

    num_input_images = 0
    output_message = messages[-1]

    input_message = messages[:-1]

    input_string = """"""
    for message in input_message:
        input_string += message["content"][0]["text"] + "\n"
        if len(message["content"]) > 1:
            num_input_images += 1

    input_text_tokens = get_input_token_length(input_string)

    input_image_tokens = num_image_token * num_input_images

    output_tokens = get_input_token_length(output_message["content"][0]["text"])

    return (input_text_tokens + input_image_tokens), output_tokens


# Code based on https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/agent.py


def parse_dag(text):
    pattern = r"<json>(.*?)</json>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            json_data = json.loads(json_str)
            return Dag(**json_data["dag"])
        except json.JSONDecodeError:
            print("Error: Invalid JSON")
            return None
        except KeyError:
            print("Error: 'dag' key not found in JSON")
            return None
        except ValidationError as e:
            print(f"Error: Invalid data structure - {e}")
            return None
    else:
        print("Error: JSON not found")
        return None


def parse_single_code_from_string(input_string):
    input_string = input_string.strip()
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return input_string.strip()

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = [
            "WAIT",
            "DONE",
            "FAIL",
        ]  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    if len(codes) <= 0:
        return "fail"
    return codes[0]


def get_input_token_length(input_string):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(input_string)
    return len(tokens)


def sanitize_code(code):
    # This pattern captures the outermost double-quoted text
    if "\n" in code:
        pattern = r'(".*?")'
        # Find all matches in the text
        matches = re.findall(pattern, code, flags=re.DOTALL)
        if matches:
            # Replace the first occurrence only
            first_match = matches[0]
            code = code.replace(first_match, f'"""{first_match[1:-1]}"""', 1)
    return code


def extract_first_agent_function(code_string):
    # Regular expression pattern to match 'agent' functions with any arguments, including nested parentheses
    pattern = r'agent\.[a-zA-Z_]+\((?:[^()\'"]|\'[^\']*\'|"[^"]*")*\)'

    # Find all matches in the string
    matches = re.findall(pattern, code_string)

    # Return the first match if found, otherwise return None
    return matches[0] if matches else None


def load_knowledge_base(kb_path: str) -> Dict:
    try:
        with open(kb_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return {}


def load_embeddings(embeddings_path: str) -> Dict:
    try:
        with open(embeddings_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}


def save_embeddings(embeddings_path: str, embeddings: Dict):
    try:
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        print(f"Error saving embeddings: {e}")
