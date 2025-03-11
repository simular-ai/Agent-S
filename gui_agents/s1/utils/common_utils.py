import base64
import io
import json
import os
import pickle
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Tuple, Union
from xml.etree.ElementTree import Element

import numpy as np
import tiktoken
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError


def find_leaf_nodes(xlm_file_str):
    if not xlm_file_str:
        return []

    root = ET.fromstring(xlm_file_str)

    # Recursive function to traverse the XML tree and collect leaf nodes
    def collect_leaf_nodes(node, leaf_nodes):
        # If the node has no children, it is a leaf node, add it to the list
        if not list(node):
            leaf_nodes.append(node)
        # If the node has children, recurse on each child
        for child in node:
            collect_leaf_nodes(child, leaf_nodes)

    # List to hold all leaf nodes
    leaf_nodes = []
    collect_leaf_nodes(root, leaf_nodes)
    return leaf_nodes


state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"


class Node(BaseModel):
    name: str
    info: str


class Dag(BaseModel):
    nodes: List[Node]
    edges: List[List[Node]]


NUM_IMAGE_TOKEN = 1105  # Value set of screen of size 1920x1080 for openai vision


def call_llm_safe(agent) -> Union[str, Dag]:
    # Retry if fails
    max_retries = 3  # Set the maximum number of retries
    attempt = 0
    response = ""
    while attempt < max_retries:
        try:
            response = agent.get_response()
            break  # If successful, break out of the loop
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
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


def judge_node(node: Element, platform="ubuntu", check_image=False) -> bool:
    keeps: bool = (
        node.tag.startswith("document")
        or node.tag.endswith("item")
        or node.tag.endswith("button")
        or node.tag.endswith("heading")
        or node.tag.endswith("label")
        or node.tag.endswith("scrollbar")
        or node.tag.endswith("searchbox")
        or node.tag.endswith("textbox")
        or node.tag.endswith("link")
        or node.tag.endswith("tabelement")
        or node.tag.endswith("textfield")
        or node.tag.endswith("textarea")
        or node.tag.endswith("menu")
        or node.tag.endswith("menu-item")
        or node.tag
        in {
            "alert",
            "canvas",
            "check-box",
            "combo-box",
            "entry",
            "icon",
            "image",
            "paragraph",
            "scroll-bar",
            "section",
            "slider",
            "static",
            "table-cell",
            "terminal",
            "text",
            "netuiribbontab",
            "start",
            "trayclockwclass",
            "traydummysearchcontrol",
            "uiimage",
            "uiproperty",
            "uiribboncommandbar",
        }
    )

    keeps = (
        keeps
        and (
            platform == "ubuntu"
            and node.get("{{{:}}}showing".format(state_ns), "false") == "true"
            and node.get("{{{:}}}visible".format(state_ns), "false") == "true"
            or platform == "windows"
            and node.get("{{{:}}}visible".format(state_ns), "false") == "true"
        )
        and (
            node.get("name", "") != ""
            or node.text is not None
            and len(node.text) > 0
            or check_image
            and node.get("image", "false") == "true"
        )
    )
    # and (node.get("{{{:}}}enabled".format(state_ns), "false") == "true" \
    #      or node.get("{{{:}}}editable".format(state_ns), "false") == "true" \
    #      or node.get("{{{:}}}expandable".format(state_ns), "false") == "true" \
    #      or node.get("{{{:}}}checkable".format(state_ns), "false") == "true"
    #      ) \

    coordinates: Tuple[int, int] = eval(
        node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
    )
    sizes: Tuple[int, int] = eval(
        node.get("{{{:}}}size".format(component_ns), "(-1, -1)")
    )
    keeps = (
        keeps
        and coordinates[0] >= 0
        and coordinates[1] >= 0
        and sizes[0] > 0
        and sizes[1] > 0
    )
    return keeps


def filter_nodes(root: Element, platform="ubuntu", check_image=False):
    filtered_nodes = []
    all_nodes = []
    for node in root.iter():
        all_nodes.append(node)

    for node in root.iter():
        if judge_node(node, platform, check_image):
            filtered_nodes.append(node)

    return filtered_nodes


def draw_bounding_boxes(nodes, image_file_content, down_sampling_ratio=1.0):
    # Load the screenshot image
    image_stream = io.BytesIO(image_file_content)
    image = Image.open(image_stream)
    if float(down_sampling_ratio) != 1.0:
        image = image.resize(
            (
                int(image.size[0] * down_sampling_ratio),
                int(image.size[1] * down_sampling_ratio),
            )
        )
    draw = ImageDraw.Draw(image)
    marks = []
    drew_nodes = []
    text_informations: List[str] = ["index\ttag\tname\ttext"]

    try:
        # Adjust the path to the font file you have or use a default one
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Fallback to a basic font if the specified font can't be loaded
        font = ImageFont.load_default()

    index = 1

    # Loop over all the visible nodes and draw their bounding boxes
    for _node in nodes:
        coords_str = _node.attrib.get(
            "{uri:deskat:component.at-spi.gnome.org}screencoord"
        )
        size_str = _node.attrib.get("{uri:deskat:component.at-spi.gnome.org}size")

        if coords_str and size_str:
            try:
                # Parse the coordinates and size from the strings
                coords = tuple(map(int, coords_str.strip("()").split(", ")))
                size = tuple(map(int, size_str.strip("()").split(", ")))

                import copy

                original_coords = copy.deepcopy(coords)
                original_size = copy.deepcopy(size)

                if float(down_sampling_ratio) != 1.0:
                    # Downsample the coordinates and size
                    coords = tuple(int(coord * down_sampling_ratio) for coord in coords)
                    size = tuple(int(s * down_sampling_ratio) for s in size)

                # Check for negative sizes
                if size[0] <= 0 or size[1] <= 0:
                    raise ValueError(f"Size must be positive, got: {size}")

                # Calculate the bottom-right corner of the bounding box
                bottom_right = (coords[0] + size[0], coords[1] + size[1])

                # Check that bottom_right > coords (x1 >= x0, y1 >= y0)
                if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
                    raise ValueError(
                        f"Invalid coordinates or size, coords: {coords}, size: {size}"
                    )

                # Check if the area only contains one color
                cropped_image = image.crop((*coords, *bottom_right))
                if len(set(list(cropped_image.getdata()))) == 1:
                    continue

                # Draw rectangle on image
                draw.rectangle([coords, bottom_right], outline="red", width=1)

                # Draw index number at the bottom left of the bounding box with black background
                text_position = (
                    coords[0],
                    bottom_right[1],
                )  # Adjust Y to be above the bottom right
                text_bbox: Tuple[int, int, int, int] = draw.textbbox(
                    text_position, str(index), font=font, anchor="lb"
                )
                # offset: int = bottom_right[1]-text_bbox[3]
                # text_bbox = (text_bbox[0], text_bbox[1]+offset, text_bbox[2], text_bbox[3]+offset)

                # draw.rectangle([text_position, (text_position[0] + 25, text_position[1] + 18)], fill='black')
                draw.rectangle(text_bbox, fill="black")
                draw.text(
                    text_position, str(index), font=font, anchor="lb", fill="white"
                )

                # each mark is an x, y, w, h tuple
                marks.append(
                    [
                        original_coords[0],
                        original_coords[1],
                        original_size[0],
                        original_size[1],
                    ]
                )
                drew_nodes.append(_node)

                if _node.text:
                    node_text = (
                        _node.text
                        if '"' not in _node.text
                        else '"{:}"'.format(_node.text.replace('"', '""'))
                    )
                elif _node.get(
                    "{uri:deskat:uia.windows.microsoft.org}class", ""
                ).endswith("EditWrapper") and _node.get(
                    "{uri:deskat:value.at-spi.gnome.org}value"
                ):
                    node_text: str = _node.get(
                        "{uri:deskat:value.at-spi.gnome.org}value"
                    )
                    node_text = (
                        node_text
                        if '"' not in node_text
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
                else:
                    node_text = '""'
                text_information: str = "{:d}\t{:}\t{:}\t{:}".format(
                    index, _node.tag, _node.get("name", ""), node_text
                )
                text_informations.append(text_information)

                index += 1

            except ValueError:
                pass

    output_image_stream = io.BytesIO()
    image.save(output_image_stream, format="PNG")
    image_content = output_image_stream.getvalue()

    return marks, drew_nodes, "\n".join(text_informations), image_content


def print_nodes_with_indent(nodes, indent=0):
    for node in nodes:
        print(" " * indent, node.tag, node.attrib)
        print_nodes_with_indent(node, indent + 2)


# Code based on https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/agent.py


def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")


def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu", tag=False):
    # leaf_nodes = find_leaf_nodes(accessibility_tree)
    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tposition (top-left x&y)\tsize (w&h)"
    ]
    # Linearize the accessibility tree nodes into a table format

    for node in filtered_nodes:
        # linearized_accessibility_tree += node.tag + "\t"
        # linearized_accessibility_tree += node.attrib.get('name') + "\t"
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )
        elif node.get("{uri:deskat:uia.windows.microsoft.org}class", "").endswith(
            "EditWrapper"
        ) and node.get("{uri:deskat:value.at-spi.gnome.org}value"):
            text: str = node.get("{uri:deskat:value.at-spi.gnome.org}value")
            text = text if '"' not in text else '"{:}"'.format(text.replace('"', '""'))
        else:
            text = '""'
        # linearized_accessibility_tree += node.attrib.get(
        # , "") + "\t"
        # linearized_accessibility_tree += node.attrib.get('{uri:deskat:component.at-spi.gnome.org}size', "") + "\n"
        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                node.get("{uri:deskat:component.at-spi.gnome.org}screencoord", ""),
                node.get("{uri:deskat:component.at-spi.gnome.org}size", ""),
            )
        )
    if tag:
        linearized_accessibility_tree = tag_accessibility_tree(
            linearized_accessibility_tree
        )

    return "\n".join(linearized_accessibility_tree)


def tag_accessibility_tree(linear_accessibility_tree):
    # Add 'id' to the first line
    linear_accessibility_tree[0] = "id\t" + linear_accessibility_tree[0]

    # Start idx from 1 to correctly index into the list
    for idx in range(1, len(linear_accessibility_tree)):
        line = linear_accessibility_tree[idx]
        linear_accessibility_tree[idx] = f"[{str(idx)}]\t" + line

    return linear_accessibility_tree


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform=platform, check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    return marks, drew_nodes, tagged_screenshot, element_list


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


def parse_subinfo(subinfo_string):
    matches = re.findall(r"```json\s+(.*?)\s+```", subinfo_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            subinfo_dict = json.loads(matches[0])
            return subinfo_dict
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {"error": e}
    else:
        return {
            "error": "Subinfo generated in incorrect format. Please use the correct format."
        }


def parse_actions_from_string(input_string):
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r"```json\s+(.*?)\s+```", input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r"```\s+(.*?)\s+```", input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_fixed_action_from_string(input_string):
    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action = match
            return action
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"

    return "agent.wait()"


def parse_code_from_string(input_string):
    input_string = "\n".join(
        [line.strip() for line in input_string.split(";") if line.strip()]
    )
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

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

    return codes


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

    return codes[0]


def parse_action_from_fixed_code(action_string, linearized_accessibility_tree):

    import re

    def parse_action_from_agent_code(action_str):
        # First, extract the code block within triple backticks
        code_block_pattern = r"```(.*?)```"
        code_block_match = re.search(code_block_pattern, action_str, re.DOTALL)

        if not code_block_match:
            raise ValueError("No code block found")

        code_block = code_block_match.group(1).strip()

        # Define a regex pattern to extract the action type and parameters
        action_pattern = r"agent\.(\w+)\((.*?)\)"
        match = re.match(action_pattern, code_block, re.IGNORECASE)

        if match:
            action_type = match.group(1)
            params_str = match.group(2)

            # Split the parameters by comma and strip any surrounding whitespace or quotes
            params = [
                param.strip().strip('"').strip("'") for param in params_str.split(",")
            ]

            # Convert numeric parameters to integers
            for i in range(len(params)):
                try:
                    params[i] = int(params[i])
                except ValueError:
                    pass

            return action_type, params
        else:
            raise ValueError("Invalid action string format")

    parsed_action = parse_action_from_agent_code(action_string)
    action_type, params = parsed_action
    code = ""

    def get_position_from_tree(element_id):
        element = linearized_accessibility_tree[element_id]
        position_str, size_str = element.split("\t")[-2].replace("(", "").replace(
            ")", ""
        ), element.split("\t")[-1].replace("(", "").replace(")", "")
        top_x_str, top_y_str = position_str.split(",")
        top_x, top_y = int(top_x_str.strip()), int(top_y_str.strip())
        size_x_str, size_y_str = size_str.split(",")
        size_x, size_y = int(size_x_str.strip()), int(size_y_str.strip())
        centroid_x, centroid_y = top_x + size_x // 2, top_y + size_y // 2
        return centroid_x, centroid_y

    if action_type == "left_click_element_by_id":
        element_id = int(params[0])
        centroid_x, centroid_y = get_position_from_tree(element_id)
        code = f"""position = ({centroid_x}, {centroid_y}); pyautogui.click(position)
        """

    elif action_type == "right_click_element_by_id":
        element_id = int(params[0])
        centroid_x, centroid_y = get_position_from_tree(element_id)
        code = f"""
        position = ({centroid_x}, {centroid_y}); pyautogui.click(position, button='right')
        """

    elif action_type == "hover_over_element_by_id":
        element_id = int(params[0])
        centroid_x, centroid_y = get_position_from_tree(element_id)
        code = (
            f"""position = ({centroid_x}, {centroid_y}); pyautogui.moveTo(position)"""
        )

    elif action_type == "type_write_element_by_id":
        element_id = int(params[0])
        text = params[1]
        centroid_x, centroid_y = get_position_from_tree(element_id)
        code = f"""
        position = ({centroid_x}, {centroid_y}); pyautogui.click(position); time.sleep(0.75); pyautogui.typewrite("{text}")"""

    elif action_type == "press_key_combinations":
        keys = params
        keys_str = '", "'.join(keys)
        code = f"""
        pyautogui.hotkey("{keys_str}")
        """

    elif action_type == "wait":
        code = """WAIT"""

    elif action_type == "done":
        code = """DONE"""

    elif action_type == "fail":
        code = "FAIL"

    return [code.strip()]


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += (
            "tag_"
            + str(i + 1)
            + "="
            + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        )
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ["WAIT", "DONE", "FAIL"]:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


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


def calculate_iou(rect1, rect2):
    """
    Calculate the Intersection over Union (IoU) of two rectangles using numpy.

    Parameters:
    rect1, rect2: Tuples containing the coordinates of the rectangles in the form (x_min, y_min, x_max, y_max)

    Returns:
    IoU: Intersection over Union value
    """
    # Convert the coordinates to tensors
    box1 = np.array([rect1], dtype=np.float32)
    box2 = np.array([rect2], dtype=np.float32)

    # Calculate IoU using numpy
    iou = box_iou(box1, box2)

    return iou


def text_cvt_orc_format_paddle(paddle_result):
    texts = []
    print("paddle_result: ", paddle_result)
    for i, line in enumerate(paddle_result[0]):
        points = np.array(line[0])
        print("points: ", points)
        location = {
            "left": int(min(points[:, 0])),
            "top": int(min(points[:, 1])),
            "right": int(max(points[:, 0])),
            "bottom": int(max(points[:, 1])),
        }
        print("location: ", location)
        content = line[1][0]
        texts.append((i, content, location))
    return texts


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        print("MAX TOKEN LENGTH OF ACCESSIBILITY TREE EXCEEDED")
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


def get_input_token_length(input_string):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(input_string)
    return len(tokens)


def load_osworld_example(base_path: str, domain: str, id: int):
    example_path = f"{base_path}/{domain}"
    example_path = (
        f"/Users/saaketagashe/Documents/OSWorld/evaluation_examples/examples/{domain}"
    )
    examples = os.listdir(example_path)

    with open(example_path + "/" + examples[id], "r") as f:
        example = json.load(f)

    return example


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
