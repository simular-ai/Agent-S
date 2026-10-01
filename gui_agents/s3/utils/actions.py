"""Parse model actions without evaluating Python; prepare and dispatch desktop actions."""

import ast
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ActionSpec:
    name: str
    arguments: dict


@dataclass
class PreparedAction:
    name: str
    arguments: dict
    preview: str


def parse_action(agent, code: str) -> ActionSpec:
    if not isinstance(code, str) or len(code) > 65536:
        raise ValueError("Action must be a bounded Python expression")
    node = ast.parse(code.strip(), mode="eval").body
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "agent"
        and not node.func.attr.startswith("_")
    ):
        raise ValueError("Use one agent.action(...) call with literal arguments")
    method = getattr(agent, node.func.attr, None)
    if not callable(method) or not getattr(method, "is_agent_action", False):
        raise ValueError("Action is not in the agent's action API")
    if any(k.arg is None for k in node.keywords):
        raise ValueError("Expanded keyword arguments are not allowed")
    if len({k.arg for k in node.keywords}) != len(node.keywords):
        raise ValueError("Duplicate keyword argument")
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    bound = inspect.signature(method).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    if (
        node.func.attr == "type"
        and getattr(agent, "require_grounded_typing", False)
        and values["element_description"] is None
    ):
        raise ValueError(
            "Manual typing requires an element_description; approval controls can change keyboard focus"
        )
    if node.func.attr == "call_code_agent" and not getattr(agent, "env", None):
        raise ValueError("Local code execution is disabled")
    for key, value in values.items():
        if key in ("hold_keys", "press_keys", "keys"):
            if (
                not isinstance(value, list)
                or len(value) > 32
                or any(
                    not isinstance(k, str)
                    or not k
                    or len(k) > 32
                    or not k.replace("_", "").isalnum()
                    for k in value
                )
            ):
                raise ValueError("Keys must be a bounded list of key names")
        elif key in ("overwrite", "enter", "shift"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
        elif key in ("num_clicks", "clicks"):
            if (
                type(value) is not int
                or abs(value) > 100
                or (key == "num_clicks" and value < 1)
            ):
                raise ValueError(f"Invalid {key}")
        elif key == "time":
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 30
            ):
                raise ValueError("Wait must be between 0 and 30 seconds")
        elif key in ("button_type", "button"):
            if value not in ("left", "middle", "right"):
                raise ValueError("Invalid mouse button")
        elif key == "text" and node.func.attr == "save_to_knowledge":
            if not isinstance(value, list) or any(
                not isinstance(v, str) for v in value
            ):
                raise ValueError("Notes must be strings")
        elif key in (
            "element_description",
            "starting_description",
            "ending_description",
            "starting_phrase",
            "ending_phrase",
            "app_code",
            "app_or_filename",
            "task",
            "text",
        ):
            optional = key == "task" or (
                key == "element_description" and node.func.attr == "type"
            )
            if value is None and not optional:
                raise ValueError(f"{key} cannot be null")
            if value is not None and (not isinstance(value, str) or len(value) > 32768):
                raise ValueError(f"{key} must be a bounded string")
            required_text = key not in ("task", "text") and value is not None
            if required_text and not value.strip():
                raise ValueError(f"{key} cannot be empty")
    return ActionSpec(node.func.attr, values)


def prepare_action(agent, spec: ActionSpec, obs: dict) -> PreparedAction:
    """Ground once, without performing input or invoking the coding agent."""
    agent.assign_screenshot(obs)
    data = dict(spec.arguments)
    if spec.name == "set_cell_values":
        raise ValueError(
            "Spreadsheet scripting is not supported by the supervised desktop runner"
        )

    def point(description):
        coords = agent.resize_coordinates(agent.generate_coords(description, obs))
        return checked_point(agent, coords)

    if spec.name in ("click", "scroll") or (
        spec.name == "type" and data["element_description"] is not None
    ):
        data["point"] = point(data["element_description"])
    elif spec.name == "drag_and_drop":
        data["start"] = point(data["starting_description"])
        data["end"] = point(data["ending_description"])
    elif spec.name == "highlight_text_span":
        # OCR coordinates refer to the captured image, not physical desktop pixels.
        from io import BytesIO
        from PIL import Image

        image = Image.open(BytesIO(obs["screenshot"]))
        for key, phrase, alignment in (
            ("start", data["starting_phrase"], "start"),
            ("end", data["ending_phrase"], "end"),
        ):
            x, y = agent.generate_text_coords(phrase, obs, alignment)
            data[key] = checked_point(
                agent,
                [
                    round(x * agent.width / image.width),
                    round(y * agent.height / image.height),
                ],
            )
    return PreparedAction(
        spec.name,
        data,
        json.dumps({"action": spec.name, **data}, ensure_ascii=False, indent=2),
    )


def checked_point(agent, coords):
    if len(coords) != 2 or any(type(v) is not int for v in coords):
        raise ValueError("Grounding must produce two integer coordinates")
    if not (0 <= coords[0] < agent.width and 0 <= coords[1] < agent.height):
        raise ValueError("Grounded point is outside the selected monitor")
    return coords


def execute_action(agent, action: PreparedAction, stop, monitor: dict) -> None:
    """Dispatch known operations. Model-generated text is never exec'd here."""
    import pyautogui as gui

    gui.FAILSAFE = True
    data, name = action.arguments, action.name
    held = []

    def check():
        if stop.is_set():
            raise InterruptedError("Task stopped")

    def wait(seconds):
        if stop.wait(seconds):
            raise InterruptedError("Task stopped")

    def point(key):
        x, y = data[key]
        return x + monitor["left"], y + monitor["top"]

    try:
        check()
        for key in data.get("hold_keys", []):
            gui.keyDown(key)
            held.append(key)
        if name == "click":
            gui.click(
                *point("point"), clicks=data["num_clicks"], button=data["button_type"]
            )
        elif name == "type":
            if "point" in data:
                gui.click(*point("point"))
            check()
            if data["overwrite"]:
                gui.hotkey("command" if agent.platform == "darwin" else "ctrl", "a")
                gui.press("backspace")
            text = data["text"]
            if any(ord(c) > 127 for c in text):
                import pyperclip

                pyperclip.copy(text)
                gui.hotkey("command" if agent.platform == "darwin" else "ctrl", "v")
            else:
                for offset in range(0, len(text), 256):
                    check()
                    gui.write(text[offset : offset + 256])
            if data["enter"]:
                check()
                gui.press("enter")
        elif name in ("drag_and_drop", "highlight_text_span"):
            gui.moveTo(*point("start"))
            check()
            gui.dragTo(*point("end"), duration=1, button=data.get("button", "left"))
        elif name == "scroll":
            gui.moveTo(*point("point"))
            check()
            (gui.hscroll if data["shift"] else gui.scroll)(data["clicks"])
        elif name == "hotkey":
            gui.hotkey(*data["keys"])
        elif name == "hold_and_press":
            for key in data["press_keys"]:
                check()
                gui.press(key)
        elif name == "wait":
            wait(data["time"])
        elif name == "save_to_knowledge":
            agent.notes.extend(data["text"])
        elif name == "call_code_agent":
            # The complete coding-agent invocation is the approved action.
            agent.call_code_agent(data["task"])
        elif name in ("open", "switch_applications"):
            target = data.get("app_or_filename", data.get("app_code"))
            if name == "switch_applications" and agent.platform == "windows":
                from pywinauto import Desktop

                windows = [
                    window
                    for window in Desktop(backend="win32").windows()
                    if target.casefold() in window.window_text().casefold()
                ]
                if not windows:
                    raise ValueError(f"No open window matches {target!r}")
                check()
                windows[0].restore()
                windows[0].set_focus()
                return
            if agent.platform == "windows":
                gui.hotkey("win", "r")
            elif agent.platform == "darwin":
                gui.hotkey("command", "space")
            else:
                gui.press("win")
            wait(0.5)
            if any(ord(char) > 127 for char in target):
                import pyperclip

                pyperclip.copy(target)
                gui.hotkey("command" if agent.platform == "darwin" else "ctrl", "v")
            else:
                gui.write(target)
            check()
            gui.press("enter")
            wait(1)
        else:
            raise ValueError(f"Unsupported executable action: {name}")
    finally:
        # Safety aborts must not leave modifier keys or a drag held down.
        previous = gui.FAILSAFE
        gui.FAILSAFE = False
        try:
            for key in reversed(held):
                gui.keyUp(key)
            if name in ("drag_and_drop", "highlight_text_span"):
                gui.mouseUp()
        finally:
            gui.FAILSAFE = previous
