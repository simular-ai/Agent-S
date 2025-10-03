from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_formatted,
    split_thinking_response,
    compress_image,
)
from gui_agents.s3.utils.formatters import (
    THOUGHTS_ANSWER_TAG_FORMATTER,
)
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Dict
import base64
import cv2
import numpy as np


class BehaviorNarrator:
    def __init__(self, engine_params):
        self.judge_agent = LMMAgent(engine_params=engine_params)

    @staticmethod
    def extract_mouse_action(action: str) -> list[str]:
        mouse_actions = []
        for sub_action in action.split(";"):
            sub_action = sub_action.strip()
            if (
                sub_action.startswith("pyautogui.click")
                or sub_action.startswith("pyautogui.moveTo")
                or sub_action.startswith("pyautogui.dragTo")
            ):
                mouse_actions.append(sub_action)
        return mouse_actions

    @staticmethod
    def mark_action(mouse_actions: list[str], img: Image):
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(25)

        drag_start_width, drag_start_height = None, None

        for mouse_action in mouse_actions:
            width, height = mouse_action.split("(")[1].strip(")").split(", ")[:2]
            width, height = int(width), int(height)

            # Clamp coordinates within bounds
            width = max(0, min(img.width - 1, width))
            height = max(0, min(img.height - 1, height))

            def place_text(label, color):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = (
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                )  # Measure text size
                offset_x, offset_y = -5, 5  # Default offset
                if width + offset_x + text_w > img.width:  # Out of bounds on right
                    offset_x = -text_w - 5
                if height + offset_y + text_h > img.height:  # Out of bounds on bottom
                    offset_y = -text_h - 5
                if width + offset_x < 0:  # Out of bounds on left
                    offset_x = 5
                if height + offset_y < 0:  # Out of bounds on top
                    offset_y = 5
                draw.text(
                    (width + offset_x, height + offset_y), label, fill=color, font=font
                )

            if mouse_action.startswith("pyautogui.click"):
                draw.circle((width, height), radius=3, fill=(255, 0, 0))
                place_text("Click", (255, 0, 0))
            if mouse_action.startswith("pyautogui.moveTo"):
                draw.circle((width, height), radius=3, fill=(0, 0, 255))
                place_text("MoveTo", (0, 0, 255))
                drag_start_height, drag_start_width = height, width
            if mouse_action.startswith("pyautogui.dragTo"):
                draw.line(
                    [(drag_start_width, drag_start_height), (width, height)],
                    fill=(0, 255, 0),
                    width=2,
                )
                draw.circle((width, height), radius=3, fill=(0, 255, 0))
                place_text("DragTo", (0, 255, 0))

    @staticmethod
    def get_mouse_action_representation(mouse_actions: list[str]) -> str:
        """
        Returns a string representation of the mouse action for the given action.
        """
        assert (
            len(mouse_actions) <= 2
        ), f"Multiple mouse action types found: {mouse_actions}"
        if len(mouse_actions) == 1:
            action = mouse_actions[0]
            if action.startswith("pyautogui.click"):
                return "The red circle labeled 'Click' marks the position where the mouse was clicked."
            elif action.startswith("pyautogui.moveTo"):
                return "The blue circle labeled 'MoveTo' marks the position where the mouse was moved to."
            else:
                raise ValueError(f"Unknown single action type: {action}")
        else:
            assert mouse_actions[0].startswith("pyautogui.moveTo") and mouse_actions[
                1
            ].startswith("pyautogui.dragTo")
            return "The blue circle labeled 'MoveTo' marks the starting position of the mouse.\nThe green circle labeled 'DragTo' marks the ending position.\nThe green line illustrates the mouse's drag path."

    @staticmethod
    def get_zoomed_image(
        image_bytes: bytes,
        x: int,
        y: int,
        width: int = 300,
        height: int = 300,
        upscaling: bool = False,
        scale: int = 4,
        add_bounding_box: bool = False,
    ) -> bytes:
        """Returns a zoomed image centered around (x, y) coordinates.

        Args:
            image_bytes (bytes): The original image in bytes.
            x (int): The x-coordinate of the center point.
            y (int): The y-coordinate of the center point.
            width (int): The width of the zoomed area.
            height (int): The height of the zoomed area.
            padding (int): Extra padding around the zoomed area.
            upscaling (bool): Whether to upscale and enhance the zoomed image.
            scale (int): The upscaling factor if upscaling is True.
            add_bounding_box (bool): Whether to add a bounding box around the zoomed area in the original image.

        Returns:
            bytes: The zoomed image in bytes.
            bytes: The original image with bounding box in bytes (if add_bounding_box is True). Otherwise, returns original bytes.
        """
        # Find zoom dimensions
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        cx, cy = x - width // 2, y - height // 2  # Center coordinates
        W, H = img.size
        left = min(max(cx, 0), W - width)
        top = min(max(cy, 0), H - height)
        right = left + width
        bottom = top + height
        zoomed_img = img.crop((left, top, right, bottom))
        # Add noticeable bounding box to original image
        if add_bounding_box:
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            original_with_box_bytes = compress_image(
                image=draw_img
            )  # Compress to reduce size
        else:
            original_with_box_bytes = image_bytes
        if upscaling:
            # Upscale and enhance zoomed image
            zoomed_img = cv2.cvtColor(
                np.array(zoomed_img), cv2.COLOR_RGB2BGR
            )  # PIL -> OpenCV
            zoomed_img = cv2.resize(
                zoomed_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
            )
            zoomed_img = cv2.fastNlMeansDenoisingColored(
                zoomed_img, None, 5, 5, 7, 21
            )  # light denoise (helps with JPEG speckle)
            zoomed_img = Image.fromarray(
                cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB)
            )  # OpenCV -> PIL
        zoomed_img_bytes = compress_image(image=zoomed_img)  # Compress to reduce size
        return zoomed_img_bytes, original_with_box_bytes

    def judge(
        self,
        screenshot_num: int,
        before_img_bytes: bytes,
        after_img_bytes: bytes,
        pyautogui_action: str,
    ) -> Dict[str, str]:
        if pyautogui_action == "DONE":
            return {
                "fact_thoughts": "The agent has indicated that it is done with the task.",
                "fact_answer": "The agent has indicated that it is done with the task.",
            }
        elif pyautogui_action == "FAIL":
            return {
                "fact_thoughts": "The agent has indicated that it is impossible to proceed further with the task.",
                "fact_answer": "The agent has indicated that it is impossible to proceed further with the task.",
            }
        # Prepare ANNOTATED BEFORE image
        mouse_actions = BehaviorNarrator.extract_mouse_action(pyautogui_action)
        before_img = Image.open(BytesIO(before_img_bytes))
        BehaviorNarrator.mark_action(mouse_actions, before_img)
        out_buffer = BytesIO()
        before_img.save(out_buffer, format="PNG")
        marked_before_img_bytes = out_buffer.getvalue()
        marked_before_img_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(marked_before_img_bytes).decode('utf-8')}",
                "detail": "high",
            },
        }
        if mouse_actions:
            coords = mouse_actions[-1].split("(")[1].strip(")").split(", ")
            x, y = int(coords[0]), int(coords[1])
            zoomed_after_img_bytes, marked_after_img_bytes = (
                BehaviorNarrator.get_zoomed_image(
                    image_bytes=after_img_bytes,
                    x=x,
                    y=y,
                    width=300,
                    height=300,
                    scale=4,
                    upscaling=True,
                    add_bounding_box=True,
                )
            )
            after_img_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(marked_after_img_bytes).decode('utf-8')}",
                    "detail": "high",
                },
            }
            zoomed_after_img_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(zoomed_after_img_bytes).decode('utf-8')}",
                    "detail": "high",
                },
            }
        else:
            after_img_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(after_img_bytes).decode('utf-8')}",
                    "detail": "high",
                },
            }
            zoomed_after_img_message = None

        fact_message = [
            {
                "role": "system",
                "content": PROCEDURAL_MEMORY.BEHAVIOR_NARRATOR_SYSTEM_PROMPT,
            }
        ]
        fact_message_content = [
            {"type": "text", "text": "BEFORE:"},
            marked_before_img_message,
            {"type": "text", "text": f"Agent Action: {pyautogui_action}"},
            {"type": "text", "text": "AFTER:"},
            after_img_message,
        ]
        if zoomed_after_img_message:
            fact_message_content += [
                {"type": "text", "text": "ZOOMED AFTER:"},
                zoomed_after_img_message,
            ]
        fact_message += [{"role": "user", "content": fact_message_content}]
        fact_response = call_llm_formatted(
            self.judge_agent,
            [THOUGHTS_ANSWER_TAG_FORMATTER],
            messages=fact_message,
            temperature=0.0,
        )
        fact_answer, fact_thoughts = split_thinking_response(fact_response)

        result = {
            "fact_thoughts": fact_thoughts,
            "fact_answer": f"Fact Caption from Screenshot {screenshot_num}: {fact_answer}",
        }
        return result
