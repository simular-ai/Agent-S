import os
import base64
from typing import List, Tuple, Optional, List

from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import call_llm_formatted, split_thinking_response


def get_final_screenshot_file(task_dir: str) -> str:
    """Get the final screenshot file name from a task directory."""
    screenshot_files = []
    for filename in os.listdir(task_dir):
        if filename.startswith("step_") and filename.endswith(".png"):
            screenshot_files.append(filename)

    if not screenshot_files:
        return "step_0.png"  # fallback

    # Sort by step number and get the last one
    def extract_step_num(filename):
        try:
            return int(filename.split("_")[1].split(".")[0])
        except:
            return 0

    screenshot_files.sort(key=extract_step_num)
    return screenshot_files[-1]


def image_to_openai_message_format(
    image_path: str, caption: str = ""
) -> Optional[dict]:
    """Convert an image file to OpenAI message format."""
    if not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        content = []
        if caption:
            content.append({"type": "text", "text": caption})

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "high",
                },
            }
        )

        return {"role": "user", "content": content}
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


class ComparativeJudge:
    def __init__(self, engine_params):
        self.judge_agent = LMMAgent(engine_params=engine_params)

    def judge(
        self,
        task_description: str,
        task: str,
        result_dirs: List[str],
        all_fact_captions: List[List[str]],
    ) -> Tuple[str, str, Optional[str]]:
        """
        Fact captions + initial/final screenshots judging.
        Pipeline: use provided fact captions → include initial/final screenshots → judge.
        """
        num_trajectories = len(result_dirs)
        system_prompt = PROCEDURAL_MEMORY.VLM_EVALUATOR_PROMPT_COMPARATIVE_BASELINE
        system_prompt = system_prompt.replace(
            "<TASK_DESCRIPTION_INPUT>", task_description
        )
        system_prompt = system_prompt.replace(
            "<NUMBER OF TRAJECTORIES>", str(num_trajectories)
        )

        messages = [{"role": "system", "content": system_prompt}]

        for i, (result_dir, fact_captions) in enumerate(
            zip(result_dirs, all_fact_captions)
        ):
            task_dir = os.path.join(result_dir, task.split("/")[0], task.split("/")[1])
            result_initial_screenshot = os.path.join(task_dir, "step_0.png")
            result_final_screenshot = os.path.join(
                task_dir, get_final_screenshot_file(task_dir)
            )
            initial_screenshot_message = image_to_openai_message_format(
                result_initial_screenshot, caption=f"Initial screenshot of result{i+1}"
            )
            final_screenshot_message = image_to_openai_message_format(
                result_final_screenshot, caption=f"Final screenshot of result{i+1}"
            )
            if (
                initial_screenshot_message is not None
                and final_screenshot_message is not None
            ):
                messages.append(initial_screenshot_message)
                messages.append(final_screenshot_message)
            if fact_captions:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Fact captions for Trajectory {i+1}:",
                            }
                        ]
                        + [
                            {"type": "text", "text": caption}
                            for caption in fact_captions
                        ],
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please evaluate the {num_trajectories} trajectories based on the criteria provided in the system prompt.",
                    }
                ],
            }
        )

        response = call_llm_formatted(self.judge_agent, [], messages=messages)
        answer, thoughts = split_thinking_response(response)

        try:
            judge_choice = int(answer)
            if 1 <= judge_choice <= num_trajectories:
                selected_trajectory = result_dirs[judge_choice - 1]
            else:
                selected_trajectory = None
        except ValueError:
            selected_trajectory = None

        return answer, thoughts, selected_trajectory
