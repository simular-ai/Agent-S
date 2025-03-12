import argparse
import datetime
import io
import logging
import os
import platform
import sys
import time

import pyautogui

if platform.system() == "Darwin":
    current_platform = "macos"
elif platform.system() == "Linux":
    current_platform = "ubuntu"
elif platform.system() == "Windows":
    current_platform = "windows"
else:
    raise ValueError("Unsupported platform")

from gui_agents.s2.agents.grounding import OSWorldACI
from gui_agents.s2.agents.agent_s import GraphSearchAgent

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def show_permission_dialog(code: str, action_description: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def run_agent(agent, instruction: str):
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for _ in range(15):
        # Get screen shot using pyautogui.
        # Take a screenshot
        screenshot = pyautogui.screenshot()

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Get next action code from the agent
        info, code = agent.predict(instruction=instruction, observation=obs)

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            agent.update_narrative_memory(traj)
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Ask for permission before executing
            exec(code[0])
            time.sleep(1.0)

            # Update task and subtask trajectories and optionally the episodic memory
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["executor_plan"]
            )
            subtask_traj = agent.update_episodic_memory(info, subtask_traj)


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphSearchAgent with specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Specify the model to use (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--endpoint_provider",
        type=str,
        default="huggingface",
        help="Specify the endpoint provider (e.g., huggingface, sagemaker)",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="",
        help="Specify the endpoint URL to your HuggingFace Inference Endpoint URL or SageMaker Endpoint URL.",
    )
    args = parser.parse_args()

    grounding_agent = OSWorldACI(
        platform=current_platform,
        endpoint_provider=args.endpoint_provider,
        endpoint_url=args.endpoint_url,
    )

    while True:
        query = input("Query: ")
        if "gpt" in args.model:
            engine_type = "openai"
        elif "claude" in args.model:
            engine_type = "anthropic"
        engine_params = {
            "engine_type": engine_type,
            "model": args.model,
        }

        agent = GraphSearchAgent(
            engine_params,
            grounding_agent,
            platform=current_platform,
            action_space="pyautogui",
            observation_type="mixed",
        )

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
