import argparse
import datetime
import io
import logging
import os
import platform
import pyautogui
import signal
import sys
import time

from PIL import Image

from gui_agents.s2.agents.grounding import OSWorldACI
from gui_agents.s2.agents.agent_s import AgentS2

current_platform = platform.system().lower()

# Global flag to track pause state for debugging
paused = False


def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        # Import termios and tty on Unix-like systems
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Windows fallback
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except:
        return input()  # Fallback for non-terminal environments


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\n🔸 Agent-S Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\n🛑 Exiting Agent-S...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\n▶️  Resuming Agent-S workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Agent-S...")
                sys.exit(0)
    else:
        # Already paused, second Ctrl+C means quit
        print("\n\n🛑 Exiting Agent-S...")
        sys.exit(0)


# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

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


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(agent, instruction: str, scaled_width: int, scaled_height: int):
    global paused
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for step in range(15):
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)

        # Get screen shot using pyautogui
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Check again for pause state before prediction
        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/15: Getting next action from agent...")

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
            print("⏳ Agent requested wait...")
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Check for pause state before execution
            while paused:
                time.sleep(0.1)

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
    parser = argparse.ArgumentParser(description="Run AgentS2 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Specify the model to use (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )

    # Grounding model config option 1: API based
    parser.add_argument(
        "--grounding_model_provider",
        type=str,
        default="anthropic",
        help="Specify the provider to use for the grounding model (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--grounding_model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Specify the grounding model to use (e.g., claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--grounding_model_resize_width",
        type=int,
        default=1366,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_model_resize_height",
        type=int,
        default=None,
        help="Height of screenshot image after processor rescaling",
    )

    # Grounding model config option 2: Self-hosted endpoint based
    parser.add_argument(
        "--endpoint_provider",
        type=str,
        default="",
        help="Specify the endpoint provider for your grounding model, only HuggingFace TGI support for now",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="",
        help="Specify the endpoint URL for your grounding model",
    )
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )

    parser.add_argument(
        "--embedding_engine_type",
        type=str,
        default="openai",
        help="Specify the embedding engine type (supports openai, gemini)",
    )

    args = parser.parse_args()
    assert (
        args.grounding_model_provider and args.grounding_model
    ) or args.endpoint_url, "Error: No grounding model was provided. Either provide an API based model, or a self-hosted HuggingFace endpoint"

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=2400
    )

    # Load the general engine params
    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
    }

    # Load the grounding engine from a HuggingFace TGI endpoint
    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "base_url": args.endpoint_url,
            "api_key": args.endpoint_api_key,
        }
    else:
        grounding_height = args.grounding_model_resize_height
        # If not provided, use the aspect ratio of the screen to compute the height
        if grounding_height is None:
            grounding_height = (
                screen_height * args.grounding_model_resize_width / screen_width
            )

        engine_params_for_grounding = {
            "engine_type": args.grounding_model_provider,
            "model": args.grounding_model,
            "grounding_width": args.grounding_model_resize_width,
            "grounding_height": grounding_height,
        }

    grounding_agent = OSWorldACI(
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )

    agent = AgentS2(
        engine_params,
        grounding_agent,
        platform=current_platform,
        action_space="pyautogui",
        observation_type="mixed",
        search_engine=None,
        embedding_engine_type=args.embedding_engine_type,
    )

    while True:
        query = input("Query: ")

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query, scaled_width, scaled_height)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
