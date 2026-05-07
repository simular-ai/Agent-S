import argparse
import datetime
import io
import logging
import os
import platform
import signal
import sys
import time

from PIL import Image

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.utils.local_env import LocalEnv
from gui_agents.s3.utils.docker_env import DockerEnv
from gui_agents.s3.utils.kvm_env import KvmEnv

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


def run_agent(agent, instruction: str, scaled_width: int, scaled_height: int, code_env=None):
    global paused
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    print(f"[STEPTIME 0] run_agent_start={time.monotonic():.3f}")
    for step in range(15):
        _t_step = time.monotonic()
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)
        # Get screenshot from the env backend if available, else pyautogui (host)
        _t_ss0 = time.monotonic()
        if code_env is not None and hasattr(code_env, "controller") and hasattr(code_env.controller, "screenshot"):
            png_bytes = code_env.controller.screenshot()
            screenshot = Image.open(io.BytesIO(png_bytes))
        else:
            import pyautogui  # host-side screenshot needs a display
            screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)
        _t_ss1 = time.monotonic()
        print(f"[STEPTIME {step+1}] screenshot={(_t_ss1-_t_ss0)*1000:.0f}ms")

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
        _t_p0 = time.monotonic()
        info, code = agent.predict(instruction=instruction, observation=obs)
        _t_p1 = time.monotonic()
        print(f"[STEPTIME {step+1}] predict={(_t_p1-_t_p0)*1000:.0f}ms")

        if "done" in code[0].lower() or "fail" in code[0].lower():
            print(f"[STEPTIME {step+1}] done_path total_step={(time.monotonic()-_t_step)*1000:.0f}ms")
            # Skip the host-side dialog when there's no display (headless / batch
            # runs). zenity otherwise blocks ~25 s before failing.
            host_has_display = bool(os.environ.get("DISPLAY")) or platform.system() == "Darwin"
            if host_has_display:
                _t_dlg0 = time.monotonic()
                if platform.system() == "Darwin":
                    os.system(
                        f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                    )
                elif platform.system() == "Linux":
                    os.system(
                        f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                    )
                print(f"[STEPTIME {step+1}] dialog={(time.monotonic()-_t_dlg0)*1000:.0f}ms")
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

            # Route execution: container/guest if env backend present, else local exec
            _t_e0 = time.monotonic()
            if code_env is not None and hasattr(code_env, "controller") and hasattr(code_env.controller, "run_python_script"):
                # Predicted code uses pyautogui — run it inside the env so it
                # acts on the env's display, not the host's.
                code_env.controller.run_python_script(code[0])
            else:
                exec(code[0])
            _t_e1 = time.monotonic()
            print(f"[STEPTIME {step+1}] exec={(_t_e1-_t_e0)*1000:.0f}ms")
            time.sleep(1.0)
            print(f"[STEPTIME {step+1}] total={(time.monotonic()-_t_step)*1000:.0f}ms")

            # Update task and subtask trajectories
            if "reflection" in info and "executor_plan" in info:
                traj += (
                    "\n\nReflection:\n"
                    + str(info["reflection"])
                    + "\n\n----------------------\n\nPlan:\n"
                    + info["executor_plan"]
                )


def main():
    print(f"[STEPTIME 0] main_start={time.monotonic():.3f}")
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Specify the model to use (e.g., gpt-5-2025-08-07)",
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
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # Grounding model config: Self-hosted endpoint based (required)
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url",
        type=str,
        required=True,
        help="The URL of the grounding model",
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        required=True,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        required=True,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        required=True,
        help="Height of screenshot image after processor rescaling",
    )

    # AgentS3 specific arguments
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=8,
        help="Maximum number of image turns to keep in trajectory",
    )
    parser.add_argument(
        "--enable_reflection",
        action="store_true",
        default=True,
        help="Enable reflection agent to assist the worker agent",
    )
    parser.add_argument(
        "--enable_local_env",
        action="store_true",
        default=False,
        help="Enable local coding environment for code execution (WARNING: Executes arbitrary code locally)",
    )
    parser.add_argument(
        "--env_backend",
        type=str,
        default=None,
        choices=[None, "local", "docker", "podman", "kvm"],
        help="Coding environment backend. Overrides --enable_local_env when set. "
             "'docker'/'podman' require --container; 'kvm' requires --kvm-* flags.",
    )
    parser.add_argument("--container", type=str, default=None,
                        help="Container name/id for docker/podman backend.")
    parser.add_argument("--container_user", type=str, default=None)
    parser.add_argument("--container_workdir", type=str, default=None)
    parser.add_argument("--container_python", type=str, default="python3")
    parser.add_argument("--kvm_transport", type=str, default="ssh", choices=["ssh", "qga"])
    parser.add_argument("--kvm_ssh_host", type=str, default=None)
    parser.add_argument("--kvm_ssh_user", type=str, default=None)
    parser.add_argument("--kvm_ssh_port", type=int, default=22)
    parser.add_argument("--kvm_ssh_key", type=str, default=None)
    parser.add_argument("--kvm_domain", type=str, default=None,
                        help="libvirt domain name for qga transport.")
    parser.add_argument("--kvm_virsh_uri", type=str, default=None)
    parser.add_argument("--kvm_display", type=str, default=":0",
                        help="DISPLAY env var inside the guest (default :0).")
    parser.add_argument("--kvm_xauthority", type=str, default=None,
                        help="XAUTHORITY path inside the guest (e.g. /home/user/.Xauthority).")
    parser.add_argument("--kvm_python", type=str, default="python3",
                        help="Python interpreter inside the guest.")
    parser.add_argument(
        "--task",
        type=str,
        help="The task instruction for Agent-S3 to perform.",
    )

    args = parser.parse_args()

    # Initialize env backend early so we can read its screen size (no host display needed)
    code_env = None
    backend = args.env_backend or ("local" if args.enable_local_env else None)
    if backend == "local":
        print("⚠️  WARNING: Local coding environment enabled. This will execute arbitrary code locally!")
        code_env = LocalEnv()
    elif backend in ("docker", "podman"):
        if not args.container:
            raise SystemExit(f"--env_backend {backend} requires --container")
        runtime = "podman" if backend == "podman" else "docker"
        print(f"Using {runtime} container '{args.container}' as code environment")
        code_env = DockerEnv(
            container=args.container, runtime=runtime,
            user=args.container_user, workdir=args.container_workdir,
            python_bin=args.container_python,
        )
    elif backend == "kvm":
        if args.kvm_transport == "ssh":
            if not (args.kvm_ssh_host and args.kvm_ssh_user):
                raise SystemExit("--kvm_transport ssh requires --kvm_ssh_host and --kvm_ssh_user")
            print(f"Using KVM guest via ssh {args.kvm_ssh_user}@{args.kvm_ssh_host} as code environment")
            code_env = KvmEnv(transport="ssh", ssh_host=args.kvm_ssh_host, ssh_user=args.kvm_ssh_user,
                              ssh_port=args.kvm_ssh_port, ssh_key=args.kvm_ssh_key,
                              python_bin=args.kvm_python,
                              display=args.kvm_display, xauthority=args.kvm_xauthority)
        else:
            if not args.kvm_domain:
                raise SystemExit("--kvm_transport qga requires --kvm_domain")
            print(f"Using KVM guest via QGA on domain '{args.kvm_domain}' as code environment")
            code_env = KvmEnv(transport="qga", domain=args.kvm_domain, virsh_uri=args.kvm_virsh_uri,
                              python_bin=args.kvm_python,
                              display=args.kvm_display, xauthority=args.kvm_xauthority)

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    print(f"[STEPTIME 0] before_screen_size={time.monotonic():.3f}")
    if code_env is not None and hasattr(code_env, "controller") and hasattr(code_env.controller, "screen_size"):
        screen_width, screen_height = code_env.controller.screen_size()
    else:
        import pyautogui
        screen_width, screen_height = pyautogui.size()
    print(f"[STEPTIME 0] after_screen_size={time.monotonic():.3f}")
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=2400
    )

    # Load the general engine params
    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    # Load the grounding engine from a custom endpoint
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    _t_setup0 = time.monotonic()
    grounding_agent = OSWorldACI(
        env=code_env,
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )
    _t_setup1 = time.monotonic()
    print(f"[STEPTIME 0] OSWorldACI_init={(_t_setup1-_t_setup0)*1000:.0f}ms")

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
    )
    print(f"[STEPTIME 0] AgentS3_init={(time.monotonic()-_t_setup1)*1000:.0f}ms total_setup={(time.monotonic()-_t_setup0)*1000:.0f}ms")

    task = args.task

    # handle query from command line
    if isinstance(task, str) and task.strip():
        agent.reset()
        run_agent(agent, task, scaled_width, scaled_height, code_env=code_env)
        return

    while True:
        query = input("Query: ")

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query, scaled_width, scaled_height, code_env=code_env)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
