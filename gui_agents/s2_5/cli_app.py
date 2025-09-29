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

from gui_agents.s2_5.agents.grounding import OSWorldACI
from gui_agents.s2_5.agents.agent_s import AgentS2_5
from gui_agents.s2_5.agents.accessibility_agent import AccessibilityConfig

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
            return msvcrt.getch().decode('utf-8', errors='ignore')
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
    
    # Start accessibility session if enabled
    accessibility_session_id = None
    if hasattr(agent, 'accessibility_agent') and agent.accessibility_agent:
        accessibility_session_id = agent.start_accessibility_session(
            f"Task: {instruction}"
        )
        print(f"🔍 Started accessibility session: {accessibility_session_id}")
    
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
        
        # Display accessibility information if available
        if "accessibility_violations" in info and info["accessibility_violations"]:
            violations_count = len(info["accessibility_violations"])
            print(f"⚠️  Found {violations_count} accessibility violations")
            
            # Show critical violations
            critical_violations = [
                v for v in info["accessibility_violations"] 
                if v.get("severity") == "critical"
            ]
            if critical_violations:
                print(f"🚨 Critical accessibility issues:")
                for violation in critical_violations[:3]:  # Show first 3
                    print(f"   - {violation.get('description', 'Unknown violation')}")
        
        if "accessibility_score" in info:
            score = info["accessibility_score"]
            if score < 70:
                print(f"📊 Accessibility Score: {score:.1f}/100 (Needs Improvement)")
            elif score < 90:
                print(f"📊 Accessibility Score: {score:.1f}/100 (Good)")
            else:
                print(f"📊 Accessibility Score: {score:.1f}/100 (Excellent)")

        if "done" in code[0].lower() or "fail" in code[0].lower():
            # End accessibility session if active
            if accessibility_session_id and hasattr(agent, 'end_accessibility_session'):
                session_summary = agent.end_accessibility_session()
                if session_summary:
                    print("\n🔍 Accessibility Testing Complete")
                    print(f"   - Violations Found: {session_summary['violations_found']}")
                    print(f"   - Compliance Score: {session_summary['compliance_score']:.1f}/100")
                    print(f"   - Keyboard Tests: {session_summary['keyboard_tests_run']}")
                    print(f"   - Evidence Captured: {session_summary['evidence_captured']}")
                    
                    # Generate and display report paths
                    report_paths = agent.generate_accessibility_report()
                    if report_paths:
                        print(f"   - Reports Generated:")
                        for report_type, path in report_paths.items():
                            print(f"     • {report_type.upper()}: {path}")
                    
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

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

            # Update task and subtask trajectories
            if "reflection" in info and "executor_plan" in info:
                traj += (
                    "\n\nReflection:\n"
                    + str(info["reflection"])
                    + "\n\n----------------------\n\nPlan:\n"
                    + info["executor_plan"]
                )


def main():
    parser = argparse.ArgumentParser(description="Run AgentS2_5 with specified model.")
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
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)"
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

    # AgentS2_5 specific arguments
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
    
    # Accessibility testing arguments
    parser.add_argument(
        "--enable_accessibility",
        action="store_true",
        default=False,
        help="Enable accessibility and 508 compliance testing",
    )
    parser.add_argument(
        "--accessibility_monitoring",
        action="store_true",
        default=True,
        help="Enable real-time accessibility monitoring",
    )
    parser.add_argument(
        "--accessibility_keyboard_tests",
        action="store_true",
        default=True,
        help="Enable keyboard navigation testing",
    )
    parser.add_argument(
        "--accessibility_compliance_level",
        type=str,
        default="AA",
        choices=["AA", "AAA"],
        help="WCAG compliance level to test against (AA or AAA)",
    )
    parser.add_argument(
        "--accessibility_screenshots",
        action="store_true",
        default=True,
        help="Capture screenshots for accessibility violations",
    )
    parser.add_argument(
        "--accessibility_reports_dir",
        type=str,
        default="accessibility_reports",
        help="Directory to save accessibility reports",
    )

    args = parser.parse_args()

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
        "temperature": getattr(args, 'model_temperature', None),
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

    grounding_agent = OSWorldACI(
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )

    # Initialize accessibility configuration if enabled
    accessibility_config = None
    if args.enable_accessibility:
        accessibility_config = AccessibilityConfig(
            enable_real_time_monitoring=args.accessibility_monitoring,
            enable_keyboard_testing=args.accessibility_keyboard_tests,
            enable_compliance_checking=True,
            enable_violation_detection=True,
            capture_screenshots=args.accessibility_screenshots,
            compliance_level=args.accessibility_compliance_level,
            report_dir=args.accessibility_reports_dir
        )
        print("🔍 Accessibility testing enabled")
        print(f"   - Compliance Level: WCAG {args.accessibility_compliance_level}")
        print(f"   - Real-time Monitoring: {args.accessibility_monitoring}")
        print(f"   - Keyboard Testing: {args.accessibility_keyboard_tests}")
        print(f"   - Screenshot Capture: {args.accessibility_screenshots}")
        print(f"   - Reports Directory: {args.accessibility_reports_dir}")

    agent = AgentS2_5(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
        enable_accessibility=args.enable_accessibility,
        accessibility_config=accessibility_config,
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
