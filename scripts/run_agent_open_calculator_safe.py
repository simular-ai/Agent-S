"""Run Agent-S in a safe simulated mode to test 'open calculator'.

This script injects a FakeAgent that returns code to open the calculator using
platform-appropriate commands. It stubs out pyautogui, os.system, and subprocess
calls to prevent any real side-effects and prints the actions instead.
"""
import platform
import pyautogui
import os
import subprocess
from PIL import Image

from gui_agents.s3 import cli_app


class FakeOpenCalcAgent:
    def reset(self):
        pass

    def predict(self, instruction: str, observation: dict):
        print("[FakeOpenCalcAgent] predict called. instruction=", instruction)
        system = platform.system().lower()
        if system == "windows":
            # Typical sequence using Windows start menu via pyautogui
            code = "import pyautogui, time; pyautogui.hotkey('win'); time.sleep(0.5); pyautogui.typewrite('Calculator'); pyautogui.press('enter')"
        elif system == "darwin":
            code = "import os; os.system('open -a Calculator')"
        else:
            # Linux: try common calculator commands
            code = "import os; os.system('gnome-calculator &')"
        # Return a final 'done' to make run_agent stop after executing this code
        return {"reflection": "try open calc"}, [code, "done"]


# Stubs to prevent real side-effects
_original_screenshot = pyautogui.screenshot


def fake_screenshot():
    return Image.new("RGB", (800, 600), color=(255, 255, 255))


def stubbed_hotkey(*args, **kwargs):
    print(f"[stub] pyautogui.hotkey called with {args} {kwargs}")


def stubbed_typewrite(text, **kwargs):
    print(f"[stub] pyautogui.typewrite called with {text!r} {kwargs}")


def stubbed_press(key, **kwargs):
    print(f"[stub] pyautogui.press called with {key!r} {kwargs}")


def stubbed_os_system(cmd):
    print(f"[stub] os.system would run: {cmd!r}")
    return 0


def stubbed_subprocess_run(*args, **kwargs):
    print(f"[stub] subprocess.run called: args={args} kwargs={kwargs}")
    class R:
        returncode = 0
    return R()


def main():
    # Patch screenshot and UI/system functions
    pyautogui.screenshot = fake_screenshot
    pyautogui.hotkey = stubbed_hotkey
    pyautogui.typewrite = stubbed_typewrite
    pyautogui.press = stubbed_press
    os_system_orig = os.system
    subprocess_run_orig = subprocess.run
    os.system = stubbed_os_system
    subprocess.run = stubbed_subprocess_run

    try:
        agent = FakeOpenCalcAgent()
        scaled_w, scaled_h = 320, 180
        print("Starting simulated agent run to open calculator. No real actions will be performed.")
        # run_agent will call predict, then exec the first returned code string
        # For automated simulation, disable interactive confirmation
        cli_app.run_agent(agent, "Open calculator", scaled_w, scaled_h, require_exec_confirmation=False)
        print("Simulated run finished.")
    finally:
        # Restore patched functions just in case
        pyautogui.screenshot = _original_screenshot
        os.system = os_system_orig
        subprocess.run = subprocess_run_orig


if __name__ == "__main__":
    main()
