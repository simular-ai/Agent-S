"""Run Agent-S to open Calculator once (real execution).

This script uses a small agent that requests the Calculator to be opened once via
`subprocess.Popen(['calc.exe'])`. It patches screenshots to avoid heavy screen
captures and then restores original functions.

CAUTION: This will open Calculator on your machine once. You've already confirmed.
"""
import subprocess
import platform
from PIL import Image
import pyautogui

from gui_agents.s3 import cli_app


class RealOpenCalcAgent:
    def __init__(self):
        self.called = False

    def reset(self):
        self.called = False

    def predict(self, instruction: str, observation: dict):
        print(f"[RealOpenCalcAgent] predict called. instruction= {instruction!r}")
        if not self.called:
            self.called = True
            # Use subprocess to open Calculator on Windows
            if platform.system().lower() == "windows":
                code = "import subprocess; subprocess.Popen(['calc.exe'])"
            elif platform.system().lower() == "darwin":
                code = "import subprocess; subprocess.Popen(['open', '-a', 'Calculator'])"
            else:
                # Try common calculator commands for Linux
                code = "import subprocess; subprocess.Popen(['gnome-calculator'])"
            return {"reflection": "open calc"}, [code]
        else:
            # Signal done on next call so run_agent will stop
            return {"reflection": "done"}, ["done"]


def fake_screenshot():
    return Image.new("RGB", (800, 600), color=(255, 255, 255))


def main():
    # Patch screenshot to avoid heavy or permission-requiring screen capture
    original_screenshot = pyautogui.screenshot
    pyautogui.screenshot = fake_screenshot

    try:
        agent = RealOpenCalcAgent()
        scaled_w, scaled_h = 320, 180
        print("Running agent to open Calculator (will perform one real open).")
        # User has already confirmed; disable interactive confirmation for this run
        cli_app.run_agent(agent, "Open calculator", scaled_w, scaled_h, require_exec_confirmation=False)
        print("Agent run finished.")
    finally:
        pyautogui.screenshot = original_screenshot


if __name__ == "__main__":
    main()
