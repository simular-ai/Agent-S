"""Run Agent-S to open Calculator and compute 45+78 in the Calculator app (real actions).

This will:
- Launch Calculator (subprocess)
- Wait shortly for it to open
- Type '45+78' and press Enter in the Calculator window using pyautogui

CAUTION: This will interact with your desktop and send keystrokes.
"""
import time
import subprocess
import platform
from PIL import Image
import pyautogui

from gui_agents.s3 import cli_app


class RealComputeAgent:
    def __init__(self):
        self.called = False

    def reset(self):
        self.called = False

    def predict(self, instruction: str, observation: dict):
        print(f"[RealComputeAgent] predict called. instruction= {instruction!r}")
        if not self.called:
            self.called = True
            system = platform.system().lower()
            if system == "windows":
                code = (
                    "import subprocess, time, pyautogui;"
                    " subprocess.Popen(['calc.exe']);"
                    " time.sleep(0.8);"
                    " pyautogui.typewrite('45'); pyautogui.press('+'); pyautogui.typewrite('78'); pyautogui.press('enter');"
                )
            elif system == "darwin":
                code = (
                    "import subprocess, time, pyautogui; subprocess.Popen(['open','-a','Calculator']); time.sleep(0.8); pyautogui.typewrite('45'); pyautogui.press('+'); pyautogui.typewrite('78'); pyautogui.press('enter');"
                )
            else:
                code = (
                    "import subprocess, time, pyautogui; subprocess.Popen(['gnome-calculator']); time.sleep(0.8); pyautogui.typewrite('45'); pyautogui.press('+'); pyautogui.typewrite('78'); pyautogui.press('enter');"
                )
            return {"reflection": "open calc and compute"}, [code, "done"]
        else:
            return {"reflection": "done"}, ["done"]


def fake_screenshot():
    return Image.new("RGB", (800, 600), color=(255, 255, 255))


def main():
    # Patch screenshot to avoid heavy or permission-requiring screen capture
    original_screenshot = pyautogui.screenshot
    pyautogui.screenshot = fake_screenshot

    try:
        agent = RealComputeAgent()
        scaled_w, scaled_h = 320, 180
        print("Running agent to open Calculator and compute 45+78 (will perform real actions).")
        # User asked for this; disable interactive confirmation for this run
        cli_app.run_agent(agent, "Open calculator and compute 45+78", scaled_w, scaled_h, require_exec_confirmation=False)
        print("Agent run finished.")
    finally:
        pyautogui.screenshot = original_screenshot


if __name__ == "__main__":
    main()
