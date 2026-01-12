"""Safe runner for Agent-S (s3) that avoids real LLM calls and UI actions.

This script injects a FakeAgent and patches pyautogui.screenshot to return a blank image.
It calls cli_app.run_agent once so you can inspect behavior without side effects.
"""
from PIL import Image
import pyautogui

from gui_agents.s3 import cli_app


class FakeAgent:
    def reset(self):
        pass

    def predict(self, instruction: str, observation: dict):
        print("[FakeAgent] predict called. instruction=", instruction)
        # Return an info dict and a list of code strings. Use 'done' to stop gracefully.
        return {"reflection": "fake", "executor_plan": "none"}, ["done"]


def fake_screenshot():
    # return a small white image to simulate a screenshot
    return Image.new("RGB", (640, 360), color=(255, 255, 255))


def main():
    # Patch screenshot to avoid real screen capture
    pyautogui.screenshot = fake_screenshot

    agent = FakeAgent()
    # scaled width/height used by run_agent (choose small values for quick run)
    scaled_w, scaled_h = 320, 180

    print("Starting safe agent run (FakeAgent). You should see one step and immediate completion.")
    # For automated safe run disable interactive confirmation
    cli_app.run_agent(agent, "Test instruction: do nothing", scaled_w, scaled_h, require_exec_confirmation=False)
    print("Safe run finished.")


if __name__ == "__main__":
    main()
