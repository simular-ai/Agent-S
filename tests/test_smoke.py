import importlib
import io
import sys

from PIL import Image


# ---- Insert lightweight dummy modules to avoid heavy external deps at import time ----
class DummyPytesseractModule:
    Output = type("Output", (), {})()

    @staticmethod
    def image_to_data(image, output_type=None):
        # Return minimal dict expected by grounding.get_ocr_elements
        return {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "block_num": [],
        }


sys.modules.setdefault("pytesseract", DummyPytesseractModule)


class DummyPyAutoGUI:
    def size(self):
        return (100, 100)

    def screenshot(self):
        return Image.new("RGB", (100, 100))

    def press(self, *args, **kwargs):
        pass

    def click(self, *args, **kwargs):
        pass

    def hotkey(self, *args, **kwargs):
        pass


sys.modules.setdefault("pyautogui", DummyPyAutoGUI())

# ---- Monkeypatch LMMAgent to avoid external LLM calls ----
import gui_agents.s3.core.mllm as mllm  # noqa: E402


class FakeLMMAgent:
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        self.messages = []
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, prompt):
        self.system_prompt = prompt

    def add_message(self, text_content=None, image_content=None, role=None, **kwargs):
        self.messages.append(
            {
                "role": role or "user",
                "content": [{"type": "text", "text": text_content}],
            }
        )

    def get_response(self, *args, **kwargs):
        # Return a response that contains a single valid action: agent.wait
        return "<thoughts>thinking</thoughts><answer>```python\nagent.wait(1.333)\n```</answer>"


mllm.LMMAgent = FakeLMMAgent
import gui_agents.s3.agents.code_agent as _code_agent
_code_agent.LMMAgent = FakeLMMAgent
import gui_agents.s3.agents.grounding as _grounding
_grounding.LMMAgent = FakeLMMAgent


def _create_screenshot_bytes():
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_agent_smoke_flow():
    from gui_agents.s3.agents.agent_s import AgentS3
    from gui_agents.s3.agents.grounding import OSWorldACI

    screenshot = _create_screenshot_bytes()

    grounding = OSWorldACI(
        env=None,
        platform="linux",
        engine_params_for_generation={"engine_type": "mock"},
        engine_params_for_grounding={
            "engine_type": "mock",
            "grounding_width": 100,
            "grounding_height": 100,
        },
        width=100,
        height=100,
    )

    agent = AgentS3(
        worker_engine_params={"engine_type": "mock", "model": "gpt-4o"},
        grounding_agent=grounding,
        platform="linux",
    )

    info, actions = agent.predict(
        instruction="Wait a bit", observation={"screenshot": screenshot}
    )

    assert isinstance(actions, list) and len(actions) > 0
    assert "time.sleep" in actions[0]


def test_cli_help_runs_ok():
    # ensure cli module can be imported with dummy pyautogui in sys.modules
    cli = importlib.import_module("gui_agents.s3.cli_app")

    # Running help should exit with code 0
    import sys as _sys

    prev_argv = _sys.argv.copy()
    try:
        _sys.argv = ["agent_s", "--help"]
        try:
            cli.main()
        except SystemExit as e:
            assert e.code == 0
    finally:
        _sys.argv = prev_argv
