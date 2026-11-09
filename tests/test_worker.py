import io

from PIL import Image

from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.core import mllm as mllm_mod


# Monkeypatch LMMAgent used in Worker via module replacement
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
        return "<thoughts>thinking</thoughts><answer>```python\nagent.wait(0.5)\n```</answer>"


mllm_mod.LMMAgent = FakeLMMAgent
import gui_agents.s3.agents.code_agent as _code_agent
_code_agent.LMMAgent = FakeLMMAgent
import gui_agents.s3.agents.grounding as _grounding
_grounding.LMMAgent = FakeLMMAgent


def _create_screenshot():
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_worker_generate_next_action():
    screenshot = _create_screenshot()
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
        instruction="Wait small", observation={"screenshot": screenshot}
    )

    assert isinstance(actions, list)
    assert len(actions) == 1
    assert "time.sleep" in actions[0] or "wait" in actions[0]
