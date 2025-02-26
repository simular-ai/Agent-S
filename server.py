import asyncio
import platform
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gui_agents.core.AgentS import GraphSearchAgent
from gui_agents.aci.LinuxOSACI import LinuxACI
from gui_agents.aci.MacOSACI import MacOSACI
from gui_agents.aci.WindowsOSACI import WindowsOSACI

app = FastAPI()

engine_params = {
    "engine_type": "openai",
    "model": "gpt-4o",
}

# Determine the operating system and select appropriate ACI
os_name = platform.system().lower()
if os_name == "linux":
    grounding_agent = LinuxACI()
    platform_name = "ubuntu"
elif os_name == "darwin":
    grounding_agent = MacOSACI()
    platform_name = "macos"
elif os_name == "windows":
    grounding_agent = WindowsOSACI()
    platform_name = "windows"
else:
    raise ValueError(f"Unsupported operating system: {os_name}")

agent = GraphSearchAgent(
    engine_params,
    grounding_agent,
    platform=platform_name,
    action_space="pyautogui",
    observation_type="mixed",
)

class InstructionData(BaseModel):
    screenshot: str
    accessibility_tree: str

class CommandRequest(BaseModel):
    obs: InstructionData
    instruction: str


async def stream_code(code: str):
    for line in code.splitlines(keepends=True):
        yield line
        await asyncio.sleep(0.1)

@app.post("/execute")
async def execute_command_stream(cmd: CommandRequest):
    obs = {
        "screenshot": cmd.obs.screenshot,
        "accessibility_tree": cmd.obs.accessibility_tree,
    }
    instruction = cmd.instruction
    info, code = agent.predict(instruction=instruction, observation=obs)

    return StreamingResponse(stream_code(code), media_type="text/plain")
