import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gui_agents.core.AgentS import GraphSearchAgent
from gui_agents.aci.LinuxOSACI import LinuxACI

app = FastAPI()

engine_params = {
    "engine_type": "openai",
    "model": "gpt-4o",
}
grounding_agent = LinuxACI()

agent = GraphSearchAgent(
    engine_params,
    grounding_agent,
    platform="ubuntu",
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
