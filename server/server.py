import asyncio
import os
import platform
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gui_agents.core.AgentS import GraphSearchAgent
import io
import pyautogui
import time

# Determine the operating system and select appropriate ACI
os_name = platform.system().lower()
if os_name == "linux":
    from gui_agents.aci.LinuxOSACI import LinuxACI, UIElement
    grounding_agent = LinuxACI()
    platform_name = "ubuntu"
elif os_name == "darwin":
    from gui_agents.aci.MacOSACI import MacOSACI, UIElement
    grounding_agent = MacOSACI()
    platform_name = "macos"
elif os_name == "windows":
    from gui_agents.aci.WindowsOSACI import WindowsOSACI, UIElement
    grounding_agent = WindowsOSACI()
    platform_name = "windows"
else:
    raise ValueError(f"Unsupported operating system: {os_name}")

app = FastAPI()

class InstructionData(BaseModel):
    screenshot: str
    accessibility_tree: str

class CommandRequest(BaseModel):
    obs: InstructionData
    instruction: str

class RunRequest(BaseModel):
    model: str
    instruction: str
    api_key: str | None = None

async def stream_code(code: str):
    for line in code.splitlines(keepends=True):
        yield line
        await asyncio.sleep(0.1)

def run_agent(agent: GraphSearchAgent, instruction: str):
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for _ in range(15):

        print("interation", _)

        obs["accessibility_tree"] = UIElement.systemWideElement()

        # Get screen shot using pyautogui.
        # Take a screenshot
        screenshot = pyautogui.screenshot()

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Get next action code from the agent
        info, code = agent.predict(instruction=instruction, observation=obs)

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            agent.update_narrative_memory(traj)
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Ask for permission before executing
            exec(code[0])
            time.sleep(1.0)

            # Update task and subtask trajectories and optionally the episodic memory
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["executor_plan"]
            )
            subtask_traj = agent.update_episodic_memory(info, subtask_traj)

@app.post("/run")
async def run(request: RunRequest):
    if "gpt" in request.model:
        engine_type = "openai"
    elif "claude" in request.model:
        engine_type = "anthropic"

    engine_params = {
        "engine_type": engine_type,
        "model": request.model,
        "api_key": request.api_key,
    }

    print("engine_params", engine_params)

    agent = GraphSearchAgent(
        engine_params,
        grounding_agent,
        platform=platform_name,
        action_space="pyautogui",
        observation_type="mixed",
    )

    agent.reset()

    print("start the agent")

    # Run the agent on your own device
    run_agent(agent, request.instruction)
    
    return {"status": "completed"}

@app.post("/execute")
async def execute_command_stream(cmd: CommandRequest):
    engine_params = {
        "engine_type": "openai",
        "model": "gpt-4o",
    }

    agent = GraphSearchAgent(
        engine_params,
        grounding_agent,
        platform=platform_name,
        action_space="pyautogui",
        observation_type="mixed",
    )

    obs = {
        "screenshot": cmd.obs.screenshot,
        "accessibility_tree": cmd.obs.accessibility_tree,
    }
    instruction = cmd.instruction
    info, code = agent.predict(instruction=instruction, observation=obs)

    return StreamingResponse(stream_code(code), media_type="text/plain")