import asyncio
import os
import platform
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gui_agents.s1.core.AgentS import GraphSearchAgent
import io
import pyautogui
import time
from threading import Event, Lock

# Determine the operating system and select appropriate ACI
current_platform = platform.system().lower()
if current_platform == "linux":
    from gui_agents.s1.aci.LinuxOSACI import LinuxACI, UIElement

    grounding_agent = LinuxACI()
elif current_platform == "darwin":
    from gui_agents.s1.aci.MacOSACI import MacOSACI, UIElement

    grounding_agent = MacOSACI()
elif current_platform == "windows":
    from gui_agents.s1.aci.WindowsOSACI import WindowsACI, UIElement

    grounding_agent = WindowsACI()
else:
    raise ValueError(f"Unsupported operating system: {current_platform}")

app = FastAPI()

# Add global lock and status tracking
agent_lock = Lock()
agent_status = {"is_running": False, "current_instruction": None, "start_time": None}

# Add a stop event
stop_event = Event()


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
    global stop_event
    stop_event.clear()  # Reset the stop event
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for _ in range(15):
        # Check if stop was requested
        if stop_event.is_set():
            print("Agent execution stopped by user")
            return

        print("iteration", _)

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
    global agent_status

    # Check if agent is already running
    if not agent_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="An agent is already running. Use /status to check current run or /stop to stop it.",
        )

    try:
        agent_status = {
            "is_running": True,
            "current_instruction": request.instruction,
            "start_time": time.time(),
            "model": request.model,
        }

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
            platform=current_platform,
            action_space="pyautogui",
            observation_type="mixed",
        )

        agent.reset()
        print("start the agent")
        run_agent(agent, request.instruction)

        return {"status": "completed"}

    finally:
        agent_status = {
            "is_running": False,
            "current_instruction": None,
            "start_time": None,
        }
        agent_lock.release()


@app.get("/status")
async def get_status():
    if agent_status["is_running"]:
        duration = time.time() - agent_status["start_time"]
        return {
            "status": "running",
            "instruction": agent_status["current_instruction"],
            "model": agent_status["model"],
            "running_for_seconds": round(duration, 2),
        }
    return {"status": "idle"}


@app.post("/execute")
async def execute_command_stream(cmd: CommandRequest):
    engine_params = {
        "engine_type": "openai",
        "model": "gpt-4o",
    }

    agent = GraphSearchAgent(
        engine_params,
        grounding_agent,
        platform=current_platform,
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


@app.post("/stop")
async def stop_agent():
    if not agent_status["is_running"]:
        raise HTTPException(status_code=404, detail="No agent is currently running")

    global stop_event
    stop_event.set()
    return {"status": "stop signal sent"}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",  # Allows external access
        port=8000,  # Default port for FastAPI
        reload=True,  # Auto-reload on code changes
    )
