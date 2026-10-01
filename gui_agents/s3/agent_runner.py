"""Execution-free prediction and approved dispatch, run only in a task worker."""

import datetime
import time
import uuid

from gui_agents.s3.screens import (
    capture_screenshot_png,
    selected_monitor,
    to_ui_jpeg,
    foreground_window,
    restore_foreground_window,
)
from gui_agents.s3.ui_config import AgentConfig, engine_params
from gui_agents.s3.utils.actions import execute_action


def build_agent(config, monitor, stop):
    from gui_agents.s3.agents.agent_s import AgentS3
    from gui_agents.s3.agents.grounding import OSWorldACI
    from gui_agents.s3.utils.local_env import LocalEnv

    local = LocalEnv() if config.enable_local_env else None
    if local:
        local.controller.stop_event = stop
        local.controller.script_timeout = config.action_timeout
    import platform

    grounding = OSWorldACI(
        local,
        platform.system().lower(),
        engine_params(config),
        engine_params(config, "grounding"),
        width=monitor["width"],
        height=monitor["height"],
    )
    grounding.deferred_execution = True
    grounding.require_grounded_typing = config.approval == "manual"
    agent = AgentS3(
        engine_params(config),
        grounding,
        platform=grounding.platform,
        max_trajectory_length=config.max_trajectory_length,
        enable_reflection=config.enable_reflection,
    )
    return agent, grounding


def run_agent_loop(instruction, config, emit, stop, commands):
    config = AgentConfig.model_validate(config)
    monitor = selected_monitor(config.monitor)
    agent, grounding = build_agent(config, monitor, stop)
    from gui_agents.s3.overlay import OverlayController

    overlay = OverlayController()
    result = {"done_reason": "max_steps_reached", "error": None}
    try:
        if config.overlay:
            status = overlay.show(
                instruction, f"{config.provider}/{config.model}", monitor
            )
            emit("overlay", status)
        for number in range(1, config.max_steps + 1):
            if stop.is_set():
                raise InterruptedError("Task stopped")
            # Keep the overlay hidden from capture until the action completes.
            if config.overlay:
                overlay.withdraw()
            emit("state", {"status": "planning"})
            if selected_monitor(config.monitor) != monitor:
                raise InterruptedError("Monitor layout changed; replan the task")
            focus = foreground_window()
            png = capture_screenshot_png(monitor)
            emit("screenshot", to_ui_jpeg(png))
            emit("log", f"Step {number}/{config.max_steps}: asking planner")
            info, _ = agent.predict(
                instruction=instruction, observation={"screenshot": png}
            )
            if stop.is_set():
                raise InterruptedError("Task stopped")
            action = grounding.prepared_action
            if action is None:
                raise ValueError("Planner did not prepare an action")
            record = {
                "n": number,
                "plan": str(info.get("plan", ""))[:8192],
                "plan_code": str(info.get("plan_code", ""))[:8192],
                "exec_code": action.preview,
                "reflection": str(info.get("reflection") or "")[:8192],
                "status": "planned",
                "error": "",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            if action.name in ("done", "fail"):
                record["status"] = "finished"
                emit("step", record)
                result["done_reason"] = "done" if action.name == "done" else "failed"
                break
            if config.approval == "dry_run":
                record["status"] = "dry_run"
                emit("step", record)
            else:
                if config.approval == "manual":
                    token = uuid.uuid4().hex
                    record.update(status="awaiting_approval", approval_token=token)
                    emit("step", record)
                    if config.overlay:
                        overlay.update(
                            f"Step {number}: awaiting approval", action.preview
                        )
                        overlay.restore()
                    while True:
                        if stop.is_set():
                            raise InterruptedError("Task stopped")
                        try:
                            command = commands.get(timeout=0.2)
                        except Exception as exc:
                            import queue

                            if isinstance(exc, queue.Empty):
                                continue
                            raise
                        if (
                            command.get("step") == number
                            and command.get("approval_token") == token
                        ):
                            if not command["approved"]:
                                record["status"] = "rejected"
                                emit("step", record)
                                raise InterruptedError("Step rejected")
                            record["status"] = "approved"
                            emit("step", record)
                            break
                    if config.overlay:
                        overlay.withdraw()
                if stop.is_set():
                    raise InterruptedError("Task stopped")
                if selected_monitor(config.monitor) != monitor:
                    raise InterruptedError("Monitor layout changed; replan the task")
                if config.approval == "manual" and action.name not in (
                    "open",
                    "switch_applications",
                    "wait",
                    "save_to_knowledge",
                    "call_code_agent",
                ):
                    restore_foreground_window(focus)
                if stop.is_set():
                    raise InterruptedError("Task stopped")
                emit("state", {"status": "executing"})
                try:
                    execute_action(grounding, action, stop, monitor)
                except Exception as exc:
                    record.update(status="error", error=str(exc)[:2000])
                    emit("step", record)
                    raise
                record["status"] = "executed"
                emit("step", record)
            if config.overlay:
                overlay.update(f"Step {number}: {record['status']}", action.preview)
                overlay.restore()
            if stop.wait(0.5):
                raise InterruptedError("Task stopped")
    except InterruptedError as exc:
        if "record" in locals() and record["status"] in (
            "planned",
            "awaiting_approval",
            "approved",
        ):
            record["status"] = "cancelled"
            emit("step", record)
        result.update(done_reason="stopped", error=None)
        emit("log", str(exc))
    except Exception as exc:
        import sys

        gui = sys.modules.get("pyautogui")

        result.update(
            done_reason=(
                "stopped"
                if isinstance(exc, getattr(gui, "FailSafeException", ()))
                else "error"
            ),
            error=str(exc)[:4000],
        )
    finally:
        overlay.close()
    return result
