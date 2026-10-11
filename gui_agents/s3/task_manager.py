"""One supervised worker for all desktop entrypoints, with bounded history."""

import copy
import datetime
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager

from gui_agents.s3.ui_config import AgentConfig, engine_params

ACTIVE_STATES = {
    "queued",
    "planning",
    "awaiting_approval",
    "executing",
    "stopping",
    "finishing",
}


class TaskConflict(Exception):
    pass


@contextmanager
def desktop_lease():
    """Prevent separate UI instances/workers from driving the same desktop."""
    if os.name == "nt":
        import win32event

        handle = win32event.CreateMutex(None, False, "Local\\AgentS3Desktop")
        acquired = win32event.WaitForSingleObject(handle, 0) in (
            win32event.WAIT_OBJECT_0,
            win32event.WAIT_ABANDONED,
        )
        try:
            if not acquired:
                raise TaskConflict("Another Agent S UI worker owns this desktop")
            yield
        finally:
            if acquired:
                win32event.ReleaseMutex(handle)
            handle.Close()
    else:
        import fcntl
        from gui_agents.s3.ui_config import state_directory

        path = state_directory() / "desktop.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TaskConflict(
                    "Another Agent S UI worker owns this desktop"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


def _worker(instruction, config, events, stop, commands, start_gate):
    if not start_gate.wait(timeout=10):
        return
    if os.name != "nt":
        os.setsid()
    from gui_agents.s3.screens import enable_physical_coordinates
    from gui_agents.s3.agent_runner import run_agent_loop

    enable_physical_coordinates()

    def emit(kind, data):
        events.send((kind, data))

    try:
        with desktop_lease():
            result = run_agent_loop(instruction, config, emit, stop, commands)
    except Exception as exc:
        result = {"done_reason": "error", "error": str(exc)[:4000]}
    emit("result", result)


class TaskManager:
    def __init__(self, max_tasks=25, retention_seconds=3600, context=None):
        self.context = context or multiprocessing.get_context("spawn")
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.tasks = {}
        self.max_tasks = max_tasks
        self.retention_seconds = retention_seconds
        self.closed = False

    def _prune(self, reserve=0):
        now = time.monotonic()
        finished = sorted(
            (t for t in self.tasks.values() if t["status"] not in ACTIVE_STATES),
            key=lambda t: t["created_monotonic"],
        )
        for task in finished:
            if (
                now - task.get("finished_monotonic", task["created_monotonic"])
                > self.retention_seconds
                or len(self.tasks) + reserve > self.max_tasks
            ):
                del self.tasks[task["id"]]

    def launch(self, instruction, config):
        config = AgentConfig.model_validate(config)
        if not config.model or (not config.ground_model and config.provider != "codex"):
            if not config.ground_model:
                raise ValueError("Configure both planner and grounding model IDs")
        if config.provider == "codex" and not config.ground_model:
            raise ValueError("Codex supplies the planner; still configure a grounding model ID")
        engine_params(config)
        engine_params(config, "grounding")
        with self.lock:
            if self.closed:
                raise TaskConflict("Task manager is shutting down")
            if any(t["status"] in ACTIVE_STATES for t in self.tasks.values()):
                raise TaskConflict("Another desktop task is active")
            self._prune(reserve=1)
            task_id = uuid.uuid4().hex
            event_reader, event_writer = self.context.Pipe(duplex=False)
            events = queue.Queue(maxsize=128)
            commands = self.context.Queue(maxsize=8)
            stop = self.context.Event()
            start_gate = self.context.Event()
            now = time.monotonic()
            task = {
                "id": task_id,
                "instruction": instruction,
                "status": "queued",
                "config": config,
                "steps": [],
                "logs": [],
                "current_step": 0,
                "latest_screenshot": None,
                "done_reason": None,
                "error": None,
                "overlay": "off",
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "created_monotonic": now,
                "phase_started": now,
                "stop_requested": None,
                "pending": None,
                "stop": stop,
                "commands": commands,
                "events": events,
                "event_reader": event_reader,
                "reader_done": threading.Event(),
                "version": 0,
            }
            process = self.context.Process(
                target=_worker,
                args=(
                    instruction,
                    config.model_dump(),
                    event_writer,
                    stop,
                    commands,
                    start_gate,
                ),
            )
            task["process"] = process
            self.tasks[task_id] = task
            try:
                process.start()
                event_writer.close()
                task["job"] = self._attach_job(process)
                start_gate.set()
            except Exception:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
                del self.tasks[task_id]
                event_reader.close()
                event_writer.close()
                commands.close()
                raise
            reader = threading.Thread(target=self._receive, args=(task,), daemon=True)
            task["reader_thread"] = reader
            reader.start()
            thread = threading.Thread(target=self._watch, args=(task,), daemon=True)
            task["watcher"] = thread
            thread.start()
            return self.public(task_id)

    def _receive(self, task):
        # A partial IPC message must never block deadline/cancellation supervision.
        # Only this receiver blocks on the pipe; EOF is guaranteed when the worker
        # dies because the parent closed its copy of the write end after spawn.
        try:
            while True:
                task["events"].put(task["event_reader"].recv())
        except (EOFError, OSError):
            pass
        finally:
            task["reader_done"].set()

    def _event(self, task, kind, data):
        with self.changed:
            if kind == "step":
                record = copy.deepcopy(data)
                for index, existing in enumerate(task["steps"]):
                    if existing["n"] == record["n"]:
                        task["steps"][index] = record
                        break
                else:
                    task["steps"].append(record)
                task["current_step"] = record["n"]
                if (
                    record["status"] == "awaiting_approval"
                    and task["stop_requested"] is None
                ):
                    task["pending"] = {
                        "step": record["n"],
                        "approval_token": record["approval_token"],
                    }
                    task["status"] = "awaiting_approval"
                    task["phase_started"] = time.monotonic()
            elif kind == "state" and task["stop_requested"] is None:
                if data["status"] != "awaiting_approval":
                    task["status"] = data["status"]
                    task["phase_started"] = time.monotonic()
            elif kind == "log":
                task["logs"].append(str(data)[:2000])
                task["logs"] = task["logs"][-200:]
            elif kind == "screenshot":
                task["latest_screenshot"] = data
            elif kind == "overlay":
                task["overlay"] = data
            task["version"] += 1
            self.changed.notify_all()

    def _kill(self, process):
        if not process.is_alive():
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            import signal

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.is_alive():
            process.kill()

    def _attach_job(self, process):
        if os.name != "nt":
            return None
        import win32api
        import win32con
        import win32job

        job = win32job.CreateJobObject(None, "AgentS-" + uuid.uuid4().hex)
        try:
            info = win32job.QueryInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"][
                "LimitFlags"
            ] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation, info
            )
            handle = win32api.OpenProcess(
                win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                False,
                process.pid,
            )
            try:
                win32job.AssignProcessToJobObject(job, handle)
            finally:
                handle.Close()
            return job
        except Exception:
            job.Close()
            raise

    def _watch(self, task):
        process, result = task["process"], None
        try:
            while True:
                try:
                    kind, data = task["events"].get(timeout=0.1)
                    if kind == "result":
                        result = data
                        self._event(task, "state", {"status": "finishing"})
                    else:
                        self._event(task, kind, data)
                except queue.Empty:
                    pass
                with self.lock:
                    now = time.monotonic()
                    elapsed = now - task["created_monotonic"]
                    phase = now - task["phase_started"]
                    config = task["config"]
                    exceeded = (
                        elapsed > config.task_timeout
                        or (
                            task["status"] == "planning"
                            and phase > config.inference_timeout
                        )
                        or (
                            task["status"] == "executing"
                            and phase > config.action_timeout
                        )
                    )
                    if exceeded and task["stop_requested"] is None:
                        task["error"] = "Task or phase deadline exceeded"
                        task["stop_requested"] = now
                        task["status"] = "stopping"
                        task["stop"].set()
                    kill = (
                        task["stop_requested"] is not None
                        and now - task["stop_requested"] > 2
                    )
                if kill:
                    self._kill(process)
                if not process.is_alive():
                    if task.get("job") is not None:
                        task["job"].Close()
                        task["job"] = None
                    task["reader_done"].wait(timeout=1)
                    while True:
                        try:
                            kind, data = task["events"].get_nowait()
                            if kind == "result":
                                result = data
                            else:
                                self._event(task, kind, data)
                        except queue.Empty:
                            break
                    break
            process.join(timeout=3)
        except Exception as exc:
            result = {
                "done_reason": "error",
                "error": f"Worker supervision failed: {exc}",
            }
            self._kill(process)
            process.join(timeout=3)
        finally:
            # Never release the single-desktop gate while a worker is alive.
            while process.is_alive():
                self._kill(process)
                process.join(timeout=1)
            if task.get("job") is not None:
                task["job"].Close()
                task["job"] = None
            with self.changed:
                if task["error"]:
                    reason = "error"
                elif task["stop_requested"] is not None:
                    reason = "stopped"
                else:
                    reason = (result or {}).get("done_reason", "error")
                    task["error"] = (result or {}).get("error") or (
                        f"Worker exited without result ({process.exitcode})"
                        if result is None
                        else None
                    )
                task.update(
                    status="incomplete" if reason == "max_steps_reached" else reason,
                    done_reason=reason,
                    pending=None,
                    finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
                    finished_monotonic=time.monotonic(),
                    overlay="off",
                )
                task["version"] += 1
                self.changed.notify_all()
            task["event_reader"].close()
            task["reader_thread"].join(timeout=1)
            task["commands"].cancel_join_thread()
            task["commands"].close()

    def public(self, task_id, detail="full"):
        with self.lock:
            task = self.tasks[task_id]
            fields = (
                "id",
                "instruction",
                "status",
                "current_step",
                "created_at",
                "finished_at",
                "done_reason",
                "error",
                "overlay",
                "version",
            )
            result = {key: task[key] for key in fields}
            result.update(
                max_steps=task["config"].max_steps,
                approval=task["config"].approval,
                profile_id=task["config"].profile_id,
                profile_name=task["config"].profile_name,
                provider=task["config"].provider,
                model=task["config"].model,
                steps=copy.deepcopy(
                    task["steps"]
                    if detail == "full"
                    else task["steps"][-1:] if detail == "summary" else []
                ),
                logs=list(
                    task["logs"]
                    if detail == "full"
                    else task["logs"][-10:] if detail == "summary" else []
                ),
                has_screenshot=task["latest_screenshot"] is not None,
                awaiting_approval=task["pending"] is not None,
                pending_approval=copy.deepcopy(task["pending"]),
            )
            return result

    def list(self):
        with self.lock:
            self._prune()
            return [self.public(task_id, detail="list") for task_id in self.tasks]

    def active(self):
        with self.lock:
            return sum(t["status"] in ACTIVE_STATES for t in self.tasks.values())

    def screenshot(self, task_id):
        with self.lock:
            return self.tasks[task_id]["latest_screenshot"]

    def approve(self, task_id, step, token, approved):
        with self.changed:
            task = self.tasks[task_id]
            expected = {"step": step, "approval_token": token}
            if task["status"] != "awaiting_approval" or task["pending"] != expected:
                raise TaskConflict(
                    "Approval is stale or the task is not awaiting this step"
                )
            task["commands"].put_nowait({**expected, "approved": approved})
            task["pending"] = None
            task["status"] = "planning"
            task["phase_started"] = time.monotonic()
            task["version"] += 1
            self.changed.notify_all()
            return self.public(task_id)

    def stop(self, task_id):
        with self.changed:
            task = self.tasks[task_id]
            if task["status"] in ACTIVE_STATES and task["stop_requested"] is None:
                task["stop_requested"] = time.monotonic()
                task["status"] = "stopping"
                task["pending"] = None
                task["stop"].set()
                task["version"] += 1
                self.changed.notify_all()
            return self.public(task_id)

    def delete(self, task_id):
        with self.lock:
            if self.tasks[task_id]["status"] in ACTIVE_STATES:
                raise TaskConflict(
                    "Stop the task and wait for cleanup before deleting it"
                )
            del self.tasks[task_id]

    def shutdown(self):
        with self.lock:
            self.closed = True
            active = [t for t in self.tasks.values() if t["status"] in ACTIVE_STATES]
        for task in active:
            self.stop(task["id"])
        for task in active:
            task["watcher"].join(timeout=10)
