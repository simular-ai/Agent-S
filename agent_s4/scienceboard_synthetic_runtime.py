"""Vendored verbatim from data/scienceboard_synthetic_50/runtime.py (the synthetic bank's own adapter).

Setup-only bridge for an already provisioned ScienceBoard VM manager.

No model, VM provider, evaluator or GUI dependency is imported by this module.
The caller owns manager lifetime, task execution, and collection of raw artifacts.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path, PurePosixPath
import re
import time

APPLICATIONS = {"Celestia", "GrassGIS", "KAlgebra", "ChimeraX", "Lean", "TeXstudio"}
COMMON = {"execute", "launch", "touch", "opt"}
APP_OPS = {
    "Celestia": set(),
    "Lean": set(),
    "TeXstudio": set(),
    "KAlgebra": {"tab", "func_2d", "func_3d"},
    "GrassGIS": {"map", "layer", "scale"},
    "ChimeraX": {"open", "turn", "color", "clear_log"},
}
REQUIRED_ARGS = {
    "execute": {"command"},
    "launch": {"command"},
    "touch": {"path", "text"},
    "opt": {"depth"},
    "tab": {"index"},
    "func_2d": {"expr"},
    "func_3d": {"expr"},
    "map": {"grassdb", "location", "mapset"},
    "layer": {"query"},
    "scale": {"scale"},
    "open": {"name"},
    "turn": {"axis", "angle"},
    "color": {"style"},
    "clear_log": set(),
}


class TrainingEvaluationUnavailable(RuntimeError):
    """This task supplies experience, not a ground-truth outcome."""


def validate_spec(spec: dict, directory: Path) -> list[str]:
    errors = []
    fields = {
        "id",
        "title",
        "primary_application",
        "applications",
        "instruction",
        "steps",
        "verification",
        "category",
        "difficulty_rationale",
        "initial_state",
        "initialize",
        "assumptions",
        "materials",
    }
    errors.extend(f"Missing field: {key}" for key in sorted(fields - spec.keys()))
    if errors:
        return errors
    if not re.fullmatch(r"sb_[a-z]+_\d{3}", spec["id"]):
        errors.append("ID must match sb_<application>_<NNN>")
    app = spec["primary_application"]
    if app not in APPLICATIONS:
        errors.append("Unsupported primary application")
    if not isinstance(spec["applications"], list) or app not in spec["applications"]:
        errors.append("Primary application must be in applications")
    elif set(spec["applications"]) - APPLICATIONS:
        errors.append("Applications must use canonical ScienceBoard names")
    if spec["verification"] != "none":
        errors.append("Training verification must be none")
    if {
        "evaluate",
        "evaluator",
        "ans",
        "answer",
        "expected",
        "reward",
        "success",
    } & spec.keys():
        errors.append(
            "Training specification must not include evaluation or answer fields"
        )
    if type(spec["steps"]) is not int or spec["steps"] not in {5, 10, 15, 20}:
        errors.append("Use a benchmark-style 5/10/15/20 action budget")
    for key in (
        "title",
        "instruction",
        "category",
        "difficulty_rationale",
        "initial_state",
    ):
        if not isinstance(spec[key], str) or not spec[key].strip():
            errors.append(f"{key} must be nonempty text")
    if not isinstance(spec["assumptions"], list):
        errors.append("assumptions must be a list")
    declared = spec["materials"]
    if not isinstance(declared, list) or not all(isinstance(p, str) for p in declared):
        return errors + ["materials must be a list of relative paths"]
    if len(declared) != len(set(declared)):
        errors.append("Repeated material path")
    for name in declared:
        p = PurePosixPath(name)
        source = directory / name
        if (
            p.is_absolute()
            or ".." in p.parts
            or not p.parts
            or p.parts[0] != "materials"
        ):
            errors.append(f"Invalid material path: {name}")
        elif (
            not source.is_file()
            or source.is_symlink()
            or not source.resolve().is_relative_to(directory.resolve())
        ):
            errors.append(f"Missing or unsafe material: {name}")
    actual = {
        str(p.relative_to(directory))
        for p in (directory / "materials").rglob("*")
        if p.is_file()
    }
    if actual != set(declared):
        errors.append(f"Material inventory mismatch: {sorted(actual ^ set(declared))}")
    if not isinstance(spec["initialize"], list) or not spec["initialize"]:
        return errors + ["initialize must contain setup operations"]
    for i, op in enumerate(spec["initialize"]):
        if not isinstance(op, dict):
            errors.append(f"Setup {i} must be an object")
            continue
        func = op.get("func")
        if func not in COMMON | APP_OPS.get(app, set()):
            errors.append(f"Setup {i}: unsupported {app} operation {func}")
            continue
        required = REQUIRED_ARGS[func]
        extra = {"shell"} if func in {"execute", "launch"} else set()
        if required - op.keys() or op.keys() - required - extra - {"func", "wait"}:
            errors.append(f"Setup {i}: invalid arguments for {func}")
        wait = op.get("wait", 0)
        if not isinstance(wait, (int, float)) or wait < 0 or wait > 120:
            errors.append(f"Setup {i}: invalid wait")
        if func in {"execute", "launch"}:
            command = op.get("command")
            if not (
                isinstance(command, str)
                and command.strip()
                or isinstance(command, list)
                and command
                and all(isinstance(s, str) for s in command)
            ):
                errors.append(f"Setup {i}: command must be text or argv")
            if type(op.get("shell", False)) is not bool:
                errors.append(f"Setup {i}: shell must be boolean")
        if func == "touch":
            p = PurePosixPath(op.get("path", ""))
            root = PurePosixPath(f"/home/user/ScienceBoardSynthetic/{spec['id']}")
            lean_file = PurePosixPath(f"/home/user/sci/Sci/{spec['id']}.lean")
            if ".." in p.parts or not (p.is_relative_to(root) or p == lean_file):
                errors.append(f"Setup {i}: touch path outside task workspace")
    return errors


class TrainingTask:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.spec = json.loads((self.directory / "task.json").read_text())
        errors = validate_spec(self.spec, self.directory)
        if errors:
            raise ValueError("; ".join(errors))
        self.id = self.spec["id"]
        self.instruction = self.spec["instruction"]
        self.steps = self.spec["steps"]
        self.guest_root = f"/home/user/ScienceBoardSynthetic/{self.id}"

    def evaluate(self, *args, **kwargs):
        raise TrainingEvaluationUnavailable(
            "Synthetic training tasks are ungraded; retain raw artifacts and nullable outcomes."
        )

    eval = evaluate

    def prepare(self, manager, *, reset=True, sleep=time.sleep):
        """Reset the guest and install/initialize inputs; never execute a task agent.

        `manager` is the primary application's entered ScienceBoard VMManager.
        `reset=False` requires the caller to have reset and prepared the guest already.
        Only call this on the explicitly provisioned disposable task VM.
        """

        def request(route, payload):
            response = manager._request(route, {"json": payload})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Setup request failed: {route} ({response.status_code})"
                )
            if route == "POST/setup/execute":
                body = response.json()
                if body.get("status") != "success" or body.get("returncode") != 0:
                    raise RuntimeError(f"Guest setup command failed: {body}")
            return response

        def execute(command):
            request("POST/setup/execute", {"command": command, "shell": False})

        if reset:
            if not manager.revert("sci_bench"):
                raise RuntimeError("Guest snapshot reset failed")
            # Match sci/vm/vtask.py startup before installing task-local inputs.
            sleep(1)
            execute(["/bin/bash", "/home/user/server/reset.sh"])
        execute(["mkdir", "-p", self.guest_root])
        install = (
            "import base64,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_bytes(base64.b64decode(sys.argv[2]))"
        )
        for name in self.spec["materials"]:
            relative = PurePosixPath(name).relative_to("materials")
            data = base64.b64encode((self.directory / name).read_bytes()).decode(
                "ascii"
            )
            execute(["python3", "-c", install, f"{self.guest_root}/{relative}", data])
        port = str(getattr(manager, "port", 8000))
        for original in self.spec["initialize"]:
            op = copy.deepcopy(original)
            func, wait = op.pop("func"), op.pop("wait", 0)
            if func in {"execute", "launch"}:
                c = op["command"]
                op["command"] = (
                    [s.replace("«PORTLIKE»", port) for s in c]
                    if isinstance(c, list)
                    else c.replace("«PORTLIKE»", port)
                )
                request(f"POST/setup/{func}", op)
            elif func == "touch":
                execute(["mkdir", "-p", str(PurePosixPath(op["path"]).parent)])
                if not manager.write_file(op["path"], op["text"]):
                    raise RuntimeError("Guest input write failed")
            elif func == "opt":
                request("POST/opt", op)
            elif func in {"tab", "func_2d", "func_3d", "map", "layer", "scale"}:
                method = {
                    "tab": "operate_tab",
                    "func_2d": "operate_func2d",
                    "func_3d": "operate_func3d",
                    "map": "operate_map",
                    "layer": "operate_layer",
                    "scale": "operate_scale",
                }[func]
                if not getattr(manager, method)(**op):
                    raise RuntimeError(f"Application setup failed: {func}")
            else:
                command = {
                    "open": lambda: f"open {op['name']}",
                    "turn": lambda: f"turn {op['axis']} {op['angle']}",
                    "color": lambda: (
                        "rainbow"
                        if op["style"] == "rainbow"
                        else f"color {op['style']}"
                    ),
                    "clear_log": lambda: "log clear",
                }[func]()
                _, success = manager._call(command)
                if not success:
                    raise RuntimeError(f"ChimeraX setup failed: {func}")
            sleep(wait)
        return {
            "task_id": self.id,
            "status": "initialized",
            "reward": None,
            "success": None,
        }

    def unscored_record(self, *, stop_reason, artifact_paths=()):
        return {
            "task_id": self.id,
            "status": "unscored",
            "reward": None,
            "success": None,
            "stop_reason": stop_reason,
            "artifacts": list(artifact_paths),
        }
