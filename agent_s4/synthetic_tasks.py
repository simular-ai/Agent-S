"""Opt-in synthetic task overlay, separate from the pinned official checkout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT_ENV = "OSWORLD_SYNTHETIC_ROOT"
IDS = [f"s_{i:03d}" for i in range(1, 31)]


def validate_package(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads(
        (
            root / "evaluation_examples/synthetic_assets/PACKAGE_MANIFEST.json"
        ).read_text()
    )
    if (
        manifest.get("kind") != "cues-synthetic-json-v1"
        or manifest.get("task_count") != 30
    ):
        raise ValueError("Expected the 30-task synthetic package")
    if sorted(item["id"] for item in manifest["tasks"]) != IDS:
        raise ValueError("Synthetic manifest must contain exactly s_001 through s_030")
    for relative, expected in manifest["files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid package path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Synthetic package hash mismatch: {relative}")
    for item in manifest["tasks"]:
        task = json.loads((root / item["json_path"]).read_text())
        if (
            task.get("id") != item["id"]
            or task.get("verification") != "dummy_constant_zero"
        ):
            raise ValueError(
                f"Invalid synthetic identity/score semantics: {item['id']}"
            )
        evaluator = task["evaluator"]
        if (
            evaluator["func"] != "literal_match"
            or evaluator["result"]
            != {"type": "rule", "rules": "SYNTHETIC_TASK_NOT_GRADED"}
            or evaluator["expected"]
            != {"type": "rule", "rules": "DUMMY_PLACEHOLDER_ALWAYS_ZERO"}
        ):
            raise ValueError(f"Unexpected dummy evaluator: {item['id']}")
        resolve_assets(task["config"], root)
    return manifest


def resolve_assets(value, root: Path):
    """Resolve coordinator-side downloads/state files independent of eval cwd."""
    if isinstance(value, str) and value.startswith(
        "evaluation_examples/synthetic_assets/"
    ):
        path = (root / value).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid synthetic asset: {value}")
        return str(path)
    if isinstance(value, list):
        return [resolve_assets(v, root) for v in value]
    if isinstance(value, dict):
        return {k: resolve_assets(v, root) for k, v in value.items()}
    return value


def extend_task_index(index: dict[str, str]) -> None:
    configured = os.environ.get(ROOT_ENV)
    if not configured:
        return
    root = Path(configured).resolve()
    manifest = validate_package(root)
    if not manifest.get("website_configuration", {}).get("configured"):
        raise ValueError(
            "Run prepare_osworld_synthetic.py to configure website origins first"
        )
    for item in manifest["tasks"]:
        task_id = item["id"]
        if task_id in index:
            raise ValueError(
                f"Synthetic task ID collides with existing task: {task_id}"
            )
        index[task_id] = str(root / item["json_path"])


def prepare_loaded_task(task: dict, task_path: str) -> dict:
    if task.get("verification") != "dummy_constant_zero":
        return task
    root = Path(task_path).resolve().parents[3]
    task["config"] = resolve_assets(task["config"], root)
    return SyntheticTask(task)


def checked_guest_execute(controller, command: list[str]) -> None:
    """Honor the package's check=True instead of silently ignoring exit status."""
    import requests

    response = requests.post(
        controller.http_server + "/setup/execute",
        json={"command": command, "shell": False, "timeout": 120},
        timeout=150,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"Synthetic setup command failed (exit={result.get('returncode')}): "
            f"{result.get('error', '')[:1500]}"
        )


def set_guest_utc(controller) -> None:
    """Put the guest OS on UTC. Must run before Chrome starts.

    An already-running Chrome keeps the timezone it was launched with, so
    changing the OS underneath it leaves ``Date`` on the old offset.
    """
    checked_guest_execute(
        controller,
        [
            "python3",
            "-c",
            'import subprocess,sys; subprocess.run(["sudo","-S","timedatectl","set-timezone","Etc/UTC"], '
            'input=sys.argv[1]+"\\n", text=True, capture_output=True, check=True)',
            controller.client_password,
        ],
    )


def verify_browser_utc(controller) -> None:
    """Confirm Chrome itself is on UTC before any timed state is seeded."""
    import time
    from playwright.sync_api import sync_playwright

    # Verify a new blank page before any timed website state is seeded;
    # never reload an active checkout timer.
    with sync_playwright() as playwright:
        # socat accepts on 9222 as soon as it starts, so an early connect is
        # answered and then dropped ("socket hang up") until Chrome itself
        # binds 1337. Retry on the same cadence as _chrome_open_tabs_setup.
        endpoint = f"http://{controller.vm_ip}:{controller.chromium_port}"
        for attempt in range(15):
            try:
                browser = playwright.chromium.connect_over_cdp(endpoint)
                break
            except Exception:
                if attempt == 14:
                    raise
                time.sleep(5)
        page = browser.contexts[0].new_page()
        try:
            for _ in range(20):
                timezone = page.evaluate(
                    "Intl.DateTimeFormat().resolvedOptions().timeZone"
                )
                offset = page.evaluate(
                    'new Date("2026-10-21T12:00:00Z").getTimezoneOffset()'
                )
                if timezone in {"UTC", "Etc/UTC", "Etc/GMT", "GMT"} and offset == 0:
                    return
                time.sleep(0.5)
            raise RuntimeError(
                f"Synthetic task requires browser UTC; found {timezone}, offset={offset}"
            )
        finally:
            page.close()


class SyntheticTask(dict):
    """Use V2's custom setup hook while retaining the supplied dummy evaluator."""

    def setup(self, controller, use_proxy=False):
        # Order is load-bearing: the guest clock must be UTC before Chrome
        # launches (a running Chrome keeps its original timezone), and the
        # browser must be confirmed on UTC before any timed state is seeded.
        needs_utc = self["id"] in {"s_020", "s_028"}
        if needs_utc:
            set_guest_utc(controller)
        for action in self["config"]:
            if needs_utc and action["type"] == WEB_SETUP_ACTION:
                verify_browser_utc(controller)
                needs_utc = False
            params = action["parameters"]
            if action["type"] == "execute" and params.get("check") is True:
                checked_guest_execute(controller, params["command"])
            else:
                if not controller.setup([action], use_proxy):
                    raise RuntimeError(f"Synthetic setup failed: {action['type']}")
        if needs_utc:
            raise RuntimeError(
                f"{self['id']} requires browser UTC but has no {WEB_SETUP_ACTION} step"
            )


# Chrome is not running in the guest image; the official V2 chrome tasks start
# it themselves before touching CDP. The imported synthetic package omitted
# these two steps, so every web task failed setup with ECONNREFUSED on 9222.
# Ports mirror the official tasks: Chrome listens on 1337, socat republishes it
# on the 9222 the security group and setup controller expect.
CHROME_LAUNCH_STEPS = [
    {
        "type": "launch",
        "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]},
    },
    {
        "type": "launch",
        "parameters": {
            "command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]
        },
    },
]
WEB_SETUP_ACTION = "chrome_open_tabs_with_state"


def _launch_commands(steps) -> list[list[str]]:
    return [
        step.get("parameters", {}).get("command")
        for step in steps
        if step.get("type") == "launch"
    ]


def ensure_chrome_launch(root: Path, apply: bool = True) -> list[str]:
    """Guarantee each web task starts Chrome before connecting over CDP.

    Returns the task IDs that were missing the launch steps. Idempotent: a
    package that already carries them is left byte-identical.
    """
    root = Path(root).resolve()
    manifest_path = root / "evaluation_examples/synthetic_assets/PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    required = [step["parameters"]["command"] for step in CHROME_LAUNCH_STEPS]
    repaired: list[str] = []
    dirty = False
    for item in manifest["tasks"]:
        if not item["web"]:
            continue
        path = root / item["json_path"]
        task = json.loads(path.read_text())
        config = task["config"]
        indices = [
            i for i, step in enumerate(config) if step.get("type") == WEB_SETUP_ACTION
        ]
        if len(indices) != 1:
            raise ValueError(
                f"{item['id']}: expected exactly one {WEB_SETUP_ACTION} step, "
                f"found {len(indices)}"
            )
        index = indices[0]
        present = _launch_commands(config[:index])
        if all(command in present for command in required):
            if item["web_config_index"] != index:
                item["web_config_index"] = index
                dirty = True
            continue
        repaired.append(item["id"])
        if not apply:
            continue
        config[index:index] = json.loads(json.dumps(CHROME_LAUNCH_STEPS))
        item["web_config_index"] = index + len(CHROME_LAUNCH_STEPS)
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
        manifest["files_sha256"][str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        dirty = True
    if apply and dirty:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(manifest_path)
    return repaired
