"""No desktop input: synthetic planners, images, transport and worker processes."""

import asyncio
import copy
import io
import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from gui_agents.s3 import agent_runner
from gui_agents.s3.core.engine import LMMEngineOpenAI
from gui_agents.s3.core.openai_compatible import (
    is_local_url,
    normalize_base_url,
    resolve_lmstudio_params,
)
from gui_agents.s3.task_manager import (
    ACTIVE_STATES,
    TaskConflict,
    TaskManager,
    desktop_lease,
)
from gui_agents.s3.ui_config import (
    AgentConfig,
    ConfigStore,
    backend,
    protect,
    request_target,
    unprotect,
)
from gui_agents.s3.core import codex as codex_module
from gui_agents.s3.ui_server import create_app
from gui_agents.s3.utils.actions import PreparedAction, parse_action
from gui_agents.s3.utils.common_utils import create_pyautogui_code
from gui_agents.s3.utils.formatters import CODE_VALID_FORMATTER


def config(**changes):
    return AgentConfig(model="planner-stub", ground_model="ground-stub", **changes)


class ActionTests(unittest.TestCase):
    def setUp(self):
        from gui_agents.s3.agents.grounding import OSWorldACI

        self.agent = object.__new__(OSWorldACI)
        self.agent.env = None
        self.agent.deferred_execution = True
        self.agent.width, self.agent.height = 1920, 1080

    def test_expression_cannot_execute(self):
        for code in [
            "agent.wait(__import__('os').system('bad'))",
            "agent.wait(1); print('bad')",
            "agent.__class__()",
            "agent.wait(*[1])",
            "agent.hotkey([\"ctrl'); print('bad') #\"])",
        ]:
            with self.subTest(code=code), self.assertRaises((ValueError, SyntaxError)):
                parse_action(self.agent, code)

    def test_validation_never_grounds(self):
        with patch.object(self.agent, "generate_coords") as ground:
            valid, _ = CODE_VALID_FORMATTER(
                self.agent, {}, "```python\nagent.click('button')\n```"
            )
            self.assertTrue(valid)
            ground.assert_not_called()

    def test_empty_launcher_targets_are_rejected_before_input(self):
        for code in (
            "agent.open('')",
            "agent.open(None)",
            "agent.switch_applications('  ')",
            "agent.click(None)",
        ):
            with self.assertRaises(ValueError):
                parse_action(self.agent, code)

    def test_manual_typing_requires_a_grounded_target(self):
        self.agent.require_grounded_typing = True
        with self.assertRaises(ValueError):
            parse_action(self.agent, "agent.type(text='synthetic')")
        self.assertEqual(
            parse_action(self.agent, "agent.type('the editor', text='synthetic')").name,
            "type",
        )

    def test_preparation_grounds_exactly_once(self):
        self.agent.engine_params_for_grounding = {
            "grounding_width": 1920,
            "grounding_height": 1080,
        }
        with patch.object(
            self.agent, "generate_coords", return_value=[100, 200]
        ) as ground:
            create_pyautogui_code(
                self.agent, "agent.click('button')", {"screenshot": b"synthetic"}
            )
            self.assertEqual(ground.call_count, 1)
            self.assertEqual(self.agent.prepared_action.arguments["point"], [100, 200])

    def test_coding_agent_is_deferred(self):
        self.agent.env = SimpleNamespace(controller=Mock())
        with patch.object(
            self.agent, "call_code_agent", wraps=self.agent.call_code_agent
        ) as method:
            method.is_agent_action = True
            # Mock replaces its signature; validate against the real bound API.
            from gui_agents.s3.utils.actions import prepare_action, ActionSpec

            prepared = prepare_action(
                self.agent, ActionSpec("call_code_agent", {"task": "synthetic"}), {}
            )
            self.assertEqual(prepared.name, "call_code_agent")
            method.assert_not_called()

    def test_dry_run_never_dispatches_and_stop_after_prediction_prevents_action(self):
        image = io.BytesIO()
        Image.new("RGB", (32, 18)).save(image, "PNG")
        stop = threading.Event()
        ground = SimpleNamespace(
            prepared_action=PreparedAction("type", {}, "type done, fail, next, wait")
        )
        agent = SimpleNamespace(predict=Mock(return_value=({"plan": "synthetic"}, [])))
        events = []
        with patch.object(
            agent_runner, "selected_monitor", return_value={"width": 32, "height": 18}
        ), patch.object(
            agent_runner, "build_agent", return_value=(agent, ground)
        ), patch.object(
            agent_runner, "capture_screenshot_png", return_value=image.getvalue()
        ), patch.object(
            agent_runner, "execute_action"
        ) as execute:
            result = agent_runner.run_agent_loop(
                "test",
                config(max_steps=1).model_dump(),
                lambda kind, data: events.append((kind, data)),
                stop,
                queue.Queue(),
            )
            execute.assert_not_called()
            self.assertEqual(result["done_reason"], "max_steps_reached")
            self.assertEqual(
                next(data for kind, data in events if kind == "step")["status"],
                "dry_run",
            )
            agent.predict.side_effect = lambda **kw: (stop.set() or ({}, []))
            result = agent_runner.run_agent_loop(
                "test",
                config(max_steps=1, approval="auto").model_dump(),
                lambda *args: None,
                stop,
                queue.Queue(),
            )
            execute.assert_not_called()
            self.assertEqual(result["done_reason"], "stopped")

    def test_real_worker_dry_run_cannot_invoke_coding_agent(self):
        from gui_agents.s3.core.mllm import LMMAgent
        from gui_agents.s3.agents.code_agent import CodeAgent

        output = io.BytesIO()
        Image.new("RGB", (32, 18)).save(output, "PNG")
        with patch.object(
            agent_runner, "selected_monitor", return_value={"width": 32, "height": 18}
        ), patch.object(
            agent_runner, "capture_screenshot_png", return_value=output.getvalue()
        ), patch.object(
            LMMAgent,
            "get_response",
            return_value="```python\nagent.call_code_agent()\n```",
        ), patch.object(
            CodeAgent, "execute"
        ) as execute:
            result = agent_runner.run_agent_loop(
                "test",
                config(max_steps=1, enable_local_env=True).model_dump(),
                lambda *args: None,
                threading.Event(),
                queue.Queue(),
            )
        self.assertEqual(result["done_reason"], "max_steps_reached")
        execute.assert_not_called()

    def test_monitor_offset_and_modifier_release_on_abort(self):
        from gui_agents.s3.utils.actions import execute_action

        class SafetyAbort(Exception):
            pass

        gui = SimpleNamespace(
            FAILSAFE=True,
            keyDown=Mock(),
            keyUp=Mock(),
            click=Mock(side_effect=SafetyAbort()),
        )
        action = PreparedAction(
            "click",
            {
                "point": [100, 200],
                "num_clicks": 1,
                "button_type": "left",
                "hold_keys": ["ctrl"],
            },
            "synthetic",
        )
        with patch.dict("sys.modules", {"pyautogui": gui}), self.assertRaises(
            SafetyAbort
        ):
            execute_action(
                self.agent, action, threading.Event(), {"left": -1920, "top": 0}
            )
        gui.click.assert_called_once_with(-1820, 200, clicks=1, button="left")
        gui.keyUp.assert_called_once_with("ctrl")
        self.assertTrue(gui.FAILSAFE)

    def test_execution_error_ends_the_task(self):
        output = io.BytesIO()
        Image.new("RGB", (32, 18)).save(output, "PNG")
        ground = SimpleNamespace(
            prepared_action=PreparedAction("click", {}, "synthetic")
        )
        agent = SimpleNamespace(predict=Mock(return_value=({}, [])))
        with patch.object(
            agent_runner, "selected_monitor", return_value={"width": 32, "height": 18}
        ), patch.object(
            agent_runner, "build_agent", return_value=(agent, ground)
        ), patch.object(
            agent_runner, "capture_screenshot_png", return_value=output.getvalue()
        ), patch.object(
            agent_runner, "execute_action", side_effect=RuntimeError("synthetic abort")
        ):
            result = agent_runner.run_agent_loop(
                "test",
                config(max_steps=3, approval="auto").model_dump(),
                lambda *args: None,
                threading.Event(),
                queue.Queue(),
            )
        self.assertEqual(agent.predict.call_count, 1)
        self.assertEqual(result["done_reason"], "error")
        self.assertIn("synthetic abort", result["error"])


class ConfigTests(unittest.TestCase):
    def test_url_parsing_and_legacy_engine_path(self):
        for value in [
            "https://localhost.evil.example",
            "https://127.0.0.1@evil.example",
            "https://evil.example/127.0.0.1",
            "http://10.evil.example",
            "http://192.168.1.3",
        ]:
            self.assertFalse(is_local_url(value))
        self.assertTrue(is_local_url("http://[::1]:1234"))
        self.assertEqual(
            normalize_base_url("http://192.168.1.3:11434"),
            "http://192.168.1.3:11434/v1",
        )
        self.assertEqual(
            normalize_base_url("https://host/v1beta/openai/"),
            "https://host/v1beta/openai",
        )
        self.assertEqual(
            LMMEngineOpenAI(
                model="stub", base_url="http://localhost:8000/custom"
            ).base_url,
            "http://localhost:8000/custom",
        )
        self.assertNotIn(
            "api_key",
            resolve_lmstudio_params(
                {"base_url": "https://remote.example", "model": "stub"}
            ),
        )

    def test_strict_fail_closed_configuration(self):
        for changes in [
            {"approval": "dry-run"},
            {"approval": None},
            {"enable_local_env": "false"},
            {"max_steps": 10**9},
            {"grounding_width": 0},
            {"unexpected": True},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AgentConfig.model_validate({**config().model_dump(), **changes})

    def test_default_openai_request_contract(self):
        from gui_agents.s3.core import engine

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="stub"))]
        )
        client = Mock()
        client.chat.completions.create.return_value = response
        messages = [{"role": "user", "content": "synthetic"}]
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "synthetic", "OPENAI_ORG_ID": "synthetic-org"},
        ), patch.object(engine, "OpenAI", return_value=client) as constructor:
            self.assertEqual(LMMEngineOpenAI(model="stub").generate(messages), "stub")
        constructor.assert_called_once_with(
            api_key="synthetic", organization="synthetic-org"
        )
        client.chat.completions.create.assert_called_once_with(
            model="stub", messages=messages, temperature=0.0
        )

    def test_provider_credentials_and_azure_target(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic"}):
            resolved = backend(config(provider="openai", model_url=""))
        self.assertEqual(resolved["base_url"], "https://api.openai.com/v1")
        azure = config(
            provider="azure",
            model_url="https://resource.openai.azure.com",
            model_api_key="azure-key",
            azure_api_version="2025-01-01",
        )
        url, headers = request_target(backend(azure), "chat/completions", "deployment")
        self.assertIn(
            "/openai/deployments/deployment/chat/completions?api-version=", url
        )
        self.assertEqual(headers["api-key"], "azure-key")

    def test_private_atomic_persistence(self):
        self.assertEqual(unprotect(protect("synthetic")), "synthetic")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            store.update({"model_api_key": "synthetic-secret"})
            self.assertNotIn("model_api_key", store.snapshot().public())
            if os.name == "nt":
                self.assertNotIn("synthetic-secret", path.read_text())
            self.assertEqual(
                ConfigStore(path).snapshot().model_api_key, "synthetic-secret"
            )
            store.update({"provider": "openai"})
            self.assertEqual(store.snapshot().model_api_key, "")

    def test_profile_migration_isolation_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            legacy = config().model_dump()
            legacy["model_api_key"] = "legacy-secret"
            path.write_text(json.dumps(legacy))
            store = ConfigStore(path)
            listed = store.list_profiles()
            self.assertEqual(len(listed["profiles"]), 1)
            self.assertEqual(store.snapshot().model_api_key, "legacy-secret")
            self.assertEqual(store.snapshot().profile_name, "Default")
            created = store.create_profile("Second")
            second_id = created["active_profile_id"]
            store.update({"model_api_key": "second-secret"})
            store.select_profile(listed["active_profile_id"])
            self.assertEqual(store.snapshot().model_api_key, "legacy-secret")
            store.select_profile(second_id)
            self.assertEqual(store.snapshot().model_api_key, "second-secret")
            renamed = store.rename_profile(second_id, "Renamed")
            self.assertIn("Renamed", [p["name"] for p in renamed["profiles"]])
            deleted = store.delete_profile(second_id)
            self.assertEqual(len(deleted["profiles"]), 1)
            with self.assertRaises(ValueError):
                store.delete_profile(deleted["active_profile_id"])

    def test_codex_backend_requires_sign_in(self):
        base = config().model_dump()
        base.update({"provider": "codex", "model": "gpt-6-astra", "model_url": "", "model_api_key": ""})
        cfg = AgentConfig.model_validate(base)
        with patch.object(
            codex_module, "codex_status", return_value={"signed_in": False, "error": "not signed in"}
        ):
            with self.assertRaises(ValueError):
                backend(cfg)
        with patch.object(
            codex_module,
            "codex_status",
            return_value={"signed_in": True, "model": "gpt-6-astra"},
        ):
            resolved = backend(cfg)
        self.assertEqual(resolved["provider"], "codex")
        self.assertEqual(resolved["model"], "gpt-6-astra")

    def test_codex_generate_returns_sdk_final_response(self):
        with patch.object(codex_module, "Codex") as factory:
            client = factory.return_value.__enter__.return_value
            client.account.return_value.account = SimpleNamespace(root=SimpleNamespace(type="chatgpt"))
            client.thread_start.return_value.run.return_value = SimpleNamespace(
                final_response="```python\nagent.wait(1.0)\n```"
            )
            text = codex_module.codex_generate([{"role": "user", "content": "hi"}], "m")
        self.assertIn("agent.wait", text)
        options = client.thread_start.call_args.kwargs
        self.assertEqual(options["sandbox"], codex_module.Sandbox.read_only)
        self.assertEqual(options["approval_mode"], codex_module.ApprovalMode.deny_all)
        self.assertIn("features.shell_tool=false", factory.call_args.kwargs["config"].config_overrides)


class StubManager:
    def __init__(self):
        self.launches = []

    def shutdown(self):
        pass

    def active(self):
        return 0

    def list(self):
        return []

    def launch(self, instruction, config):
        self.launches.append((instruction, config))
        return {"id": "stub", "status": "queued"}

    def public(self, task_id, detail="full"):
        return {
            "id": task_id,
            "status": "done",
            "steps": [],
            "logs": [],
            "error": None,
            "done_reason": "done",
            "version": 1,
        }

    def stop(self, task_id):
        return self.public(task_id)


class APITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = StubManager()
        self.captured = []

        def upstream(request):
            self.captured.append(request)
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "planner-stub"}]})
            payload = json.loads(request.content)
            if payload.get("stream"):
                return httpx.Response(
                    200,
                    content=b'data: {"choices": []}\n\ndata: [DONE]\n\n',
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                401,
                json={"error": {"message": "synthetic rejection"}},
                headers={"x-request-id": "stub-request"},
            )

        self.app = create_app(
            Path(self.directory.name) / "config.json",
            token="test-token",
            manager=self.manager,
            allowed_hosts=["testserver"],
            transport=httpx.MockTransport(upstream),
        )
        self.app.state.store.update(
            {
                "model": "planner-stub",
                "ground_model": "ground-stub",
                "model_api_key": "synthetic-secret",
            }
        )
        self.client = TestClient(self.app).__enter__()
        self.headers = {"Authorization": "Bearer test-token"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def test_auth_origin_secrets_assets(self):
        self.assertEqual(self.client.get("/api/config").status_code, 401)
        self.assertEqual(
            self.client.options(
                "/api/tasks", headers={"Origin": "https://foreign.example"}
            ).status_code,
            403,
        )
        response = self.client.get("/api/config", headers=self.headers)
        self.assertNotIn("synthetic-secret", response.text)
        self.assertIn('name="agent-s-token"', self.client.get("/").text)
        self.assertEqual(self.client.get("/static/main.js").status_code, 200)

    def test_invalid_overrides_do_not_launch(self):
        for changes in [
            {"approval": "dry-run"},
            {"enable_local_env": "false"},
            {"unexpected": 1},
        ]:
            response = self.client.post(
                "/api/tasks",
                headers=self.headers,
                json={"instruction": "test", "config": changes},
            )
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.manager.launches, [])

    def test_manual_v1_fails_closed(self):
        self.app.state.store.update({"approval": "manual"})
        response = self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json={
                "model": "agent-s3",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.manager.launches, [])

    def test_agent_v1_uses_shared_manager_and_stable_sse_id(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json={
                "model": "agent-s3",
                "stream": True,
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        self.assertEqual(len(self.manager.launches), 1)
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(len({chunk["id"] for chunk in chunks}), 1)
        self.assertEqual(response.headers["x-agent-s-task-id"], "stub")

    def test_proxy_preserves_errors_and_removes_internal_fields(self):
        body = {
            "model": "stub",
            "messages": [{"role": "user", "content": "test"}],
            "agent": False,
        }
        response = self.client.post(
            "/v1/chat/completions", headers=self.headers, json=body
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["x-request-id"], "stub-request")
        response = self.client.post(
            "/v1/chat/completions", headers=self.headers, json={**body, "stream": True}
        )
        self.assertIn("[DONE]", response.text)
        for request in self.captured:
            self.assertNotIn("agent", json.loads(request.content))

    def test_bad_bodies_are_validation_errors(self):
        for path in (
            "/api/tasks",
            "/api/models/list",
            "/api/profiles",
            "/api/profiles/select",
            "/v1/chat/completions",
        ):
            self.assertEqual(
                self.client.post(path, headers=self.headers, json=[]).status_code, 422
            )

    def test_profiles_and_task_profile_binding(self):
        created = self.client.post(
            "/api/profiles", headers=self.headers, json={"name": "Second"}
        )
        self.assertEqual(created.status_code, 200, created.text)
        second_id = created.json()["active_profile_id"]
        response = self.client.post(
            "/api/profiles/select",
            headers=self.headers,
            json={"profile_id": second_id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.client.get("/api/profiles", headers=self.headers).json()["active_profile_id"],
            second_id,
        )
        task = self.client.post(
            "/api/tasks",
            headers=self.headers,
            json={"instruction": "test", "config": {}, "profile_id": second_id},
        )
        self.assertEqual(task.status_code, 200, task.text)
        instruction, launched = self.manager.launches[-1]
        self.assertEqual(instruction, "test")
        self.assertEqual(launched.profile_id, second_id)
        self.assertEqual(launched.profile_name, "Second")

    def test_codex_status_and_discovery_use_sign_in(self):
        with patch(
            "gui_agents.s3.ui_server.codex_status",
            return_value={"signed_in": True, "account": "demo", "model": "gpt-6-astra"},
        ), patch("gui_agents.s3.ui_server.codex_models", return_value=["gpt-6-astra"]):
            status = self.client.get(
                "/api/codex/status", headers=self.headers, params={"model": "gpt-6-astra"}
            )
            self.assertTrue(status.json()["signed_in"])
            models = self.client.post(
                "/api/models/list",
                headers=self.headers,
                json={"aspect": "planner", "provider": "codex", "model": "gpt-6-astra"},
            )
            self.assertEqual(models.json()["models"], ["gpt-6-astra"])
        with patch(
            "gui_agents.s3.ui_server.test_codex_vision",
            return_value={"ok": True, "model": "gpt-6-astra", "sample": "blue"},
        ):
            tested = self.client.post(
                "/api/connection/test",
                headers=self.headers,
                json={"aspect": "planner", "provider": "codex", "model": "gpt-6-astra"},
            )
            self.assertTrue(tested.json()["ok"])

    def test_proxy_delivers_first_chunk_before_upstream_finishes(self):
        from gui_agents.s3.ui_server import ChatRequest

        async def probe():
            release = asyncio.Event()

            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b"data: first\n\n"
                    await release.wait()
                    yield b"data: [DONE]\n\n"

            transport = httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Stream())
            )
            async with httpx.AsyncClient(transport=transport) as client:
                original = self.app.state.client
                self.app.state.client = client
                try:
                    endpoint = next(
                        route.endpoint
                        for route in self.app.routes
                        if getattr(route, "path", None) == "/v1/chat/completions"
                    )
                    request = SimpleNamespace(
                        url=SimpleNamespace(
                            hostname="testserver", port=None, scheme="http"
                        )
                    )
                    response = await endpoint(
                        ChatRequest(
                            model="stub",
                            messages=[{"role": "user", "content": "test"}],
                            stream=True,
                        ),
                        request,
                    )
                    self.assertEqual(
                        await asyncio.wait_for(anext(response.body_iterator), 1),
                        b"data: first\n\n",
                    )
                    release.set()
                    self.assertIn(b"[DONE]", await anext(response.body_iterator))
                    await response.body_iterator.aclose()
                finally:
                    self.app.state.client = original

        asyncio.run(probe())


def synthetic_worker(instruction, config, events, stop, commands, start_gate):
    start_gate.wait(timeout=10)
    events.send(("state", {"status": "planning"}))
    if instruction == "await":
        record = {
            "n": 1,
            "status": "awaiting_approval",
            "approval_token": "a" * 32,
            "exec_code": "synthetic",
        }
        events.send(("step", record))
        command = commands.get(timeout=5)
        events.send(
            (
                "result",
                {
                    "done_reason": "done" if command["approved"] else "stopped",
                    "error": None,
                },
            )
        )
    else:
        # Ignore cancellation to verify supervisor termination.
        time.sleep(30)


class SupervisionTests(unittest.TestCase):
    def test_desktop_lease_blocks_another_owner(self):
        outcomes = []

        def attempt():
            try:
                with desktop_lease():
                    outcomes.append("acquired")
            except TaskConflict:
                outcomes.append("busy")

        with desktop_lease():
            thread = threading.Thread(target=attempt)
            thread.start()
            thread.join(timeout=3)
            self.assertEqual(outcomes, ["busy"])
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=3)
        self.assertEqual(outcomes, ["busy", "acquired"])

    def wait_for(self, manager, task_id, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = manager.public(task_id)
            if predicate(value):
                return value
            time.sleep(0.02)
        self.fail(f"Task did not reach expected state: {manager.public(task_id)}")

    def test_single_worker_approval_and_retention(self):
        manager = TaskManager(max_tasks=1)
        with patch("gui_agents.s3.task_manager._worker", synthetic_worker):
            task = manager.launch("await", config())
            task_id = task["id"]
            with self.assertRaises(TaskConflict):
                manager.launch("await", config())
            self.wait_for(
                manager, task_id, lambda value: value["pending_approval"] is not None
            )
            with self.assertRaises(TaskConflict):
                manager.approve(task_id, 1, "b" * 32, True)
            manager.approve(task_id, 1, "a" * 32, True)
            with self.assertRaises(TaskConflict):
                manager.approve(task_id, 1, "a" * 32, True)
            self.wait_for(
                manager, task_id, lambda value: value["status"] not in ACTIVE_STATES
            )
            new = manager.launch("await", config())
            self.assertEqual(len(manager.tasks), 1)
            self.wait_for(
                manager, new["id"], lambda value: value["pending_approval"] is not None
            )
            manager.approve(new["id"], 1, "a" * 32, False)
            self.wait_for(
                manager, new["id"], lambda value: value["status"] not in ACTIVE_STATES
            )
        manager.shutdown()

    def test_stop_terminates_uncooperative_worker(self):
        manager = TaskManager()
        with patch("gui_agents.s3.task_manager._worker", synthetic_worker):
            task = manager.launch("stall", config())
            manager.stop(task["id"])
            self.wait_for(
                manager, task["id"], lambda value: value["status"] == "stopped"
            )
            self.assertFalse(manager.tasks[task["id"]]["process"].is_alive())
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
