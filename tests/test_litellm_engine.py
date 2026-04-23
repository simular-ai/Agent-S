"""Tests for the LiteLLM engine across all Agent-S versions (s2, s2_5, s3)."""

import types as builtin_types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fake response helpers
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content="hello"):
        self.content = content


class _Choice:
    def __init__(self, content="hello"):
        self.message = _Msg(content=content)


class _Response:
    def __init__(self, content="hello"):
        self.choices = [_Choice(content=content)]


# ---------------------------------------------------------------------------
# Helpers to inject a fake litellm module
# ---------------------------------------------------------------------------


def _install_fake_litellm(response_content="hello"):
    import sys

    fake = builtin_types.ModuleType("litellm")
    fake.completion = mock.MagicMock(return_value=_Response(response_content))
    sys.modules["litellm"] = fake
    return fake


def _uninstall_fake_litellm():
    import sys

    sys.modules.pop("litellm", None)


# ---------------------------------------------------------------------------
# s3 engine tests
# ---------------------------------------------------------------------------


class TestS3EngineLiteLLM:
    def setup_method(self):
        self.fake = _install_fake_litellm("s3 response")

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_generate_returns_content(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key="test-key", model="openai/gpt-4o")
        result = engine.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_new_tokens=100,
        )
        assert result == "s3 response"

    def test_generate_passes_drop_params(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key="test-key", model="anthropic/claude-3-haiku")
        engine.generate(
            messages=[{"role": "user", "content": "test"}],
        )
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_generate_passes_model(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(model="bedrock/anthropic.claude-v2")
        engine.generate(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["model"] == "bedrock/anthropic.claude-v2"

    def test_generate_forwards_api_key(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key="sk-test", model="openai/gpt-4o")
        engine.generate(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"

    def test_generate_omits_api_key_when_none(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key=None, model="openai/gpt-4o")
        engine.generate(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert "api_key" not in call_kwargs

    def test_instance_temperature_overrides_param(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(model="openai/gpt-4o", temperature=0.9)
        engine.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
        )
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_default_max_tokens(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(model="openai/gpt-4o")
        engine.generate(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# s2 engine tests
# ---------------------------------------------------------------------------


class TestS2EngineLiteLLM:
    def setup_method(self):
        self.fake = _install_fake_litellm("s2 response")

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_generate_returns_content(self):
        from gui_agents.s2.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key="test-key", model="openai/gpt-4o")
        result = engine.generate(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "s2 response"

    def test_drop_params_default(self):
        from gui_agents.s2.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(model="openai/gpt-4o")
        engine.generate(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["drop_params"] is True


# ---------------------------------------------------------------------------
# s2_5 engine tests
# ---------------------------------------------------------------------------


class TestS25EngineLiteLLM:
    def setup_method(self):
        self.fake = _install_fake_litellm("s2_5 response")

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_generate_returns_content(self):
        from gui_agents.s2_5.core.engine import LMMEngineLiteLLM

        engine = LMMEngineLiteLLM(api_key="test-key", model="openai/gpt-4o")
        result = engine.generate(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "s2_5 response"


# ---------------------------------------------------------------------------
# LMMAgent registration tests
# ---------------------------------------------------------------------------


class TestLMMAgentRegistration:
    def setup_method(self):
        self.fake = _install_fake_litellm("agent response")

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_s3_agent_creates_litellm_engine(self):
        from gui_agents.s3.core.engine import LMMEngineLiteLLM
        from gui_agents.s3.core.mllm import LMMAgent

        agent = LMMAgent(
            engine_params={"engine_type": "litellm", "model": "openai/gpt-4o"},
        )
        assert isinstance(agent.engine, LMMEngineLiteLLM)

    def test_s3_agent_get_response(self):
        from gui_agents.s3.core.mllm import LMMAgent

        agent = LMMAgent(
            engine_params={
                "engine_type": "litellm",
                "model": "openai/gpt-4o",
                "api_key": "test",
            },
            system_prompt="You are helpful.",
        )
        resp = agent.get_response(user_message="hi")
        assert resp == "agent response"

    def test_s2_agent_creates_litellm_engine(self):
        from gui_agents.s2.core.engine import LMMEngineLiteLLM
        from gui_agents.s2.core.mllm import LMMAgent

        agent = LMMAgent(
            engine_params={"engine_type": "litellm", "model": "openai/gpt-4o"},
        )
        assert isinstance(agent.engine, LMMEngineLiteLLM)

    def test_s2_5_agent_creates_litellm_engine(self):
        from gui_agents.s2_5.core.engine import LMMEngineLiteLLM
        from gui_agents.s2_5.core.mllm import LMMAgent

        agent = LMMAgent(
            engine_params={"engine_type": "litellm", "model": "openai/gpt-4o"},
        )
        assert isinstance(agent.engine, LMMEngineLiteLLM)

    def test_unsupported_engine_raises(self):
        from gui_agents.s3.core.mllm import LMMAgent

        with pytest.raises(ValueError, match="not supported"):
            LMMAgent(engine_params={"engine_type": "nonexistent", "model": "x"})
