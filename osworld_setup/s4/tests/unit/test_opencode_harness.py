"""OpenCode VM command, installation, and artifact contracts."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent_s4.harness.opencode import evaluator as opencode
from agent_s4.harness.opencode.source import OPENCODE_VERSION, VM_BINARY
from agent_s4.vm.utils import AgentCommand


def command(**kwargs):
    return opencode.build_opencode_vm_command(
        instruction='Close unique-target-application', system_prompt='SYSTEM_PROMPT_TEST',
        model='openrouter/qwen/test-model', timeout=900, api_key="test-key-'-$()",
        **kwargs,
    )


def test_vm_command_keeps_tasks_and_credentials_out_of_argv():
    result = command()
    assert isinstance(result, AgentCommand)
    assert 'unique-target-application' not in result.script
    assert 'SYSTEM_PROMPT_TEST' not in result.script
    assert "test-key-'-$()" not in result.script
    assert result.secret_env == {'OPENROUTER_API_KEY': "test-key-'-$()"}
    assert '--task-file /tmp/claude_task/instruction.txt' in result.script
    assert '$(cat ' not in result.script
    assert base64.b64encode(b'Close unique-target-application').decode() in result.script
    assert VM_BINARY in result.script
    assert result.accepted_exit_codes == (0, 124)
    assert result.runner_exit_grace == 30
    subprocess.run(['bash', '-n'], input=result.script, text=True, check=True)


def test_vm_resume_preserves_state_and_passes_only_file_paths():
    result = command(continue_session=True, no_time_limit=True, max_steps=123)
    assert 'rm -rf /tmp/claude_task/opencode-state' not in result.script
    assert '--resume' in result.script
    assert '--timeout ' not in result.script
    assert '--max-steps 123' in result.script


def test_vm_fresh_task_resets_only_opencode_state():
    result = command()
    assert 'rm -rf /tmp/claude_task/opencode-state' in result.script
    assert 'rm -rf /tmp/claude_task &&' not in result.script
    assert '/home/user/AGENTS.md' not in result.script
    assert '/home/user/.config/opencode' not in result.script


def test_install_fails_closed_without_valid_cache():
    with patch.object(opencode, 'validate_runtime_cache', side_effect=RuntimeError('bad cache')), \
         patch.object(opencode, '_execute_shell_on_vm') as execute:
        assert opencode.install_opencode_on_vm(MagicMock()) is False
    execute.assert_not_called()


def test_install_only_reuses_pinned_version():
    with patch.object(opencode, 'validate_runtime_cache'), \
         patch.object(opencode, '_execute_shell_on_vm', return_value={'returncode': 0, 'output': OPENCODE_VERSION}) as execute, \
         patch.object(opencode, 'upload_file_to_vm') as upload:
        assert opencode.install_opencode_on_vm(MagicMock()) is True
    assert execute.call_args.args[1] == f'{VM_BINARY} --version'
    upload.assert_not_called()


def test_install_replaces_wrong_version_with_verified_cache():
    with patch.object(opencode, 'validate_runtime_cache'), \
         patch.object(opencode, '_execute_shell_on_vm', side_effect=[
             {'returncode': 0, 'output': '0.0.1'}, {'returncode': 0, 'output': OPENCODE_VERSION},
         ]), patch.object(opencode, 'upload_file_to_vm', return_value=True) as upload:
        assert opencode.install_opencode_on_vm(MagicMock()) is True
    upload.assert_called_once()


def artifacts(reason='completed', exit_code=0, **extra):
    terminal = {'type': 'result', 'is_error': False, 'subtype': reason,
                'steps': 2, 'session_id': 'ses_test', 'exit_code': exit_code, **extra}
    return {'claude_trajectory': json.dumps(terminal).encode(), 'exit_code': str(exit_code).encode()}


@pytest.mark.parametrize('reason,code', [('completed',0), ('step_limit',0), ('wall_timeout',124), ('length_limit',0)])
def test_scoreable_artifacts(reason, code):
    assert opencode.validate_opencode_artifacts(artifacts(reason,code)) is None


@pytest.mark.parametrize('value', [
    {}, {'claude_trajectory': b'{broken'}, artifacts(is_error=True), artifacts(steps=0),
    artifacts(reason='length_limit', exit_code=1), artifacts(reason='incomplete'),
    {'claude_trajectory': b'{"type":"step_finish"}'},
    {**artifacts(), 'exit_code': b'137'},
])
def test_invalid_or_incomplete_artifacts_are_not_success(value):
    assert opencode.validate_opencode_artifacts(value)


def test_registry_attaches_validator():
    assert opencode.MODEL_CONFIG['opencode']['validate_artifacts'] is opencode.validate_opencode_artifacts


def test_vm_uses_private_endpoint_without_changing_builder(monkeypatch):
    from agent_s4.harness.opencode.config import build_config
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://public.example:4000/v1')
    monkeypatch.setenv('SHADOW_EVOL_VM_OPENAI_BASE_URL', 'http://private.example:4000/v1')
    monkeypatch.delenv('SHADOW_OPENCODE_BASE_URL', raising=False)
    monkeypatch.delenv('SHADOW_OPENCODE_VM_BASE_URL', raising=False)
    vm = build_config('openai/test', 'system', env=opencode.vm_config_env('openai/test'))
    host = build_config('openai/test', 'system')
    assert vm['provider']['shadow-openai']['options']['baseURL'] == 'http://private.example:4000/v1'
    assert host['provider']['shadow-openai']['options']['baseURL'] == 'http://public.example:4000/v1'


def test_explicit_opencode_endpoint_overrides_legacy_vm_route(monkeypatch):
    monkeypatch.setenv('SHADOW_EVOL_VM_OPENAI_BASE_URL', 'http://legacy.example/v1')
    monkeypatch.setenv('SHADOW_OPENCODE_BASE_URL', 'http://shared.example/v1')
    monkeypatch.delenv('SHADOW_OPENCODE_VM_BASE_URL', raising=False)
    assert opencode.vm_config_env('openai/test')['SHADOW_OPENCODE_BASE_URL'] == 'http://shared.example/v1'
    monkeypatch.setenv('SHADOW_OPENCODE_VM_BASE_URL', 'http://vm.example/v1')
    assert opencode.vm_config_env('openai/test')['SHADOW_OPENCODE_BASE_URL'] == 'http://vm.example/v1'


def test_legacy_openai_vm_route_does_not_redirect_openrouter(monkeypatch):
    monkeypatch.setenv('SHADOW_EVOL_VM_OPENAI_BASE_URL', 'http://private.example/v1')
    monkeypatch.delenv('SHADOW_OPENCODE_BASE_URL', raising=False)
    monkeypatch.delenv('SHADOW_OPENCODE_VM_BASE_URL', raising=False)
    assert 'SHADOW_OPENCODE_BASE_URL' not in opencode.vm_config_env('openrouter/test')


def test_hermes_tool_format_is_opt_in(monkeypatch):
    from agent_s4.harness.opencode.config import HERMES_FORMAT_NOTE, build_config
    env = {'OPENAI_BASE_URL': 'http://x.example/v1'}
    plain = build_config('openai/m', 'SYS', env=env, hermes_plugin_path='/p.js')
    assert 'plugin' not in plain
    assert plain['agent']['shadow']['prompt'] == 'SYS'
    hermes = build_config('openai/m', 'SYS', env={**env, 'SHADOW_OPENCODE_HERMES_TOOL_FORMAT': '1'},
                          hermes_plugin_path='/p.js')
    assert hermes['plugin'] == ['file:///p.js']
    assert hermes['agent']['shadow']['prompt'] == 'SYS' + HERMES_FORMAT_NOTE
    with pytest.raises(ValueError):
        build_config('openai/m', 'SYS', env={**env, 'SHADOW_OPENCODE_HERMES_TOOL_FORMAT': 'yes'})


def test_vm_command_stages_hermes_plugin(monkeypatch):
    monkeypatch.setenv('SHADOW_OPENCODE_HERMES_TOOL_FORMAT', '1')
    result = command()
    plugin = opencode.HERMES_PLUGIN.read_bytes()
    assert base64.b64encode(plugin).decode() in result.script
    assert opencode.VM_HERMES_PLUGIN in result.script
    subprocess.run(['bash', '-n'], input=result.script, text=True, check=True)


def test_summary_override_wins_over_interface():
    from agent_s4.harness.opencode.config import build_config
    env = {'OPENAI_BASE_URL': 'http://x.example/v1'}
    context = {'summary_max_tokens': 4000}
    models = lambda c: c['provider']['shadow-openai']['models']
    assert models(build_config('openai/m', 'S', context, env=env))['m--shadow-compaction']['limit']['output'] == 4000
    overridden = build_config('openai/m', 'S', context,
                              env={**env, 'SHADOW_OPENCODE_SUMMARY_MAX_TOKENS': '28876'})
    assert models(overridden)['m--shadow-compaction']['limit']['output'] == 28876
    assert models(overridden)['m']['limit']['output'] == 32000
    with pytest.raises(ValueError):
        build_config('openai/m', 'S', context, env={**env, 'SHADOW_OPENCODE_SUMMARY_MAX_TOKENS': '40000'})


def test_runtime_installs_before_task_setup_can_revoke_sudo():
    # V2 task 048 revokes sudo during setup; a post-setup install then fails.
    assert opencode.MODEL_CONFIG['opencode']['install_before_task_setup'] is True


def test_post_setup_install_probe_needs_no_sudo(tmp_path, monkeypatch):
    calls = []
    def execute(controller, command, timeout):
        calls.append(command)
        return {'returncode': 0, 'output': OPENCODE_VERSION + '\n'}
    monkeypatch.setattr(opencode, 'validate_runtime_cache', lambda path: {})
    monkeypatch.setattr(opencode, '_execute_shell_on_vm', execute)
    monkeypatch.setattr(opencode, 'upload_file_to_vm', lambda *a: pytest.fail('must not re-upload'))
    assert opencode.install_opencode_on_vm(object(), 'pw') is True
    assert calls == [f'{VM_BINARY} --version']


def test_disabled_tools_are_withheld_from_the_agent():
    from agent_s4.harness.opencode.config import build_config
    env = {'OPENAI_BASE_URL': 'http://x.example/v1'}
    assert 'tools' not in build_config('openai/m', 'S', env=env)['agent']['shadow']
    cfg = build_config('openai/m', 'S', env={**env, 'SHADOW_OPENCODE_DISABLED_TOOLS': 'write, todowrite'})
    # write and edit share one native permission, so both are withheld.
    assert cfg['agent']['shadow']['tools'] == {'edit': False, 'todowrite': False, 'write': False}
    with pytest.raises(ValueError):
        build_config('openai/m', 'S', env={**env, 'SHADOW_OPENCODE_DISABLED_TOOLS': 'bash'})


def test_toolcall_recoveries_setting_is_absent_unless_enabled():
    from agent_s4.harness.opencode.config import runner_settings
    assert 'toolcall_recoveries' not in runner_settings({})
    assert runner_settings({'SHADOW_OPENCODE_TOOLCALL_RECOVERIES': '2'})['toolcall_recoveries'] == 2


def test_bash_timeout_cap_plugin_is_opt_in(monkeypatch):
    from agent_s4.harness.opencode.config import build_config
    env = {'OPENAI_BASE_URL': 'http://x.example/v1'}
    assert 'plugin' not in build_config('openai/m', 'S', env=env, tool_limits_plugin_path='/l.js')
    cfg = build_config('openai/m', 'S', env={**env, 'SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS': '600000'},
                       tool_limits_plugin_path='/l.js')
    assert cfg['plugin'] == [['file:///l.js', {'maxBashTimeoutMs': 600000}]]
    both = build_config('openai/m', 'S', env={**env, 'SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS': '600000',
                                               'SHADOW_OPENCODE_HERMES_TOOL_FORMAT': '1'},
                        hermes_plugin_path='/h.js', tool_limits_plugin_path='/l.js')
    assert both['plugin'] == ['file:///h.js', ['file:///l.js', {'maxBashTimeoutMs': 600000}]]
    with pytest.raises(ValueError):
        build_config('openai/m', 'S', env={**env, 'SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS': '3600000'})
    monkeypatch.setenv('SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS', '600000')
    script = command().script
    assert base64.b64encode(opencode.TOOL_LIMITS_PLUGIN.read_bytes()).decode() in script
    assert opencode.VM_TOOL_LIMITS_PLUGIN in script
