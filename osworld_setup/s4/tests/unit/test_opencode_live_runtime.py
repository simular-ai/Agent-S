"""Opt-in real CLI tests against a local fake provider (no paid API or VMs).

OPENCODE_TEST_BINARY=/path/to/pinned/opencode pytest -q this_file.py
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import pytest

from agent_s4.harness.opencode.config import build_config
from agent_s4.harness.opencode import runner as opencode_runner

pytestmark = pytest.mark.skipif(not os.environ.get('OPENCODE_TEST_BINARY'), reason='pinned real CLI not supplied')


@pytest.fixture
def provider():
    calls = []
    responses = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            response = responses.pop(0) if responses else {'content': 'Task complete.', 'reason': 'stop'}
            delta = {'role': 'assistant', 'content': response.get('content', '')}
            if 'tool_calls' in response:
                delta['tool_calls'] = response['tool_calls']
            data = [
                {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'test-model',
                 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'test-model',
                 'choices': [{'index': 0, 'delta': {}, 'finish_reason': response['reason']}],
                 'usage': {'prompt_tokens': response.get('input_tokens', 20), 'completion_tokens': 5, 'total_tokens': response.get('input_tokens', 20) + 5}},
            ]
            payload = ''.join('data: ' + json.dumps(d) + '\n\n' for d in data) + 'data: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(payload.encode())))
            self.end_headers()
            self.wfile.write(payload.encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls, responses
    finally:
        server.shutdown()
        server.server_close()


def launch(tmp_path, url, task='Complete the task.', context=None, resume=False, model='openai/test-model',
           config_env=None, hermes_plugin_path=None):
    cfg = build_config(model, 'Complete the task using tools when necessary.', context,
        env={'SHADOW_OPENCODE_BASE_URL': url, 'SHADOW_OPENCODE_TEMPERATURE': '0.6', 'SHADOW_OPENCODE_TOP_P': '0.9',
             'SHADOW_OPENCODE_MAX_OUTPUT_TOKENS': '1024', 'SHADOW_OPENCODE_REQUEST_TIMEOUT': '15',
             **(config_env or {})},
        hermes_plugin_path=hermes_plugin_path)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    (tmp_path / 'task.txt').write_text(task)
    argv = [sys.executable, str(Path(opencode_runner.__file__)), '--binary', os.environ['OPENCODE_TEST_BINARY'],
            '--config', str(tmp_path / 'config.json'), '--task-file', str(tmp_path / 'task.txt'),
            '--state-dir', str(tmp_path / 'state'), '--cwd', str(tmp_path),
            '--stderr-file', str(tmp_path / 'stderr.log'), '--timeout', '60', '--max-steps', '10']
    if resume:
        argv.append('--resume')
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ['API_KEY', 'TOKEN', 'PROXY'])}
    env.update({'OPENAI_API_KEY': 'local-test-only', 'OPENROUTER_API_KEY': 'local-test-only',
                'PARASAIL_API_KEY': 'local-test-only', 'HOME': str(tmp_path)})
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr + (tmp_path / 'stderr.log').read_text()
    return [json.loads(line) for line in result.stdout.split('\n') if line.strip()]


def test_real_cli_sends_sampling_and_output_cap_and_resumes(tmp_path, provider):
    url, calls, responses = provider
    records = launch(tmp_path, url, task='Remember the secret task marker pebble123.')
    assert records[-1]['subtype'] == 'completed'
    assert len(calls) == 1
    assert calls[0]['temperature'] == 0.6
    assert calls[0]['top_p'] == 0.9
    assert calls[0].get('max_tokens', calls[0].get('max_completion_tokens')) == 1024
    old_session = records[-1]['session_id']
    records = launch(tmp_path, url, task='Continue now.', resume=True)
    assert records[-1]['session_id'] == old_session
    assert 'pebble123' in json.dumps(calls[-1]['messages'])


@pytest.mark.parametrize('prefix', ['openrouter', 'parasail'])
def test_other_provider_prefixes_use_chat_completions(tmp_path, provider, prefix):
    url, calls, responses = provider
    records = launch(tmp_path, url, model=f'{prefix}/test-model')
    assert records[-1]['subtype'] == 'completed'
    assert calls[0]['model'] == 'test-model'
    assert calls[0]['temperature'] == 0.6


def test_real_cli_pkill_target_does_not_kill_its_own_agent(tmp_path, provider):
    url, calls, responses = provider
    marker = 'shadow-target-' + uuid.uuid4().hex
    victim = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)', marker])
    responses.append({'reason': 'tool_calls', 'tool_calls': [{
        'index': 0, 'id': 'call_test', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({
            'command': f'pkill -f {marker}', 'description': 'Stop the target application',
        })},
    }]})
    try:
        records = launch(tmp_path, url, task=f'Close the application named {marker}, then confirm completion.')
        assert any(r['type'] == 'tool_use' for r in records)
        assert records[-1]['subtype'] == 'completed'
        victim.wait(timeout=5)
        assert victim.returncode != 0
    finally:
        if victim.poll() is None:
            victim.kill()
        victim.wait()


def test_real_cli_compacts_at_declared_input_budget(tmp_path, provider):
    url, calls, responses = provider
    responses.extend([
        {'reason': 'tool_calls', 'input_tokens': 25000, 'tool_calls': [{
            'index': 0, 'id': 'call_compact', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({
                'command': 'printf compact-probe', 'description': 'Inspect task state',
            })},
        }]},
        {'reason': 'stop', 'content': 'SUMMARY_MARKER: Continue the task after the probe.'},
        {'reason': 'stop', 'content': 'Task complete after compaction.'},
    ])
    context = {'context_window_tokens': 32768, 'reserve_tokens': 8192, 'summary_max_tokens': 128,
               'keep_recent_tokens': 2000, 'summary_instructions': 'Preserve COMPACTION_TEST_INSTRUCTION and task state.'}
    records = launch(tmp_path, url, task='Run a probe and complete the task.', context=context)
    assert records[-1]['subtype'] == 'completed'
    assert len(calls) >= 3
    summary_requests = [c for c in calls if 'COMPACTION_TEST_INSTRUCTION' in json.dumps(c['messages'])]
    assert summary_requests, json.dumps(calls)
    assert summary_requests[0]['model'] == 'test-model'
    assert summary_requests[0]['max_tokens'] == 128
    assert 'SUMMARY_MARKER' in json.dumps(calls[-1]['messages'])


def test_real_builder_runs_inside_v2_landlock(tmp_path, provider, monkeypatch):
    from agent_s4.builder_sandbox import landlock_abi
    from agent_s4.harness.opencode.builder import OpenCodeBuilder
    if not landlock_abi():
        pytest.skip('Landlock unavailable')
    url, calls, responses = provider
    clone = tmp_path / 'OSWorldV2_Evol_opencode_probe'
    clone.mkdir()
    (clone / 'shadow_interface').mkdir()
    gated = tmp_path / 'gated-secret.txt'
    gated.write_text('must not be readable')
    responses.append({'reason': 'tool_calls', 'tool_calls': [{
        'index': 0, 'id': 'call_landlock', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({
            'command': f'if cat {gated}; then exit 1; fi; printf survived > shadow_interface/probe.txt',
            'description': 'Verify workspace isolation',
        })},
    }]})
    monkeypatch.setenv('SHADOW_OPENCODE_BIN', os.environ['OPENCODE_TEST_BINARY'])
    monkeypatch.setenv('OPENAI_BASE_URL', url)
    monkeypatch.setenv('OPENAI_API_KEY', 'local-test-only')
    monkeypatch.setenv('SHADOW_OPENCODE_REQUEST_TIMEOUT', '15')
    result = OpenCodeBuilder().launch(
        clone_dir=clone, prompt='Check the allowed workspace and finish.', model='openai/test-model',
        timeout=60, env_extra=None, variant_id='probe', logs_dir=tmp_path / 'logs', log_name='probe',
    )
    assert result.exit_code == 0 and not result.crashed, result.log_path.read_text()
    assert (clone / 'shadow_interface' / 'probe.txt').read_text() == 'survived'
    assert 'Permission denied' in result.log_path.read_text()


@pytest.mark.parametrize('enabled', [False, True])
def test_real_cli_hermes_plugin_strips_leaked_calls_from_outgoing_history(tmp_path, provider, enabled):
    from agent_s4.harness.opencode.config import HERMES_PLUGIN
    url, calls, responses = provider
    leaked = '<tool_call>\n{"function="bash", "arguments": {"command": "ls"}}\n</tool_call>'
    responses.append({'reason': 'tool_calls', 'content': 'Checking.\n' + leaked, 'tool_calls': [{
        'index': 0, 'id': 'call_leak', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({
            'command': 'printf leak-probe', 'description': 'Probe',
        })},
    }]})
    records = launch(tmp_path, url, config_env={'SHADOW_OPENCODE_HERMES_TOOL_FORMAT': '1' if enabled else '0'},
                     hermes_plugin_path=HERMES_PLUGIN)
    assert records[-1]['subtype'] == 'completed'
    assert len(calls) == 2
    history = [m for m in calls[1]['messages'] if m['role'] == 'assistant']
    assert ('function=' in json.dumps(history)) is not enabled
    # The stored session, and so the recorded trajectory, keeps the original text.
    assert any('function=' in json.dumps(r) for r in records if r.get('type') == 'text')
    assert ('Tool-call format' in json.dumps(calls[0]['messages'])) is enabled
