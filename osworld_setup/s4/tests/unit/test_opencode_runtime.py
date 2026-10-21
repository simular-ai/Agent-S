from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_s4.harness.opencode.config import build_config
from agent_s4.harness.opencode.source import OPENCODE_VERSION
from agent_s4.harness.opencode import runner as opencode_runner

RUNNER = Path(opencode_runner.__file__)


def config(env=None, context=None):
    return build_config('openai/test-model', 'SYSTEM', context,
                        env={'OPENAI_BASE_URL': 'http://127.0.0.1:12345/v1', **(env or {})})


def test_config_preserves_real_context_and_reserves_output_headroom():
    value = config(context={'context_window_tokens': 131072, 'reserve_tokens': 16384,
                            'keep_recent_tokens': 20000, 'enabled': True})
    assert value['provider']['shadow-openai']['models']['test-model']['limit'] == {
        'context': 131072, 'input': 131072, 'output': 32000,
    }
    assert value['compaction'] == {'auto': True, 'prune': False, 'reserved': 36096,
                                    'preserve_recent_tokens': 20000}
    assert config(context={'enabled': False})['provider']['shadow-openai']['models']['test-model']['limit']['context'] == 131072


def test_config_sampling_and_prime_environment_are_independent():
    value = config({'SHADOW_OPENCODE_TEMPERATURE': '0.6', 'SHADOW_OPENCODE_TOP_P': '0.9',
                    'SHADOW_PRIME_AGENT_TEMPERATURE': '1.8', 'SHADOW_PRIME_CONTEXT_WINDOW_SOURCE': 'catalog'})
    assert value['agent']['shadow']['temperature'] == 0.6
    assert value['agent']['shadow']['top_p'] == 0.9
    assert value['agent']['compaction']['temperature'] == 0.6


@pytest.mark.parametrize('values', [
    {'SHADOW_OPENCODE_TEMPERATURE': 'nan'}, {'SHADOW_OPENCODE_TOP_P': '1.1'},
    {'SHADOW_OPENCODE_MAX_OUTPUT_TOKENS': '0'}, {'SHADOW_OPENCODE_CONTEXT_WINDOW': '32000'},
])
def test_invalid_settings_fail_before_launch(values):
    with pytest.raises(ValueError):
        config(values)


def test_unmapped_prime_fields_are_reported():
    with pytest.warns(UserWarning, match='min_compact_tokens'):
        config(context={'min_compact_tokens': 4000})


def prepare_runner(tmp_path, source, *, task='task', settings=None):
    binary = tmp_path / 'fake-opencode'
    binary.write_text('#!' + sys.executable + '\nimport json,sys,os,time,subprocess\n'
                      f'if "--version" in sys.argv: print({OPENCODE_VERSION!r}); sys.exit(0)\n' + source)
    binary.chmod(0o755)
    (tmp_path / 'config.json').write_text(json.dumps(config()))
    (tmp_path / 'settings.json').write_text(json.dumps(settings or {'idle_timeout': 5}))
    (tmp_path / 'task.txt').write_text(task)
    return [sys.executable, str(RUNNER), '--binary', str(binary), '--config', str(tmp_path / 'config.json'),
            '--settings', str(tmp_path / 'settings.json'), '--task-file', str(tmp_path / 'task.txt'),
            '--state-dir', str(tmp_path / 'state'), '--cwd', str(tmp_path),
            '--stderr-file', str(tmp_path / 'stderr.log'), '--exit-file', str(tmp_path / 'exit.txt')]


def events(result):
    return [json.loads(line) for line in result.stdout.split('\n') if line.strip()]


def test_runner_uses_stdin_and_preserves_unicode_json(tmp_path):
    task = 'task containing app names, quotes, $(), and \u0085'
    argv = prepare_runner(tmp_path, '''
assert 'task containing' not in ' '.join(sys.argv)
assert sys.stdin.read().startswith('task containing')
print(json.dumps({'type':'text','sessionID':'ses_test','part':{'text':'a\\u0085b'}},ensure_ascii=False))
print(json.dumps({'type':'step_finish','sessionID':'ses_test','part':{'id':'p1','reason':'stop','cost':0.1}}))
''', task=task)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert events(result)[-1]['subtype'] == 'completed'
    assert events(result)[-1]['total_cost_usd'] == 0.1
    assert (tmp_path / 'exit.txt').read_text().strip() == '0'


@pytest.mark.parametrize('source, reason', [
    ('sys.exit(0)\n', 'incomplete'),
    ('print(json.dumps({"type":"error","error":{"message":"provider failed"}}))\n', 'provider_error'),
    ('print("not json")\n', 'protocol_error'),
    ('os.kill(os.getpid(), 9)\n', 'process_exit'),
])
def test_no_false_success_on_cli_failure(tmp_path, source, reason):
    result = subprocess.run(prepare_runner(tmp_path, source), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert events(result)[-1]['is_error'] is True
    assert events(result)[-1]['subtype'] == reason


def test_timeout_kills_owned_process_group_and_reports_budget(tmp_path):
    argv = prepare_runner(tmp_path, '''
child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
open('child.pid','w').write(str(child.pid))
print(json.dumps({'type':'step_finish','sessionID':'ses_test','part':{'id':'p1','reason':'tool-calls'}}),flush=True)
time.sleep(60)
''')
    result = subprocess.run(argv + ['--timeout', '1'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 124
    assert events(result)[-1]['subtype'] == 'wall_timeout'
    pid = int((tmp_path / 'child.pid').read_text())
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def test_continuation_resumes_exact_session_and_retains_step_budget(tmp_path):
    argv = prepare_runner(tmp_path, '''
if '--session' in sys.argv:
    assert sys.argv[sys.argv.index('--session') + 1] == 'ses_exact'
    reason='stop'
else:
    reason='length'
print(json.dumps({'type':'step_finish','sessionID':'ses_exact','part':{'id':reason,'reason':reason}}))
''', settings={'idle_timeout': 5, 'length_continuations': 1})
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout
    assert events(result)[-1]['steps'] == 2
    assert len([e for e in events(result) if e['type'] == 'opencode_length_continuation']) == 1
    result = subprocess.run(argv + ['--resume', '--max-steps', '2'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert events(result)[-1]['subtype'] == 'step_limit'


def test_resume_without_session_fails_instead_of_picking_latest(tmp_path):
    argv = prepare_runner(tmp_path, 'sys.exit(0)\n')
    result = subprocess.run(argv + ['--resume'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert events(result)[-1]['subtype'] == 'runner_error'


def test_idle_watchdog_is_distinct_from_task_timeout(tmp_path):
    argv = prepare_runner(tmp_path, 'time.sleep(60)\n', settings={'idle_timeout': 0.3})
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert events(result)[-1]['subtype'] == 'idle_timeout'


def test_explicit_signal_stops_runner_and_its_child(tmp_path):
    import signal
    import time
    argv = prepare_runner(tmp_path, "open('started','w').write(str(os.getpid()))\ntime.sleep(60)\n")
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / 'started').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / 'started').exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        assert json.loads(stdout.strip().split('\n')[-1])['subtype'] == 'signal'
        pid = int((tmp_path / 'started').read_text())
        assert not Path(f'/proc/{pid}').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_state_config_drift_rejected_on_resume(tmp_path):
    argv = prepare_runner(tmp_path, '''
print(json.dumps({'type':'step_finish','sessionID':'ses_test','part':{'reason':'stop'}}))
''')
    assert subprocess.run(argv, capture_output=True, timeout=15).returncode == 0
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['agent']['shadow']['temperature'] = 0.123
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    result = subprocess.run(argv + ['--resume'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert 'changed configuration' in result.stdout


def test_exhausted_length_continuations_end_scoreable_like_prime(tmp_path):
    # Prime ends an episode whose reply hit the output cap as a normal finish
    # once continuations run out, and the grader scores the VM state.
    argv = prepare_runner(tmp_path, """
print(json.dumps({'type':'step_finish','sessionID':'ses_len','part':{'id':str(time.time()),'reason':'length'}}))
""", settings={'idle_timeout': 5, 'length_continuations': 2})
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout
    final = events(result)[-1]
    assert final['subtype'] == 'length_limit' and final['is_error'] is False and final['steps'] == 3
    assert len([e for e in events(result) if e['type'] == 'opencode_length_continuation']) == 2


TOOLCALL_422 = """
if '--session' in sys.argv:
    print(json.dumps({'type':'step_finish','sessionID':'ses_tc','part':{'id':'done','reason':'stop'}}))
else:
    print(json.dumps({'type':'step_finish','sessionID':'ses_tc','part':{'id':'s1','reason':'tool-calls'}}))
    print(json.dumps({'type':'error','sessionID':'ses_tc','error':{'name':'APIError','data':{
        'message':'litellm.APIError: No valid tool call after 6 attempts: malformed syntax','statusCode':422}}}))
"""


def test_toolcall_error_recovers_in_same_session_when_enabled(tmp_path):
    argv = prepare_runner(tmp_path, TOOLCALL_422, settings={'idle_timeout': 5, 'toolcall_recoveries': 1})
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout
    records = events(result)
    assert records[-1]['subtype'] == 'completed' and records[-1]['steps'] == 2
    assert [r['type'] for r in records].count('opencode_toolcall_recovery') == 1
    # The recovered 422 is not left as a fatal `error` event for the validator.
    assert not any(r['type'] == 'error' for r in records)
    assert any(r['type'] == 'opencode_toolcall_error' for r in records)


def test_toolcall_error_stays_fatal_by_default(tmp_path):
    argv = prepare_runner(tmp_path, TOOLCALL_422)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert events(result)[-1]['subtype'] == 'provider_error'
