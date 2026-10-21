"""Synthetic overlay keeps official tasks intact and resolves local coordinator inputs."""
import json
from pathlib import Path

import pytest

from agent_s4 import synthetic_tasks
from agent_s4.eval import osworld_v2


def test_overlay_is_opt_in(monkeypatch):
    monkeypatch.delenv(synthetic_tasks.ROOT_ENV, raising=False)
    index = {'001': '/official/task_001.py'}
    synthetic_tasks.extend_task_index(index)
    assert index == {'001': '/official/task_001.py'}


def test_overlay_preserves_official_and_rejects_collision(tmp_path, monkeypatch):
    monkeypatch.setenv(synthetic_tasks.ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(synthetic_tasks, 'validate_package', lambda root: {
        'website_configuration': {'configured': True},
        'tasks': [{'id': 's_001', 'json_path': 'evaluation_examples/examples_v2/tasks/s_001.json'}],
    })
    index = {'001': '/official/task_001.py'}
    synthetic_tasks.extend_task_index(index)
    assert index['001'] == '/official/task_001.py'
    assert index['s_001'] == str(tmp_path / 'evaluation_examples/examples_v2/tasks/s_001.json')
    with pytest.raises(ValueError, match='collides'):
        synthetic_tasks.extend_task_index(index)


def test_loaded_asset_paths_work_outside_package_cwd(tmp_path, monkeypatch):
    package = tmp_path / 'package'
    asset = package / 'evaluation_examples/synthetic_assets/s_001/input.txt'
    asset.parent.mkdir(parents=True)
    asset.write_text('input')
    task_path = package / 'evaluation_examples/examples_v2/tasks/s_001.json'
    task_path.parent.mkdir(parents=True)
    relative = str(asset.relative_to(package))
    task_path.write_text(json.dumps({
        'id': 's_001', 'verification': 'dummy_constant_zero',
        'config': [{'type': 'download', 'parameters': {'files': [{'url': relative, 'path': '/home/user/input.txt'}]}},
                   {'type': 'chrome_open_tabs_with_state', 'parameters': {'state': [relative]}}],
    }))
    monkeypatch.chdir(tmp_path)
    loaded = osworld_v2.load_task(str(task_path), '/official/evaluation_examples')
    files = loaded['config'][0]['parameters']['files']
    assert files == [{'url': str(asset), 'path': '/home/user/input.txt'}]
    assert loaded['config'][1]['parameters']['state'] == [str(asset)]
    assert json.loads(task_path.read_text())['config'][0]['parameters']['files'][0]['url'] == relative


def test_missing_and_escaping_assets_rejected(tmp_path):
    with pytest.raises(ValueError, match='asset'):
        synthetic_tasks.resolve_assets('evaluation_examples/synthetic_assets/missing', tmp_path)
    outside = tmp_path.parent / 'outside-input.txt'
    outside.write_text('outside')
    path = tmp_path / 'evaluation_examples/synthetic_assets/link'
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='asset'):
        synthetic_tasks.resolve_assets('evaluation_examples/synthetic_assets/link', tmp_path)


def test_official_json_is_unchanged():
    task = {'id': 'official', 'config': [{'parameters': {'url': 'relative/path'}}]}
    assert synthetic_tasks.prepare_loaded_task(task, '/official/examples/task.json') is task


def test_setup_honors_check_and_delegates_other_actions(monkeypatch):
    calls = []

    class Controller:
        def setup(self, config, proxy):
            calls.append(('native', config, proxy))
            return True

    monkeypatch.setattr(synthetic_tasks, 'checked_guest_execute', lambda c, cmd: calls.append(('checked', cmd)))
    monkeypatch.setattr(synthetic_tasks, 'set_guest_utc', lambda c: calls.append(('tz',)))
    monkeypatch.setattr(synthetic_tasks, 'verify_browser_utc', lambda c: calls.append(('verify',)))
    web = {'type': 'chrome_open_tabs_with_state', 'parameters': {'url': ['https://eventix.example']}}
    launch = {'type': 'launch', 'parameters': {'command': ['google-chrome', '--remote-debugging-port=1337']}}
    task = synthetic_tasks.SyntheticTask(id='s_028', config=[
        {'type': 'execute', 'parameters': {'command': ['python3', '-c', 'pass'], 'check': True, 'quiet': True}},
        launch, web,
    ])
    task.setup(Controller())
    # The guest clock is set before Chrome launches; the browser check runs
    # after that launch and still precedes the timed website seeding.
    assert calls == [
        ('tz',),
        ('checked', ['python3', '-c', 'pass']),
        ('native', [launch], False),
        ('verify',),
        ('native', [web], False),
    ]


def test_setup_rejects_utc_task_without_web_step(monkeypatch):
    class Controller:
        def setup(self, config, proxy):
            return True

    monkeypatch.setattr(synthetic_tasks, 'set_guest_utc', lambda c: None)
    monkeypatch.setattr(synthetic_tasks, 'verify_browser_utc',
                        lambda c: pytest.fail('must not run without a web step'))
    task = synthetic_tasks.SyntheticTask(id='s_020', config=[
        {'type': 'launch', 'parameters': {'command': ['google-chrome']}},
    ])
    with pytest.raises(RuntimeError, match='browser UTC'):
        task.setup(Controller())


def test_checked_execute_rejects_guest_nonzero(monkeypatch):
    from types import SimpleNamespace
    import requests

    monkeypatch.setattr(requests, 'post', lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'returncode': 1, 'error': 'missing application'},
    ))
    with pytest.raises(RuntimeError, match='missing application'):
        synthetic_tasks.checked_guest_execute(SimpleNamespace(http_server='http://guest'), ['false'])


def _web_package(tmp_path, config):
    """Minimal package holding one web task, enough for ensure_chrome_launch."""
    import hashlib
    task_path = tmp_path / 'evaluation_examples/examples_v2/tasks/s_003.json'
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(json.dumps({'id': 's_003', 'config': config}, indent=2) + '\n')
    manifest_path = tmp_path / 'evaluation_examples/synthetic_assets/PACKAGE_MANIFEST.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'tasks': [{
            'id': 's_003', 'web': [{'app': 'teamchat'}],
            'json_path': 'evaluation_examples/examples_v2/tasks/s_003.json',
            'web_config_index': config.index(
                next(s for s in config if s['type'] == 'chrome_open_tabs_with_state')),
        }],
        'files_sha256': {
            'evaluation_examples/examples_v2/tasks/s_003.json':
                hashlib.sha256(task_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return task_path, manifest_path


def test_ensure_chrome_launch_inserts_steps_and_reindexes(tmp_path):
    import hashlib
    web = {'type': 'chrome_open_tabs_with_state', 'parameters': {'url': ['https://teamchat.example']}}
    task_path, manifest_path = _web_package(tmp_path, [
        {'type': 'download', 'parameters': {'files': []}}, web,
    ])

    assert synthetic_tasks.ensure_chrome_launch(tmp_path) == ['s_003']

    config = json.loads(task_path.read_text())['config']
    assert [s['type'] for s in config] == [
        'download', 'launch', 'launch', 'chrome_open_tabs_with_state']
    assert config[1]['parameters']['command'] == ['google-chrome', '--remote-debugging-port=1337']
    assert config[2]['parameters']['command'] == ['socat', 'tcp-listen:9222,fork', 'tcp:localhost:1337']

    manifest = json.loads(manifest_path.read_text())
    assert manifest['tasks'][0]['web_config_index'] == 3
    assert manifest['files_sha256']['evaluation_examples/examples_v2/tasks/s_003.json'] == \
        hashlib.sha256(task_path.read_bytes()).hexdigest()


def test_ensure_chrome_launch_is_idempotent(tmp_path):
    web = {'type': 'chrome_open_tabs_with_state', 'parameters': {'url': ['https://teamchat.example']}}
    task_path, _ = _web_package(tmp_path, [{'type': 'download', 'parameters': {'files': []}}, web])
    synthetic_tasks.ensure_chrome_launch(tmp_path)
    settled = task_path.read_bytes()

    assert synthetic_tasks.ensure_chrome_launch(tmp_path) == []
    assert task_path.read_bytes() == settled


def test_ensure_chrome_launch_reports_without_applying(tmp_path):
    web = {'type': 'chrome_open_tabs_with_state', 'parameters': {'url': ['https://teamchat.example']}}
    task_path, _ = _web_package(tmp_path, [{'type': 'download', 'parameters': {'files': []}}, web])
    original = task_path.read_bytes()

    assert synthetic_tasks.ensure_chrome_launch(tmp_path, apply=False) == ['s_003']
    assert task_path.read_bytes() == original


def test_verify_browser_utc_retries_socket_hang_up(monkeypatch):
    """socat answers on 9222 before Chrome binds 1337; the first connects drop."""
    from types import SimpleNamespace

    attempts = []

    class Page:
        def evaluate(self, script):
            return 'UTC' if 'timeZone' in script else 0

        def close(self):
            pass

    class Browser:
        contexts = [SimpleNamespace(new_page=lambda: Page())]

    class Chromium:
        def connect_over_cdp(self, endpoint):
            attempts.append(endpoint)
            if len(attempts) < 3:
                raise RuntimeError('BrowserType.connect_over_cdp: socket hang up')
            return Browser()

    class Playwright:
        chromium = Chromium()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr('playwright.sync_api.sync_playwright', lambda: Playwright())
    monkeypatch.setattr('time.sleep', lambda s: None)

    synthetic_tasks.verify_browser_utc(
        SimpleNamespace(vm_ip='10.0.0.1', chromium_port=9222)
    )
    assert len(attempts) == 3
