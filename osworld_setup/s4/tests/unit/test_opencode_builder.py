from pathlib import Path
import json
from unittest.mock import patch

from agent_s4.harness.base import BuilderResult
from agent_s4.harness.opencode.builder import OpenCodeBuilder


def test_builder_uses_files_and_shared_config_without_exposing_task_in_argv(tmp_path, monkeypatch):
    clone = tmp_path / 'clone'
    clone.mkdir()
    (clone / 'shadow_interface').mkdir()
    (clone / 'shadow_interface' / 'context_compaction.json').write_text(json.dumps({
        'context_window_tokens': 131072, 'keep_recent_tokens': 15000,
    }))
    monkeypatch.setenv('SHADOW_OPENCODE_TEMPERATURE', '0.7')
    captured = {}
    def fake_run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return BuilderResult(0, 0.1, tmp_path / 'log.jsonl', False)
    with patch('agent_s4.harness.opencode.builder.resolve_binary', return_value=Path('/opt/opencode/opencode')), \
         patch('agent_s4.harness.opencode.builder.run_builder_subprocess', side_effect=fake_run):
        result = OpenCodeBuilder().launch(
            clone_dir=clone, prompt='Kill unique-app-name', model='openrouter/qwen/test', timeout=None,
            env_extra=None, variant_id='v1', logs_dir=tmp_path / 'logs', log_name='build',
        )
    assert not result.crashed
    assert 'unique-app-name' not in ' '.join(captured['argv'])
    argv = captured['argv']
    config = json.loads(Path(argv[argv.index('--config')+1]).read_text())
    assert config['agent']['shadow']['temperature'] == 0.7
    assert config['compaction']['preserve_recent_tokens'] == 15000
    assert Path(argv[argv.index('--task-file')+1]).read_text() == 'Kill unique-app-name'
    assert captured['timeout'] is None
    assert captured['sandbox_read_paths'] == [Path('/opt/opencode')]
    assert (tmp_path / 'logs' / 'build.opencode-config.json').exists()


def test_opencode_builder_registered():
    from agent_s4.harness import resolve_builder_harness
    assert isinstance(resolve_builder_harness('opencode'), OpenCodeBuilder)
