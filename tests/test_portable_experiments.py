"""Public experiment entry points keep qualification and result boundaries."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from scripts.experiments import execution as runner
from scripts.experiments import model_experiments as workflow


def test_catalog_matrix_preserves_protocol():
    catalog = json.loads((runner.ROOT / 'scripts/model_downloads/models.json').read_text())
    assert len(catalog) == 21
    for key in catalog:
        rows = runner.matrix(key)
        assert len(rows) == len({runner.row_id(r) for r in rows}) == 42
        for row in rows:
            config = json.loads((runner.ROOT / row['config']).read_text())
            for name, value in dict(agent=row['mode'], seed=row['seed'], count=3600,
                                    duration=15, speculative=False, cojev_history_steps=0,
                                    eightphase=False, fallback='previous', jev_retries=3,
                                    jev_timeout=120.0, min_confidence=0.0).items():
                assert config[name] == value, (key, row, name)


@pytest.fixture
def ready_report(tmp_path):
    artifacts, modes = {}, {}
    for mode in workflow.MODES:
        pair = []
        for i in range(2):
            directory = tmp_path / mode / str(i)
            directory.mkdir(parents=True)
            summary = directory / 'summary.json'
            summary.write_text(json.dumps({'jev': {'failures': 0}}))
            trace = directory / 'jev_decisions.jsonl'
            trace.write_text(json.dumps({'step': 15, 'intersection': 'junction', 'signal': 1})+'\n')
            artifacts.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary, trace)})
            pair.append(dict(status='completed', summary_path=str(summary), aql=1, awt=2, att=3,
                             simulation_duration_s=60, model_decisions=1))
        modes[mode] = dict(status='ready', capacity_probe_mapping_valid=True, largest_encoded_tokens=100,
                           short_profile=dict(warmup=3, measure=10), short_profile_gpu_rows=[4],
                           single_request_gpu_rows=1, smoke_speedup=1.1, sumo_behavior_valid=True,
                           full_step_target=196, smoke=pair)
    mapping = tmp_path / 'native_mapping.json'
    mapping.write_text('[]')
    artifacts[str(mapping)] = hashlib.sha256(mapping.read_bytes()).hexdigest()
    return dict(status='ready', identity=dict(inputs={'revision': 'frozen'}, gpu_uuid='test-device'),
                encoded_case_count=98, native_input_mapping_valid=True, modes=modes, artifacts=artifacts,
                mapping_artifact=str(mapping), mode_batch_sizes={mode: 196 for mode in workflow.MODES})


def test_gate_requires_both_modes_and_current_artifacts(ready_report):
    report = ready_report
    assert workflow.report_ready(report, {'revision': 'frozen'}, 'test-device')
    assert not workflow.report_ready(report, {'revision': 'changed'}, 'test-device')
    assert not workflow.report_ready(report, {'revision': 'frozen'}, 'other-device')
    broken = copy.deepcopy(report)
    del broken['modes']['cojevlight']
    assert not workflow.report_ready(broken, {'revision': 'frozen'}, 'test-device')
    Path(report['mapping_artifact']).write_text('changed')
    assert not workflow.report_ready(report, {'revision': 'frozen'}, 'test-device')


@pytest.mark.parametrize('field,value', [('smoke_speedup', float('nan')), ('smoke_speedup', 1.0),
    ('short_profile_gpu_rows', [1]), ('capacity_probe_mapping_valid', False),
    ('short_profile', {'warmup': 3, 'measure': 2})])
def test_partial_or_unaccelerated_batch_cannot_pass(ready_report, field, value):
    ready_report['modes']['cojevlight'][field] = value
    assert not workflow.report_ready(ready_report, {'revision': 'frozen'}, 'test-device')


def test_missing_scope_preparation_fails_before_gpu_import(tmp_path):
    with pytest.raises(RuntimeError, match='missing'):
        workflow.require_scope_ready(tmp_path, ['laya'])


def test_execute_keeps_configuration_and_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    row = dict(model_key='test', mode='jevlight', traffic_id='jinan-1', seed=1, config='test.json')
    monkeypatch.setattr(runner, 'matrix', lambda _: [row])
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'benchmark': 'test'})
    config = tmp_path / 'test.json'
    config.write_text(json.dumps(dict(agent='jevlight', count=3600)))
    original = config.read_bytes()
    (tmp_path / 'jevlight/backends').mkdir(parents=True)
    (tmp_path / 'jevlight/backends/decision_model_service.py').write_text('')
    model = tmp_path / '.local/models/test'
    model.mkdir(parents=True)
    (model / 'manifest.json').write_text('{}')
    monkeypatch.setattr(runner, 'run_child', lambda *args: 1)
    result = runner.execute(row, SimpleNamespace(log_path=None))
    assert result['status'] == 'failed'
    assert (Path(result['attempt_path']) / 'record.json').is_file()
    assert config.read_bytes() == original
    with pytest.raises(ValueError, match='protocol'):
        runner.execute(row, SimpleNamespace(), {'duration': 30})


def test_matrix_summary_preserves_seeds_and_paired_differences():
    records = [dict(row, status='completed', aql=row['seed'], awt=2*row['seed'],
                    att=row['seed'] + (1 if row['mode'] == 'cojevlight' else 0),
                    summary_path=f'run-{i}/summary.json') for i,row in enumerate(runner.matrix('laya'))]
    summary = workflow.summarize(records)
    assert summary['condition_count'] == 42
    assert summary['groups'][0]['metrics']['aql']['jevlight']['sample_std'] == 1
    assert summary['groups'][0]['metrics']['att']['co_minus_local']['values'] == [1, 1, 1]
    with pytest.raises(ValueError):
        workflow.summarize(records[:-1] + records[:1])


def test_backend_validation_is_instance_local(monkeypatch):
    from jevlight.backends import decision_model_service as service
    calls = []
    monkeypatch.setattr(service, 'validate_answers', lambda request, result: calls.append('default') or result)
    def backend():
        value = service.NativeBackend.__new__(service.NativeBackend)
        value.lock = threading.Lock()
        value.log_path = None
        value.runtime = SimpleNamespace(encode=lambda request: {}, native=lambda request: {'answers': {}})
        return value
    special, ordinary = backend(), backend()
    special.answer_validator = lambda request, result: calls.append('special') or result
    special.evaluate({}, {})
    ordinary.evaluate({}, {})
    assert calls == ['special', 'default']


@pytest.mark.parametrize('visible', [None, '0', '1,0'])
def test_standard_cuda_visibility_does_not_require_device_identity(monkeypatch, visible):
    from jevlight.backends.cuda_runtime import prepare_cuda_device
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(version=SimpleNamespace(cuda='12.8')))
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    prepare_cuda_device()


def test_cuda_engine_rejects_incompatible_abi(monkeypatch):
    from jevlight.backends.cuda_runtime import prepare_cuda_device
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(version=SimpleNamespace(cuda='12.1')))
    with pytest.raises(RuntimeError, match='CUDA 12.8'):
        prepare_cuda_device()
