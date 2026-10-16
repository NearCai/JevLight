"""Deployment selection preserves explicit adapter and kernel boundaries."""
from types import SimpleNamespace

import pytest

from scripts.experiments.serve_decision_model import build_backend


def test_step_runtime_preserves_model_configuration(monkeypatch):
    from jevlight.backends import decision_model_service
    config = {'family': 'open-jev', 'inference_engine': 'vllm_pooling', 'batch_size': 256}
    observed = []
    backend = object()
    monkeypatch.setattr(decision_model_service, 'NativeBackend',
                        lambda value: observed.append(value) or backend)
    assert build_backend(config) is backend
    assert observed == [config]


def test_native_runtime_installs_requested_attention_after_model_load(monkeypatch):
    from jevlight.backends import decision_model_service, native_batch_runtime
    calls = []
    backend = SimpleNamespace(config={'family': 'gliner2'})
    monkeypatch.setattr(native_batch_runtime, 'configure_execution', lambda: calls.append('configure'))
    monkeypatch.setattr(decision_model_service, 'NativeBackend', lambda value: calls.append('load') or backend)
    monkeypatch.setattr(native_batch_runtime, 'install', lambda value, **kw: calls.append((value, kw)))
    assert build_backend(backend.config, batch_runtime='native', attention='deberta_tiled') is backend
    assert calls == ['configure', 'load', (backend, {'attention': 'deberta_tiled'})]


@pytest.mark.parametrize('options', [
    {'batch_runtime': 'native'},
    {'attention': 'deberta_tiled'},
    {'selection': {}, 'batch_runtime': 'native'},
    {'selection': {}, 'attention': 'laya_sdpa'},
])
def test_incompatible_kernel_selection_fails_before_model_load(monkeypatch, options):
    from jevlight.backends import decision_model_service
    monkeypatch.setattr(decision_model_service, 'NativeBackend',
                        lambda value: pytest.fail('Model loaded before configuration validation'))
    with pytest.raises(ValueError):
        build_backend({'inference_engine': 'vllm_pooling'}, **options)
