"""Batch diagnostics loaded by new workers without replacing a resident runtime."""
import json
from pathlib import Path
import threading
import time
from types import MethodType
import uuid

from jevlight.jev_client import JevClient, JevResult
from jevlight.backends.step_batch import enable_step_batch
from jevlight.step_batch_transport import make_step_server as make_legacy_step_server


class BatchBackendError(RuntimeError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class StepBatchClient(JevClient):
    """Keep batch decisions/retries intact and retain backend error bodies."""

    def evaluate_batch(self, requests):
        import requests as http

        if not requests:
            return []
        if not self.batch_size or self.transport != 'local':
            raise ValueError('local batch transport is not enabled')
        if self.mock:
            return [self._mock_evaluate(item['state'], item['questions'], time.time())
                    for item in requests]
        self._pace()
        started = time.time()
        headers = {'Content-Type': 'application/json'}
        if self.api_key is not None:
            headers['Authorization'] = f'Bearer {self.api_key}'
        for attempt in range(self.max_retries + 1):
            try:
                response = http.post(f'{self.base_url}/v1/systemone/batch',
                    json={'requests': requests, 'batch_size': self.batch_size},
                    headers=headers, timeout=self.timeout)
                try:
                    try:
                        payload = response.json()
                    except ValueError:
                        response.raise_for_status()
                        raise
                    if isinstance(payload, dict) and payload.get('error'):
                        error = payload['error']
                        message = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                        status = error.get('http_status', response.status_code) if isinstance(error, dict) else response.status_code
                        raise BatchBackendError(message, status)
                    response.raise_for_status()
                    if not isinstance(payload, dict) or 'results' not in payload:
                        raise ValueError('batch response is missing results')
                    outcomes = payload['results']
                    if not isinstance(outcomes, list) or len(outcomes) != len(requests):
                        raise ValueError('batch result count mismatch')
                    return [JevResult(answers=outcome['answers'], model=outcome.get('model'),
                        input_tokens=int(outcome.get('usage', {}).get('input_tokens', 0)),
                        output_tokens=int(outcome.get('usage', {}).get('output_tokens', 0)),
                        latency=time.time()-started) for outcome in outcomes]
                finally:
                    response.close()
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                status = getattr(exc, 'status', getattr(getattr(exc, 'response', None), 'status_code', None))
                return [JevResult(error=f'{type(exc).__name__}: {exc}',
                    latency=time.time()-started, http_status=status,
                    rate_limited=self._is_rate_limited(str(exc))) for _ in requests]


def _write_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{threading.get_ident()}.tmp')
    try:
        temporary.write_text(json.dumps(event, ensure_ascii=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress_path(backend):
    log = getattr(backend, 'log_path', None)
    return Path(log).with_name('batch_progress.json') if log else getattr(backend, 'step_progress_path', None)


def encoded_request_length(encodings):
    """Measure native token/path reports without interpreting missing fields as zero."""
    lengths = []
    for row in encodings:
        for name in ('encoded_tokens', 'logical_tokens', 'candidate_path_tokens'):
            value = row.get(name)
            values = value if isinstance(value, list) else [value]
            lengths.extend(v for v in values if type(v) is int and v > 0)
    if not lengths:
        raise ValueError('Native encoding has no positive token or candidate-path length')
    return max(lengths)


def _reporting_probe(self, requests, requested):
    result = self._step_reporting_probe(requests, requested)
    path = _progress_path(self._step_reporting_backend)
    if path:
        _write_event(path, dict(kind='gpu_capacity_chunk_completed', updated_at=time.time(),
            batch_requests=len(result[0]), gpu_forward_rows=result[2][-1]['gpu_forward_rows']))
    return result


def _reporting_batch(self, requests, batch_size):
    requests = list(requests)
    backend = self._step_reporting_backend
    path = _progress_path(backend)
    progress = self._step_progress
    previous = getattr(progress, 'notify', None)
    event = dict(kind='gpu_chunk_completed', request_id=uuid.uuid4().hex,
                 batch_requests=len(requests), completed_chunks=0)
    if path:
        _write_event(path, dict(event, kind='batch_initialized', updated_at=time.time()))

    def completed():
        event['completed_chunks'] += 1
        if path:
            _write_event(path, dict(event, updated_at=time.time()))
        if previous is not None:
            previous()

    progress.notify = completed
    try:
        return self._step_reporting_native(requests, batch_size)
    except Exception as error:
        if path:
            _write_event(Path(path).with_name('batch_error.json'), dict(event,
                kind='batch_failed', updated_at=time.time(), requests=requests,
                error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        progress.notify = previous


def prepare_step_backend(backend):
    """Initialize progress before exposing either validation or formal endpoints."""
    from jevlight.backends.native_row_batch import enable_decider_full_rows, enable_kev_full_rows, enable_valen_full_rows
    family = getattr(backend, 'config', {}).get('family')
    if backend.config.get('inference_engine', 'author') == 'vllm_pooling':
        family = None
    elif backend.config.get('inference_engine', 'author') == 'vllm_marker':
        family = None
    elif backend.config.get('inference_engine', 'author') == 'vllm_native_heads':
        family = None
    if family == 'decider':
        enable_decider_full_rows(backend.runtime)
        linear = backend.config.get('batch_linear', 'author')
        if linear == 'native_shape':
            from jevlight.backends.native_row_batch import enable_decider_native_shapes
            enable_decider_native_shapes(backend.runtime)
        elif linear != 'author':
            raise ValueError(f'Unsupported native Decider projection: {linear}')
        kernel = backend.config.get('batch_kernel', 'author')
        if kernel == 'causal_conv':
            from jevlight.backends.native_row_batch import enable_native_batch_conv
            enable_native_batch_conv(backend.runtime)
        elif kernel != 'author':
            raise ValueError(f'Unsupported native Decider batch kernel: {kernel}')
    elif family == 'kev':
        batch_backend = backend.config.get('batch_backend', 'author')
        if batch_backend == 'native_cached':
            from jevlight.backends.kev_native_cache_batch import enable_kev_native_cache_batch
            enable_kev_native_cache_batch(backend.runtime)
        elif batch_backend == 'author':
            enable_kev_full_rows(backend.runtime)
        elif batch_backend == 'author_fused':
            from jevlight.backends.kev_author_fused_batch import enable_kev_author_fusion
            root = Path(__file__).resolve().parents[1]
            enable_kev_author_fusion(backend.runtime, root/backend.config['batch_fla_package_path'])
            enable_kev_full_rows(backend.runtime)
        else:
            raise ValueError(f'Unsupported native Kev batch backend: {batch_backend}')
        kernel = backend.config.get('batch_kernel', 'author')
        if kernel == 'causal_conv':
            from jevlight.backends.native_row_batch import enable_native_batch_conv
            enable_native_batch_conv(backend.runtime)
        elif kernel != 'author':
            raise ValueError(f'Unsupported native Kev batch kernel: {kernel}')
    elif family == 'valen':
        enable_valen_full_rows(backend.runtime)
        linear = backend.config.get('batch_linear', 'author')
        if linear == 'native_shape':
            from jevlight.backends.native_row_batch import enable_valen_native_shapes
            enable_valen_native_shapes(backend.runtime)
        elif linear != 'author':
            raise ValueError(f'Unsupported native Valen projection: {linear}')
    elif family == 'open-jev':
        linear = backend.config.get('batch_linear', 'author')
        if linear == 'native_shape':
            from jevlight.backends.native_row_batch import enable_open_native_shapes
            enable_open_native_shapes(backend.runtime)
        elif linear != 'author':
            raise ValueError(f'Unsupported native Open-Jev projection: {linear}')
        attention = backend.config.get('co_batch_attention', 'author')
        if attention == 'native_flash':
            from jevlight.backends.native_row_batch import enable_open_co_native_flash
            enable_open_co_native_flash(backend.runtime)
        elif attention != 'author':
            raise ValueError(f'Unsupported native Open-Jev cooperative attention: {attention}')
        if backend.config.get('batch_delta', 'author') in ('fla_fp32_co', 'fla_fp32_all'):
            from jevlight.backends.open_jev_fla_batch import enable_open_fla_batch
            root = Path(__file__).resolve().parents[1]
            enable_open_fla_batch(backend.runtime, root/backend.config['batch_fla_package_path'],
                packed_mlp=backend.config.get('batch_mlp', 'author') == 'native_packed',
                all_modes=backend.config.get('batch_delta') == 'fla_fp32_all',
                full_linear=backend.config.get('batch_projection', 'native') == 'full_gpu')
    elif family == 'clef':
        from jevlight.backends.native_row_batch import enable_clef_text_batch_probe
        enable_clef_text_batch_probe(backend.runtime)
        linear = backend.config.get('batch_linear', 'author')
        if linear == 'native_shape':
            from jevlight.backends.native_row_batch import enable_clef_native_shapes
            enable_clef_native_shapes(backend.runtime, fused_scatter=backend.config.get('batch_fused_ops', False))
        elif linear == 'native_beta_shape':
            from jevlight.backends.native_row_batch import enable_clef_native_beta_shapes
            enable_clef_native_beta_shapes(backend.runtime)
        elif linear != 'author':
            raise ValueError(f'Unsupported native Clef gate projection: {linear}')
        attention = backend.config.get('batch_attention', 'author')
        if attention == 'native_flash':
            from jevlight.backends.native_row_batch import enable_clef_native_flash
            enable_clef_native_flash(backend.runtime)
        elif attention != 'author':
            raise ValueError(f'Unsupported native Clef batch attention: {attention}')
        kernel = backend.config.get('batch_kernel', 'author')
        if kernel == 'causal_conv':
            from jevlight.backends.native_row_batch import enable_native_batch_conv
            enable_native_batch_conv(backend.runtime)
        elif kernel != 'author':
            raise ValueError(f'Unsupported native Clef batch kernel: {kernel}')
    elif family == 'lev':
        linear = backend.config.get('batch_linear', 'author')
        if linear == 'native_shape':
            from jevlight.backends.native_row_batch import enable_lev_native_shapes
            enable_lev_native_shapes(backend.runtime, fused_scatter=backend.config.get('batch_fused_ops', False))
        elif linear == 'native_gate_shape':
            from jevlight.backends.native_row_batch import enable_lev_native_gate_shapes
            enable_lev_native_gate_shapes(backend.runtime)
        elif linear == 'native_first_layer_shape':
            from jevlight.backends.native_row_batch import enable_lev_native_first_layer
            enable_lev_native_first_layer(backend.runtime)
        elif linear != 'author':
            raise ValueError(f'Unsupported native Lev gate projection: {linear}')
        attention = backend.config.get('batch_attention', 'author')
        if attention == 'native_flash':
            from jevlight.backends.native_row_batch import enable_lev_native_flash
            enable_lev_native_flash(backend.runtime)
        elif attention != 'author':
            raise ValueError(f'Unsupported native Lev batch attention: {attention}')
        kernel = backend.config.get('batch_kernel', 'author')
        if kernel == 'causal_conv':
            from jevlight.backends.native_row_batch import enable_native_batch_conv
            enable_native_batch_conv(backend.runtime)
        elif kernel != 'author':
            raise ValueError(f'Unsupported native Lev batch kernel: {kernel}')
    if family == 'lev':
        if backend.config.get('batch_fused_ops', False):
            from jevlight.backends.native_fused_ops import enable_native_fused_ops
            enable_native_fused_ops(backend.runtime, family)
            if backend.config.get('batch_mlp', 'author') == 'native_packed':
                from jevlight.backends.native_packed_mlp import enable_lev_packed_mlp
                enable_lev_packed_mlp(backend.runtime)
    elif family == 'clef':
        if backend.config.get('batch_fused_ops', False):
            from jevlight.backends.native_fused_ops import enable_native_fused_ops
            enable_native_fused_ops(backend.runtime, family)
    enable_step_batch(backend)
    runtime = backend.runtime
    if not hasattr(runtime, '_step_progress'):
        runtime._step_progress = threading.local()
    if hasattr(runtime, '_step_author_batch') and not hasattr(runtime, '_step_reporting_native'):
        runtime._step_reporting_backend = backend
        runtime._step_reporting_native = runtime.native_batch
        runtime.native_batch = MethodType(_reporting_batch, runtime)
        runtime._step_reporting_probe = runtime.probe_step_capacity
        runtime.probe_step_capacity = MethodType(_reporting_probe, runtime)
    if family == 'lev':
        if backend.config.get('batch_encoding_cache', False):
            from jevlight.backends.native_encoding_cache import enable_native_encoding_cache
            enable_native_encoding_cache(runtime, family)
    elif family == 'clef':
        if backend.config.get('batch_encoding_cache', False):
            from jevlight.backends.native_encoding_cache import enable_native_encoding_cache
            enable_native_encoding_cache(runtime, family)


def make_step_server(backend, host, port, model_name):
    prepare_step_backend(backend)
    server = make_legacy_step_server(backend, host, port, model_name)
    parent = server.RequestHandlerClass

    class ReportingHandler(parent):
        def _send(self, status, payload):
            if status >= 400 and isinstance(payload.get('error'), dict):
                payload = dict(payload, error=dict(payload['error'], http_status=status))
            super()._send(status, payload)

    server.RequestHandlerClass = ReportingHandler
    return server
