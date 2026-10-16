"""Whole control-step native batches with automatic memory capacity tiers."""
import gc
import json
import math
import time
import threading
from types import MethodType


def _backbone(runtime):
    model = getattr(runtime, 'model', None)
    if model is None:
        model = getattr(getattr(runtime, 'agent', None), 'model', None)
    for name in ('encoder', 'backbone', 'lm'):
        value = getattr(model, name, None)
        if value is not None:
            # Decider.slot_logits calls lm.model directly, bypassing the
            # causal-LM wrapper and its unused vocabulary projection.
            if name == 'lm' and getattr(value, 'model', None) is not None:
                return value.model
            return value
    return getattr(getattr(model, 'path_encoder', None), 'backbone', None) or getattr(
        getattr(runtime, 'engine', None), '_decoder', None)


def _bucket(requests):
    # Growth crosses a bucket and triggers a fresh whole-step capacity attempt.
    # No state fields are removed or changed to calculate this scheduling key.
    longest = max((len(json.dumps(r, ensure_ascii=False)) for r in requests), default=0)
    co = any('cooperative_view' in r.get('state', {}) for r in requests)
    return co, math.ceil(longest / 2048)


def _try_group(runtime, group, cap):
    import torch
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    observed = []
    def hook(module, args, kwargs):
        tensor = kwargs.get('input_ids')
        if tensor is None:
            tensor = kwargs.get('inputs_embeds')
        if tensor is None:
            tensor = next((v for v in args if torch.is_tensor(v) and v.is_cuda and v.ndim >= 2), None)
        if tensor is not None and tensor.is_cuda:
            observed.append(int(tensor.shape[0]))
    backbone = _backbone(runtime)
    handle = backbone.register_forward_pre_hook(hook, with_kwargs=True) if backbone is not None else None
    started = time.monotonic()
    try:
        result = runtime._step_author_batch(group, cap)
        torch.cuda.synchronize()
        if getattr(runtime, 'last_engine_forward_rows', None) is not None:
            observed[:] = runtime.last_engine_forward_rows
        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        margin = max(4096 * 1024**2, total * .1)
        free_lower = free - max(0, peak_reserved - reserved)
        evidence = dict(requests=len(group), capacity=cap, gpu_forward_rows=observed,
                        wall_ms=(time.monotonic()-started)*1000,
                        peak_allocated_mib=peak/1024**2, peak_reserved_mib=peak_reserved/1024**2,
                        peak_free_lower_bound_mib=free_lower/1024**2,
                        required_margin_mib=margin/1024**2)
        runtime.step_run_peak_allocated_mib=max(getattr(runtime,'step_run_peak_allocated_mib',0),peak/1024**2)
        runtime.step_run_peak_reserved_mib=max(getattr(runtime,'step_run_peak_reserved_mib',0),peak_reserved/1024**2)
        if free_lower < margin:
            return None, dict(evidence, status='insufficient_memory_margin')
        if len(result) != len(group):
            raise ValueError('native whole-step batch result count mismatch')
        return result, dict(evidence, status='ready')
    except torch.cuda.OutOfMemoryError as error:
        return None, dict(requests=len(group), capacity=cap, status='cuda_oom', reason=str(error))
    except RuntimeError as error:
        # Large native tensors can exceed a CUDA kernel's 32-bit index range
        # while memory still fits. Reduce request capacity, retaining all tokens.
        if 'canUse32BitIndexMath(input) && canUse32BitIndexMath(output)' not in str(error):
            raise
        return None, dict(requests=len(group), capacity=cap, status='native_kernel_index_limit',
                          gpu_forward_rows=observed, reason=str(error))
    except ValueError as error:
        # Respect an explicit author request-count limit; do not mask other bugs.
        if 'batch' not in str(error).lower() or not any(s in str(error).lower() for s in ('32', 'states', 'between')):
            raise
        return None, dict(requests=len(group), capacity=cap, status='author_capacity_limit', reason=str(error))
    finally:
        if handle is not None:
            handle.remove()


def _capacity(runtime, requests, requested):
    import torch
    bucket = _bucket(requests)
    cap = min(requested, len(requests), runtime._step_capacity.get(bucket, requested))
    attempts = []
    while cap:
        result, evidence = _try_group(runtime, requests[:cap], cap)
        attempts.append(evidence)
        if result is not None:
            if len(attempts) > 1:
                runtime._step_capacity[bucket] = cap
            return result, cap, attempts
        if cap == 1:
            raise RuntimeError('One complete native request cannot retain GPU memory margin')
        # Keep the final multi-request opportunity: an unsafe capacity of 3
        # must try 2 before reaching a single complete request.
        cap = max(1, (cap + 1) // 2)
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError('No complete request batch fits the GPU')


def probe_capacity(self, requests, requested):
    """Measure capacity using the first chunk only, without running a long step."""
    return _capacity(self, requests, min(requested, len(requests)))


def native_step_batch(self, requests, batch_size):
    if type(batch_size) is not int or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    requests = list(requests)
    if not requests:
        return []
    cursor, results, attempts, sizes = 0, [], [], []
    started = time.monotonic()
    while cursor < len(requests):
        group, cap, evidence = _capacity(self, requests[cursor:], batch_size)
        results.extend(group)
        cursor += len(group)
        attempts.extend(evidence)
        sizes.extend(evidence[-1]['gpu_forward_rows'])
        notify=getattr(self._step_progress,'notify',None)
        if notify is not None and not getattr(self, 'emits_gpu_chunk_progress', False):
            notify()
    report = dict(request_count=len(requests), requested_capacity=batch_size,
                  gpu_forward_rows=sizes, whole_step_one_forward=len(sizes)==1,
                  wall_ms=(time.monotonic()-started)*1000,
                  capacity_attempts=attempts)
    report['capacity_cache']=[dict(co=k[0],char_bucket=k[1],capacity=v) for k,v in self._step_capacity.items()]
    self.last_step_batch = report
    for result in results:
        result.setdefault('usage', {})['step_batch'] = report
    return results


def enable_step_batch(backend, maximum=196):
    runtime = backend.runtime
    if not hasattr(runtime, 'native_batch') or hasattr(runtime, '_step_author_batch'):
        return
    backend.config['batch_size'] = max(maximum, int(backend.config.get('batch_size', 16)))
    runtime._step_author_batch = runtime.native_batch
    runtime._step_capacity = {}
    runtime._step_progress=threading.local()
    runtime.native_batch = MethodType(native_step_batch, runtime)
    runtime.probe_step_capacity = MethodType(probe_capacity, runtime)
