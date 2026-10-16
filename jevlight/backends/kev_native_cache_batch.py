"""Batch the author's ephemeral state-prefix and question-branch GPU passes."""
from contextlib import contextmanager
from types import MethodType


@contextmanager
def _prefix_execution(runtime, meta):
    """Restore native scalar state and release captured tails after any failure."""
    previous = (runtime._native_prefix_active, runtime._native_prefix_meta, runtime._native_gate_ranges)
    runtime._native_prefix_active, runtime._native_prefix_meta, runtime._native_gate_ranges = True, meta, None
    try:
        yield
    finally:
        runtime._native_prefix_active, runtime._native_prefix_meta, runtime._native_gate_ranges = previous
        for layer in runtime.model.lm.modules():
            if hasattr(layer, '_native_conv_tail'):
                layer._native_conv_tail = None


def _native_cached_group(runtime, requests):
    import torch
    from transformers.cache_utils import DynamicCache
    from kev.model import rows_of
    from kev.api import to_answers
    packed = [runtime.encode_native(request) for request in requests]
    splits = [rows_of(encoded) for encoded, _ in packed]
    if any(not state or not rows for state, _, rows in splits):
        raise ValueError('Native cached batch needs complete states and question branches')
    model, device = runtime.model, runtime.model.device
    ids, positions, valid = model._pad_rows([(state, pos) for state, pos, _ in splits])
    lengths = torch.tensor([len(state) for state, _, _ in splits], device=device)
    at = torch.arange(ids.shape[1], device=device)
    allow = ((at[None, None, :] <= at[None, :, None]) & valid[:, None, :].bool())
    allow |= at[None, None, :] == at[None, :, None]
    dtype = next(model.lm.parameters()).dtype
    mask = torch.zeros(allow.shape, device=device, dtype=dtype).masked_fill(~allow, float('-inf'))[:, None]
    cache = DynamicCache(config=model.lm.config)
    meta = dict(valid=valid.bool(), left=torch.zeros_like(lengths), end=lengths,
                ranges=[(0, len(state)) for state, _, _ in splits])
    with _prefix_execution(runtime, meta):
        # The cache lives only for this GPU chunk; there is no reuse of traffic
        # state across requests or control steps.
        model.lm(input_ids=ids, position_ids=positions,
                 attention_mask=dict(full_attention=mask, linear_attention=valid),
                 past_key_values=cache, use_cache=True)
        runtime._native_prefix_meta = None
        owner = [index for index, (_, _, rows) in enumerate(splits) for _ in rows]
        rows = [row for _, _, branches in splits for row in branches]
        index = torch.tensor(owner, device=device)
        if owner != list(range(len(splits))):
            cache.reorder_cache(index)
        branch_ids, branch_positions, branch_valid = model._pad_rows([(row['ids'], row['pos']) for row in rows])
        width = branch_ids.shape[1]
        full_allow = torch.cat([
            valid.index_select(0, index)[:, None, :].bool().expand(-1, width, -1),
            torch.ones((len(rows), width, width), device=device, dtype=torch.bool).tril()], -1)
        full_mask = torch.zeros(full_allow.shape, device=device, dtype=dtype).masked_fill(
            ~full_allow, float('-inf'))[:, None]
        hidden = model.lm(input_ids=branch_ids, position_ids=branch_positions,
            attention_mask=dict(full_attention=full_mask, linear_attention=branch_valid),
            past_key_values=cache, use_cache=True).last_hidden_state.float()
        # Retain each native FP32 head shape and checkpoint temperature, with
        # one CPU transfer for the chunk instead of a sync for each junction.
        parts = [torch.nn.functional.softmax(model.head(h[row['decide']],
                    h[torch.tensor(row['opts'], device=device)]), -1)
                 for h, row in zip(hidden, rows, strict=True)]
        probabilities = iter(torch.cat(parts).cpu().split([len(part) for part in parts]))
        results = [dict(answers=to_answers([next(probabilities) for _ in rows], mapping),
                        usage=dict(input_tokens=len(encoded['ids']), output_tokens=0))
                   for (encoded, mapping), (_, _, rows) in zip(packed, splits, strict=True)]
    return results


def _native_cached_batch(self, requests, batch_size):
    import torch
    from jevlight.backends.decision_model_service import validate_answers
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    results = []
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            group = requests[start:start+batch_size]
            results.extend(_native_cached_group(self, group))
    return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]


def enable_kev_native_cache_batch(runtime):
    """Keep native hybrid prefix boundaries while batching all independent rows."""
    if hasattr(runtime, '_step_kev_cached_batch'):
        return
    if not runtime.model.hybrid:
        raise ValueError('Native cached Kev batch requires the author hybrid backbone')
    from jevlight.backends.kev_aligned_prefix import enable_aligned_prefix
    # The actual GPU screening retained native DeltaNet projection shapes;
    # complete masked GPU DeltaNet kernels remain batched across all rows.
    enable_aligned_prefix(runtime, projection_names=('.linear_attn.',), native_delta_shapes=False)
    runtime._step_kev_cached_batch = runtime.native_batch
    runtime.native_batch = MethodType(_native_cached_batch, runtime)
