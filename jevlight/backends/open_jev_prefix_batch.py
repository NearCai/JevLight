"""Batch independent Open-Jev prefixes with request-local native cache forks."""
import copy
from types import MethodType


def clone_batch_cache(cache):
    """Preserve author recurrent-state dtype and independent branch storage."""
    from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer
    from jev.prefix_cache import RecurrentDtypeLayer
    if type(cache) is not DynamicCache or cache.offloading:
        raise TypeError('Native prefix batch requires a resident DynamicCache')
    branch = copy.copy(cache)
    branch.layers = []
    for layer in cache.layers:
        if type(layer) is DynamicLayer:
            fields = ('keys', 'values')
        elif type(layer) in (LinearAttentionLayer, RecurrentDtypeLayer):
            fields = ('conv_states', 'recurrent_states')
        else:
            raise TypeError(f'Unreviewed native cache layer: {type(layer).__name__}')
        cloned = copy.copy(layer)
        for name in fields:
            value = getattr(layer, name)
            if value is not None:
                setattr(cloned, name, value.clone())
        branch.layers.append(cloned)
    return branch


def compile_paths(runtime, requests):
    from jev.api import compile_request, candidate_prompts
    from jev.prefix_cache import common_prefix_length
    groups = [compile_request(**request) for request in requests]
    records = [record for group in groups for record in group]
    rows = []
    for record in records:
        if record['kind'] != 'choice':
            raise ValueError('The qualified native prefix path requires phase Choice')
        rendered = [runtime.model.tokenizer.apply_chat_template(
            [dict(role='user', content=text)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
            for text in candidate_prompts(record)]
        encoded = runtime.model.tokenizer(rendered, padding=False, truncation=False)
        paths = encoded['input_ids']
        if any(not path or len(path) > runtime.model.max_length for path in paths):
            raise ValueError('A complete native candidate exceeds the verified window')
        if any(not all(mask) for mask in encoded.get('attention_mask', [])):
            raise ValueError('Native unpadded paths contain masked real tokens')
        prefix = common_prefix_length(paths)
        if prefix < 1 or len(paths) != 4:
            raise ValueError('The qualified prefix batch requires four complete phase paths')
        rows.append(dict(paths=paths, prefix=prefix))
    if not rows:
        raise ValueError('No native phase records to batch')
    return groups, rows


def score_prefix_rows(runtime, rows):
    import torch
    from jev.prefix_cache import new_cache
    model = runtime.model
    core = model.backbone.get_base_model() if hasattr(model.backbone, 'get_base_model') else model.backbone
    if core.config.model_type != 'qwen3_5_text':
        raise ValueError('Native prefix batch needs the pinned Qwen3.5/3.8 text stack')
    device = next(model.backbone.parameters()).device
    lengths = torch.tensor([row['prefix'] for row in rows], device=device)
    width = int(lengths.max())
    dimensions = [core.config.hidden_size]
    for layer in model.backbone.modules():
        if type(layer).__name__ == 'Qwen3_5GatedDeltaNet':
            dimensions.extend([layer.key_dim, layer.value_dim])
    if len(rows)*width*max(dimensions) >= 2**31:
        raise RuntimeError('canUse32BitIndexMath(input) && canUse32BitIndexMath(output): native prefix normalization preflight')
    pad = model.tokenizer.pad_token_id or 0
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        left = width-row['prefix']
        ids[index, left:] = torch.tensor(row['paths'][0][:row['prefix']], device=device)
        valid[index, left:] = 1
        positions[index, left:] = torch.arange(row['prefix'], device=device)
    # Left padding precedes the initially zero recurrent state. Real prefix
    # tails remain at the end, and each row keeps its native rotary positions.
    cache = new_cache(core.config)
    model.backbone(input_ids=ids, attention_mask=valid, position_ids=positions,
                   past_key_values=cache, use_cache=True, return_dict=True)
    scores = []
    processed = len(rows)*width
    for candidate in range(4):
        branch = clone_batch_cache(cache)
        tails = [row['paths'][candidate][row['prefix']:] for row in rows]
        sizes = torch.tensor([len(tail) for tail in tails], device=device)
        tail_width = int(sizes.max())
        tail_ids = torch.full((len(rows), tail_width), pad, dtype=torch.long, device=device)
        tail_valid = torch.zeros_like(tail_ids)
        for index, tail in enumerate(tails):
            tail_ids[index, :len(tail)] = torch.tensor(tail, device=device)
            tail_valid[index, :len(tail)] = 1
        tail_positions = lengths[:, None]+torch.arange(tail_width, device=device)[None]
        result = model.backbone(input_ids=tail_ids, attention_mask=torch.cat([valid, tail_valid], dim=1),
            position_ids=tail_positions, past_key_values=branch, use_cache=True, return_dict=True)
        hidden = result.last_hidden_state[torch.arange(len(rows), device=device), sizes-1].float()
        scores.append(model.head(hidden).squeeze(-1))
        processed += len(rows)*tail_width
        del result, branch, hidden
    values = torch.stack(scores, dim=1)
    if not torch.isfinite(values).all():
        raise ValueError('Native prefix head returned nonfinite scores')
    runtime.last_prefix_batch = dict(request_records=len(rows), gpu_prefix_rows=len(rows),
        candidate_branches=4, processed_input_tokens=processed,
        logical_input_tokens=sum(len(path) for row in rows for path in row['paths']),
        prefix_lengths=[row['prefix'] for row in rows], cache_scope='current GPU chunk only',
        recurrent_state_dtype='author-preserved kernel output', complete_paths=True)
    model.last_input_tokens = runtime.last_prefix_batch['logical_input_tokens']
    return values.cpu().tolist()


def native_prefix_batch(self, requests, batch_size):
    import torch
    from jev.api import format_response
    from jev.metrics import softmax
    from jevlight.backends.decision_model_service import validate_answers
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Native prefix batch size must be positive')
    if len(requests) <= 1 or batch_size == 1:
        return self._prefix_original_batch(requests, batch_size)
    results = []
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            selected = requests[start:start+batch_size]
            groups, rows = compile_paths(self, selected)
            values = score_prefix_rows(self, rows)
            cursor = 0
            for group in groups:
                probabilities = [softmax(row, temperature=self.temperature)
                                 for row in values[cursor:cursor+len(group)]]
                results.append(format_response(group, probabilities))
                cursor += len(group)
            if cursor != len(rows):
                raise ValueError('Native prefix question mapping differs')
    return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]


def enable_open_prefix_batch(runtime, *, merged=False, fused=False, package=None):
    if hasattr(runtime, '_prefix_original_batch'):
        return
    if merged:
        # The same author-supported PEFT merge already qualified for the
        # independent export. Frozen source adapters and FP32 readout remain.
        runtime.model.backbone = runtime.model.backbone.merge_and_unload(safe_merge=True)
        runtime.model._validate_loaded_parameters()
        runtime.model.eval()
    if fused:
        from jevlight.backends.open_jev_fla_batch import load_batch_fla
        chunk = load_batch_fla(package)
        layers = [layer for layer in runtime.model.backbone.modules()
                  if type(layer).__name__ == 'Qwen3_5GatedDeltaNet']
        if not layers:
            raise ValueError('The native fused prefix has no qualified DeltaNet layers')
        for layer in layers:
            layer.chunk_gated_delta_rule = chunk
    runtime._prefix_original_batch = runtime.native_batch
    runtime.native_batch = MethodType(native_prefix_batch, runtime)
