"""Remove author scheduling token budgets while retaining native row forwards."""
import importlib
from types import MethodType


def _kev_batch(self, requests, batch_size):
    module=importlib.import_module(type(self.model).__module__)
    original=module.rows_per_pass
    try:
        # The outer capacity controller measures CUDA memory for all requests.
        # Keep the author's collator, masks, backbone, head and decoder intact.
        module.rows_per_pass=lambda rows, prefix_len=0, budget=None: len(rows)
        return self._native_kev_native_batch(requests,batch_size)
    finally:
        module.rows_per_pass=original


def enable_kev_full_rows(runtime):
    if not hasattr(runtime,'_native_kev_native_batch'):
        runtime._native_kev_native_batch=runtime.native_batch
        runtime.native_batch=MethodType(_kev_batch,runtime)


def enable_native_batch_conv(runtime):
    """Use the candidate kernel only inside independent-request batch calls."""
    if hasattr(runtime, '_step_conv_author_batch'):
        return
    runtime._step_conv_author_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        from jevlight.backends.causal_conv import enable_batch_conv
        original = module.causal_conv1d_fn
        try:
            enable_batch_conv(self)
            return self._step_conv_author_batch(requests, batch_size)
        finally:
            # Scalar calls and native references retain their author kernel,
            # including after a failed capacity probe.
            module.causal_conv1d_fn = original
    runtime.native_batch = MethodType(batch, runtime)


def enable_decider_full_rows(runtime):
    engine=runtime.engine
    if hasattr(engine,'_native_score_items'):
        return
    engine._native_score_items=engine._system_one_probs
    def full_rows(self,items,layout,max_fwd_tokens=65536,temperature=None):
        budget=max(max_fwd_tokens,len(items)*max(len(item['ids']) for item in items))
        return self._native_score_items(items,layout,max_fwd_tokens=budget,temperature=temperature)
    engine._system_one_probs=MethodType(full_rows,engine)


def enable_valen_full_rows(runtime):
    model = runtime.model
    if hasattr(model, '_step_native_state_batch'):
        return
    model._step_native_state_batch = model.forward_state_batch
    def full_rows(self, states, max_tokens=32768):
        # The author's compiler reports the complete shared-state row length.
        # CUDA capacity selection, rather than a 32K total-token budget, now
        # controls how many independent rows reach the original forward.
        budget = max(max_tokens, len(states) * max(state.compute_tokens for state in states)) if states else max_tokens
        return self._step_native_state_batch(states, max_tokens=budget)
    model.forward_state_batch = MethodType(full_rows, model)


def enable_valen_native_shapes(runtime):
    """Preserve each native projection's shape inside a shared GPU forward."""
    import threading
    import torch
    model = runtime.model
    if hasattr(model, '_step_shape_state_batch'):
        return
    context = threading.local()
    def projection(self, inputs):
        lengths = getattr(context, 'lengths', None)
        if not lengths or inputs.ndim != 3 or inputs.shape[0] <= 1:
            return self._step_shape_linear(inputs)
        if inputs.shape[0] != len(lengths) or inputs.shape[1] != max(lengths):
            raise ValueError('Native Valen projection rows do not match the complete compiled states')
        # Only artificial right padding is removed. The exact native input
        # matrix reaches the unchanged author projection for every state.
        rows = [torch.nn.functional.pad(self._step_shape_linear(inputs[i:i+1, :length]),
                (0, 0, 0, inputs.shape[1]-length)) for i, length in enumerate(lengths)]
        return torch.cat(rows, dim=0)
    for module in model.backbone.modules():
        if type(module) is torch.nn.Linear:
            module._step_shape_linear = module.forward
            module.forward = MethodType(projection, module)
    model._step_shape_state_batch = model.forward_state_batch
    def state_batch(self, states, max_tokens=32768):
        previous = getattr(context, 'lengths', None)
        try:
            lengths = []
            for state in states:
                if not state.questions:
                    continue
                if state.inputs is None:
                    raise ValueError('Shared-state batch requires one shared input per state')
                ids = state.inputs['input_ids']
                if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
                    raise ValueError('Native Valen state has no complete single-row input')
                lengths.append(int(ids.shape[1]))
            context.lengths = lengths
            return self._step_shape_state_batch(states, max_tokens=max_tokens)
        finally:
            context.lengths = previous
    model.forward_state_batch = MethodType(state_batch, model)


def enable_open_native_shapes(runtime):
    """Keep each request's complete candidate matrix and author FP32 head."""
    import threading
    import torch
    if hasattr(runtime, '_step_open_shape_batch'):
        return
    context = threading.local()
    def projection(self, inputs):
        layout = getattr(context, 'layout', None)
        if not layout or inputs.ndim != 3 or inputs.shape[0] <= 1:
            return self._step_open_linear(inputs)
        if inputs.shape[0] != sum(rows for rows, _ in layout) or inputs.shape[1] != max(length for _, length in layout):
            raise ValueError('Open-Jev GPU rows do not match the complete native candidate paths')
        outputs, cursor = [], 0
        for rows, length in layout:
            # Native candidate matrices are contiguous [candidates,tokens,H].
            # Copying only artificial outer padding preserves that GEMM layout.
            original = inputs[cursor:cursor+rows, :length].contiguous()
            outputs.append(torch.nn.functional.pad(self._step_open_linear(original),
                           (0, 0, 0, inputs.shape[1]-length)))
            cursor += rows
        return torch.cat(outputs, dim=0)
    for module in runtime.model.backbone.modules():
        if type(module) is torch.nn.Linear:
            module._step_open_linear = module.forward
            module.forward = MethodType(projection, module)
    head = runtime.model.head
    head._step_open_head = head.forward
    def readout(self, inputs):
        layout = getattr(context, 'layout', None)
        if not layout:
            return self._step_open_head(inputs)
        if inputs.ndim != 2 or inputs.shape[0] != sum(rows for rows, _ in layout):
            raise ValueError('Open-Jev native head lost its independent candidate rows')
        outputs, cursor = [], 0
        for rows, _ in layout:
            outputs.append(self._step_open_head(inputs[cursor:cursor+rows].contiguous()))
            cursor += rows
        return torch.cat(outputs, dim=0)
    head.forward = MethodType(readout, head)
    runtime._step_open_shape_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        previous = getattr(context, 'layout', None)
        results = []
        try:
            for start in range(0, len(requests), batch_size):
                group = requests[start:start+batch_size]
                layout = []
                for request in group:
                    lengths = self.encode(request)[0]['candidate_path_tokens']
                    if not lengths or min(lengths) <= 0:
                        raise ValueError('Open-Jev native candidate path length is unknown')
                    layout.append((len(lengths), max(lengths)))
                context.layout = layout
                results.extend(self._step_open_shape_batch(group, len(group)))
            return results
        finally:
            context.layout = previous
    runtime.native_batch = MethodType(batch, runtime)


def enable_decider_native_shapes(runtime):
    """Keep the native request collator and restricted letter-head shapes."""
    import threading
    import torch
    if hasattr(runtime, '_step_decider_shape_batch'):
        return
    context = threading.local()
    def projection(self, inputs):
        layout = getattr(context, 'layout', None)
        if not layout or inputs.ndim != 3 or inputs.shape[0] <= 1:
            return self._step_decider_linear(inputs)
        if inputs.shape[0] != sum(rows for rows, _, _ in layout) or inputs.shape[1] != max(width for _, width, _ in layout):
            raise ValueError('Decider batch rows do not match complete native request items')
        outputs, cursor = [], 0
        for rows, width, _ in layout:
            values = self._step_decider_linear(inputs[cursor:cursor+rows, :width].contiguous())
            outputs.append(torch.nn.functional.pad(values, (0, 0, 0, inputs.shape[1]-width)))
            cursor += rows
        return torch.cat(outputs, dim=0)
    for module in runtime.model.lm.model.modules():
        if type(module) is torch.nn.Linear:
            module._step_decider_linear = module.forward
            module.forward = MethodType(projection, module)
    original_logits = runtime.model.slot_logits
    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        layout = getattr(context, 'layout', None)
        if not layout:
            return original_logits(input_ids, attention_mask, slot_idx, slot_batch, nopts)
        if input_ids.shape[0] != sum(rows for rows, _, _ in layout):
            raise ValueError('Decider native slot rows lost their request mapping')
        from decider.prompt import MAX_OPTIONS
        states = self.lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = states[slot_batch, slot_idx]
        weight = self.lm.lm_head.weight[self.letters]
        outputs, cursor = [], 0
        for _, _, slots in layout:
            outputs.append(torch.nn.functional.linear(hidden[cursor:cursor+slots].contiguous(), weight).float())
            cursor += slots
        if cursor != hidden.shape[0]:
            raise ValueError('Decider native restricted head lost question slots')
        logits = torch.cat(outputs, dim=0)
        options = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
        return logits.masked_fill(options >= nopts[:, None], float('-inf'))
    runtime.model.slot_logits = MethodType(slot_logits, runtime.model)
    runtime._step_decider_shape_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        previous = getattr(context, 'layout', None)
        results = []
        try:
            for start in range(0, len(requests), batch_size):
                group = requests[start:start+batch_size]
                layout = []
                for request in group:
                    _, _, items = self.engine._system_one_items(**request, max_state_tokens=self.max_state)
                    if not items or any(not item['ids'] or not item['slots'] for item in items):
                        raise ValueError('Decider native request has no complete question rows')
                    width = ((max(len(item['ids']) for item in items)+63)//64)*64
                    layout.append((len(items), width, sum(len(item['slots']) for item in items)))
                context.layout = layout
                results.extend(self._step_decider_shape_batch(group, len(group)))
            return results
        finally:
            context.layout = previous
    runtime.native_batch = MethodType(batch, runtime)


def enable_open_co_native_flash(runtime):
    """Use native variable-length Flash for independent cooperative batches."""
    import threading
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from jevlight.backends.native_flash_batch import _metadata, _pack
    if hasattr(runtime, '_step_open_co_flash_batch'):
        return
    backbone = runtime.model.backbone
    base = backbone.get_base_model() if hasattr(backbone, 'get_base_model') else backbone
    modules = [m for m in base.modules() if type(m).__name__ == 'Qwen3_5Attention']
    if not modules or any(getattr(m, 'sliding_window', None) for m in modules):
        raise ValueError('Open-Jev cooperative Flash requires native full causal Qwen3.5 attention')
    context = threading.local()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    def before(model, args, kwargs):
        for module in modules:
            module._step_open_flash_meta = None
        if not getattr(context, 'active', False):
            return
        cache = kwargs.get('past_key_values')
        if cache is not None and cache.get_seq_length() > 0:
            return
        mask = kwargs.get('attention_mask')
        if isinstance(mask, dict):
            mask = mask.get('linear_attention')
        if not torch.is_tensor(mask) or mask.ndim != 2:
            return
        metadata = _metadata(mask)
        for module in modules:
            module._step_open_flash_meta = metadata
    base.register_forward_pre_hook(before, with_kwargs=True)
    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        metadata = getattr(module, '_step_open_flash_meta', None)
        if (not getattr(context, 'active', False) or metadata is None
                or query.shape[2] != key.shape[2] or query.dtype not in (torch.bfloat16, torch.float16)
                or dropout or kwargs.get('output_attentions')):
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        q, k, v = [_pack(tensor, metadata) for tensor in (query, key, value)]
        causal = kwargs.get('is_causal', getattr(module, 'is_causal', True))
        output = torch.ops.aten._flash_attention_forward(q, k, v, metadata['cumulative'], metadata['cumulative'],
            metadata['maximum'], metadata['maximum'], 0., causal, False, scale=scaling)[0]
        padded = output.new_zeros((metadata['batch']*metadata['length'], query.shape[1], query.shape[-1]))
        padded.index_copy_(0, metadata['indices'], output)
        return padded.view(metadata['batch'], metadata['length'], query.shape[1], query.shape[-1]), None
    ALL_ATTENTION_FUNCTIONS.register('sdpa', attention)
    runtime._step_open_co_flash_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        previous = getattr(context, 'active', False)
        try:
            # Local-mode and mixed-mode batches retain their author attention.
            # The mode flag only selects a kernel; request contents are intact.
            context.active = bool(requests) and all('cooperative_view' in r.get('state', {}) for r in requests)
            return self._step_open_co_flash_batch(requests, batch_size)
        finally:
            context.active = previous
    runtime.native_batch = MethodType(batch, runtime)


def enable_clef_text_batch_probe(runtime):
    """Expose the exact native text tower to the GPU capacity observer."""
    model = runtime.model
    language = model.language_model
    base = language.get_base_model() if hasattr(language, 'get_base_model') else language
    text = base.model
    if hasattr(text, 'language_model'):
        text = text.language_model
    existing = getattr(model, 'encoder', None)
    if existing is not None and existing is not text:
        raise ValueError('Clef batch probe would replace an existing encoder')
    # This alias is only used by the capacity observer. The unchanged author
    # forward continues to use language_model and its original joint head.
    model.encoder = text


def enable_lev_native_flash(runtime):
    """Pack native variant masks only inside independent-request GPU batches."""
    import threading
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from jevlight.backends.native_flash_batch import _metadata, _pack
    if hasattr(runtime, '_step_lev_flash_batch'):
        return
    base = runtime.engine._decoder
    modules = [m for m in base.modules() if type(m).__name__ == 'Qwen3_5Attention']
    if not modules or any(getattr(m, 'sliding_window', None) for m in modules):
        raise ValueError('Lev batch Flash requires native full causal Qwen3.5 attention')
    context = threading.local()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    def before(model, args, kwargs):
        for module in modules:
            module._step_lev_flash_meta = None
        if not getattr(context, 'active', False):
            return
        cache = kwargs.get('past_key_values')
        if cache is not None and cache.get_seq_length() > 0:
            return
        mask = kwargs.get('attention_mask')
        if isinstance(mask, dict):
            mask = mask.get('linear_attention')
        if not torch.is_tensor(mask) or mask.ndim != 2:
            return
        metadata = _metadata(mask)
        for module in modules:
            module._step_lev_flash_meta = metadata
    base.register_forward_pre_hook(before, with_kwargs=True)
    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        metadata = getattr(module, '_step_lev_flash_meta', None)
        if (not getattr(context, 'active', False) or metadata is None
                or query.shape[2] != key.shape[2] or query.dtype not in (torch.bfloat16, torch.float16)
                or dropout or kwargs.get('output_attentions')):
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        q, k, v = [_pack(tensor, metadata) for tensor in (query, key, value)]
        causal = kwargs.get('is_causal', getattr(module, 'is_causal', True))
        output = torch.ops.aten._flash_attention_forward(q, k, v, metadata['cumulative'], metadata['cumulative'],
            metadata['maximum'], metadata['maximum'], 0., causal, False, scale=scaling)[0]
        padded = output.new_zeros((metadata['batch']*metadata['length'], query.shape[1], query.shape[-1]))
        padded.index_copy_(0, metadata['indices'], output)
        return padded.view(metadata['batch'], metadata['length'], query.shape[1], query.shape[-1]), None
    ALL_ATTENTION_FUNCTIONS.register('sdpa', attention)
    runtime._step_lev_flash_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        previous = getattr(context, 'active', False)
        try:
            context.active = bool(requests)
            return self._step_lev_flash_batch(requests, batch_size)
        finally:
            context.active = previous
    runtime.native_batch = MethodType(batch, runtime)


def enable_lev_native_gate_shapes(runtime):
    """Keep native small DeltaNet gate GEMMs in one complete decoder batch."""
    import threading
    import torch
    engine = runtime.engine
    if hasattr(engine, '_step_lev_gate_forward'):
        return
    if engine.config.prefix_mode != 'single' or engine.config.compile:
        raise ValueError('Lev native gate batches require the author eager single-prefix engine')
    blocks = [m for m in engine._decoder.modules() if type(m).__name__ == 'Qwen3_5GatedDeltaNet']
    if not blocks:
        raise ValueError('Lev native gate batches require Qwen3.5 DeltaNet gates')
    context = threading.local()
    def projection(self, inputs):
        layout = getattr(context, 'layout', None)
        if layout is None or inputs.ndim != 3:
            return self._step_lev_gate_projection(inputs)
        if inputs.shape[0] != sum(rows for rows, _ in layout) or inputs.shape[1] != max(width for _, width in layout):
            raise ValueError('Lev gate batch lost complete independent variant rows')
        outputs, cursor = [], 0
        for rows, width in layout:
            values = self._step_lev_gate_projection(inputs[cursor:cursor+rows, :width].contiguous())
            outputs.append(torch.nn.functional.pad(values, (0, 0, 0, inputs.shape[1]-width)))
            cursor += rows
        return torch.cat(outputs, dim=0)
    for block in blocks:
        for name in ('in_proj_a', 'in_proj_b'):
            module = getattr(block, name)
            if type(module) is not torch.nn.Linear or module.bias is not None:
                raise ValueError('Lev native gate projection must retain the author bias-free Linear')
            module._step_lev_gate_projection = module.forward
            module.forward = MethodType(projection, module)
    engine._step_lev_gate_forward = engine._forward
    def forward(self, requests):
        previous = getattr(context, 'layout', None)
        try:
            # Prepared requests already exist here: do not tokenize a second
            # time or change the author's variant ordering and readout.
            context.layout = [(r.rows, r.width) for r in requests] if len(requests) > 1 else None
            return self._step_lev_gate_forward(requests)
        finally:
            context.layout = previous
    engine._forward = MethodType(forward, engine)


def _native_packed_projection(module, inputs, context):
    """Run each native GEMM into packed storage, then restore padded row IDs."""
    import torch
    layout = getattr(context, 'layout', None)
    if layout is None or inputs.ndim != 3:
        return module._step_native_packed_projection(inputs)
    if inputs.shape[0] != sum(rows for rows, _ in layout) or inputs.shape[1] != max(width for _, width in layout):
        raise ValueError('Native projection lost complete independent row shapes')
    flat = inputs.new_empty((sum(rows*width for rows, width in layout), module.weight.shape[0]))
    cursor = offset = 0
    for rows, width in layout:
        target = flat[offset:offset+rows*width].view(rows, width, module.weight.shape[0])
        torch.nn.functional.linear(inputs[cursor:cursor+rows, :width].contiguous(),
                                   module.weight, module.bias, out=target)
        cursor += rows
        offset += rows*width
    if inputs.is_cuda and getattr(context, 'use_scatter', False):
        from jevlight.backends.native_packed_scatter import scatter_native_rows
        return scatter_native_rows(flat, layout, inputs.shape[1], context)
    if getattr(context, 'indices', None) is None:
        positions, cursor = [], 0
        for rows, width in layout:
            for row in range(rows):
                positions.extend(range((cursor+row)*inputs.shape[1], (cursor+row)*inputs.shape[1]+width))
            cursor += rows
        context.indices = torch.tensor(positions, dtype=torch.long, device=inputs.device)
    padded = inputs.new_zeros((inputs.shape[0]*inputs.shape[1], flat.shape[-1]))
    padded.index_copy_(0, context.indices, flat)
    return padded.view(inputs.shape[0], inputs.shape[1], flat.shape[-1])


def enable_lev_native_first_layer(runtime):
    """Preserve first-layer native projections without serializing the decoder."""
    import threading
    import torch
    engine = runtime.engine
    if hasattr(engine, '_step_lev_first_forward'):
        return
    if engine.config.prefix_mode != 'single' or engine.config.compile:
        raise ValueError('Lev first-layer batches require the author eager single-prefix engine')
    context = threading.local()
    def projection(self, inputs):
        return _native_packed_projection(self, inputs, context)
    for module in engine._decoder.layers[0].modules():
        if type(module) is torch.nn.Linear:
            module._step_native_packed_projection = module.forward
            module.forward = MethodType(projection, module)
    engine._step_lev_first_forward = engine._forward
    def forward(self, requests):
        previous = (getattr(context, 'layout', None), getattr(context, 'indices', None))
        try:
            context.layout = [(r.rows, r.width) for r in requests] if len(requests) > 1 else None
            context.indices = None
            return self._step_lev_first_forward(requests)
        finally:
            context.layout, context.indices = previous
    engine._forward = MethodType(forward, engine)


def enable_lev_native_shapes(runtime, *, fused_scatter=False):
    """Retain every independent native projection shape in one decoder batch."""
    import threading
    import torch
    engine = runtime.engine
    if hasattr(engine, '_step_lev_shape_forward'):
        return
    if engine.config.prefix_mode != 'single' or engine.config.compile:
        raise ValueError('Lev native-shaped batches require the author eager single-prefix engine')
    context = threading.local()
    runtime._step_lev_projection_context = context
    context.use_scatter = fused_scatter
    def projection(self, inputs):
        return _native_packed_projection(self, inputs, context)
    for module in engine._decoder.modules():
        if type(module) is torch.nn.Linear:
            module._step_native_packed_projection = module.forward
            module.forward = MethodType(projection, module)
    engine._step_lev_shape_forward = engine._forward
    def forward(self, requests):
        previous = (getattr(context, 'layout', None), getattr(context, 'indices', None))
        try:
            context.use_scatter = fused_scatter
            context.layout = [(r.rows, r.width) for r in requests] if len(requests) > 1 else None
            context.indices = None
            return self._step_lev_shape_forward(requests)
        finally:
            context.layout, context.indices = previous
    engine._forward = MethodType(forward, engine)


def enable_clef_native_shapes(runtime, *, fused_scatter=False):
    """Retain all native text projection shapes, preserving scalar and media paths."""
    import threading
    import torch
    model = runtime.model
    if hasattr(model, '_step_clef_shape_forward'):
        return
    enable_clef_text_batch_probe(runtime)
    context = threading.local()
    context.use_scatter = fused_scatter
    def projection(self, inputs):
        return _native_packed_projection(self, inputs, context)
    for module in model.encoder.modules():
        if type(module) is torch.nn.Linear:
            module._step_native_packed_projection = module.forward
            module.forward = MethodType(projection, module)
    model._step_clef_shape_forward = model.forward
    def forward(self, batch):
        previous = (getattr(context, 'layout', None), getattr(context, 'indices', None))
        try:
            context.use_scatter = fused_scatter
            records = batch['records']
            context.layout = [(1, len(r.input_ids)) for r in records] if len(records)>1 and not batch.get('media') else None
            context.indices = None
            return self._step_clef_shape_forward(batch)
        finally:
            context.layout, context.indices = previous
    model.forward = MethodType(forward, model)


def enable_clef_native_beta_shapes(runtime):
    """Preserve native DeltaNet beta projections in the complete text batch."""
    import threading
    import torch
    model = runtime.model
    if hasattr(model, '_step_clef_beta_forward'):
        return
    enable_clef_text_batch_probe(runtime)
    blocks = [m for m in model.encoder.modules() if type(m).__name__ == 'Qwen3_5GatedDeltaNet']
    if not blocks:
        raise ValueError('Clef native beta batches require Qwen3.5 DeltaNet')
    context = threading.local()
    def projection(self, inputs):
        return _native_packed_projection(self, inputs, context)
    for block in blocks:
        module = block.in_proj_b
        if type(module) is not torch.nn.Linear or module.bias is not None:
            raise ValueError('Clef native beta projection must retain the author bias-free Linear')
        module._step_native_packed_projection = module.forward
        module.forward = MethodType(projection, module)
    model._step_clef_beta_forward = model.forward
    def forward(self, batch):
        previous = (getattr(context, 'layout', None), getattr(context, 'indices', None))
        try:
            records = batch['records']
            context.layout = [(1, len(r.input_ids)) for r in records] if len(records) > 1 and not batch.get('media') else None
            context.indices = None
            return self._step_clef_beta_forward(batch)
        finally:
            context.layout, context.indices = previous
    model.forward = MethodType(forward, model)


def enable_clef_native_flash(runtime):
    """Pack complete text masks only during independent-request Clef batches."""
    import threading
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from jevlight.backends.native_flash_batch import _metadata, _pack
    if hasattr(runtime, '_step_clef_flash_batch'):
        return
    enable_clef_text_batch_probe(runtime)
    base = runtime.model.encoder
    modules = [m for m in base.modules() if type(m).__name__ == 'Qwen3_5Attention']
    if not modules or any(getattr(m, 'sliding_window', None) for m in modules):
        raise ValueError('Clef batch Flash requires native full causal Qwen3.5 attention')
    context = threading.local()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    def before(model, args, kwargs):
        for module in modules:
            module._step_clef_flash_meta = None
        if not getattr(context, 'active', False):
            return
        cache = kwargs.get('past_key_values')
        if cache is not None and cache.get_seq_length() > 0:
            return
        mask = kwargs.get('attention_mask')
        if isinstance(mask, dict):
            mask = mask.get('linear_attention')
        if not torch.is_tensor(mask) or mask.ndim != 2:
            return
        metadata = _metadata(mask)
        for module in modules:
            module._step_clef_flash_meta = metadata
    base.register_forward_pre_hook(before, with_kwargs=True)
    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        metadata = getattr(module, '_step_clef_flash_meta', None)
        if (not getattr(context, 'active', False) or metadata is None
                or query.shape[2] != key.shape[2] or query.dtype not in (torch.bfloat16, torch.float16)
                or dropout or kwargs.get('output_attentions')):
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        q, k, v = [_pack(tensor, metadata) for tensor in (query, key, value)]
        causal = kwargs.get('is_causal', getattr(module, 'is_causal', True))
        output = torch.ops.aten._flash_attention_forward(q, k, v, metadata['cumulative'], metadata['cumulative'],
            metadata['maximum'], metadata['maximum'], 0., causal, False, scale=scaling)[0]
        padded = output.new_zeros((metadata['batch']*metadata['length'], query.shape[1], query.shape[-1]))
        padded.index_copy_(0, metadata['indices'], output)
        return padded.view(metadata['batch'], metadata['length'], query.shape[1], query.shape[-1]), None
    ALL_ATTENTION_FUNCTIONS.register('sdpa', attention)
    runtime._step_clef_flash_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        previous = getattr(context, 'active', False)
        try:
            context.active = bool(requests)
            return self._step_clef_flash_batch(requests, batch_size)
        finally:
            context.active = previous
    runtime.native_batch = MethodType(batch, runtime)
