"""Verify independent request/head mappings without loading weights."""

from contextlib import nullcontext

from types import ModuleType, SimpleNamespace

import sys

import pytest

from jevlight.backends.decision_model_service import OpenJevNative, ClefNative

from tests.test_experiment_preflight import request, result

def test_pooling_capacity_failure_aborts_pending_engine_requests(monkeypatch):
    from jevlight.backends.vllm_decision_runtime import OpenJevPoolingNative
    engine_module = ModuleType('vllm')
    engine_module.PoolingParams = lambda **options: options
    author_api = ModuleType('jev.api');author_api.candidate_prompts = lambda record: []
    author_api.format_response = lambda *args: None
    author_metrics = ModuleType('jev.metrics');author_metrics.softmax = lambda scores: scores
    for name, module in [('vllm', engine_module), ('jev.api', author_api),
                         ('jev.metrics', author_metrics), ('torch', ModuleType('torch'))]:
        monkeypatch.setitem(sys.modules, name, module)
    pending = {'running-native-path': object(), 'waiting-native-path': object()}
    aborted = []
    def abort(ids, internal=False):
        assert internal is True
        aborted.extend(ids)
        for key in ids: pending.pop(key)
    core = SimpleNamespace(output_processor=SimpleNamespace(request_states=pending),
                           abort_request=abort)
    failure = RuntimeError('GPU capacity exceeded')
    def encode(*args, **kwargs):
        raise failure
    runtime = OpenJevPoolingNative.__new__(OpenJevPoolingNative, {})
    runtime.engine = SimpleNamespace(encode=encode, llm_engine=core)
    runtime.lora = object()
    runtime._compile = lambda request: ([], [[11, 22]])
    with pytest.raises(RuntimeError) as caught:
        runtime.native_batch([{}], 1)
    assert caught.value is failure
    assert aborted == ['running-native-path', 'waiting-native-path']
    assert not pending

@pytest.mark.parametrize('merged', [False, True])
def test_open_pooling_dispatch_initializes_only_selected_native_runtime(monkeypatch, merged):
    from jevlight.backends.vllm_decision_runtime import OpenJevPoolingNative
    calls = []
    monkeypatch.setattr(OpenJevPoolingNative, '__init__', lambda self, config: calls.append('fp32_adapter'))
    class Merged(OpenJevPoolingNative):
        def __init__(self, config): calls.append('author_merged')
    module = ModuleType('jevlight.backends.open_jev_merged_vllm_runtime')
    module.MergedOpenJevPoolingNative = Merged
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = OpenJevPoolingNative({'engine_adapter': 'author_merged'} if merged else {})
    assert type(runtime) is (Merged if merged else OpenJevPoolingNative)
    assert calls == ['author_merged' if merged else 'fp32_adapter']

@pytest.mark.parametrize('full', [False, True])
def test_kev_full_release_loads_its_own_tokenizer_and_adapter_keeps_pinned_base(monkeypatch, full):
    from jevlight.backends import decision_model_service as service
    calls = []
    checkpoint = SimpleNamespace(path='/frozen/kev/weights', full=full,
        meta=SimpleNamespace(base='author/pinned-base', base_revision='frozen-revision'))
    def load(device, options):
        calls.append((device, options))
        return 'author-tokenizer', 'author-model'
    checkpoint.load=load
    author=ModuleType('kev.checkpoint')
    author.Checkpoint=lambda path:checkpoint
    author.LoadOptions=lambda **options:options
    monkeypatch.setitem(sys.modules, 'kev.checkpoint', author)
    torch=ModuleType('torch');torch.bfloat16=object()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(service, 'add_author_source', lambda key:None)
    def local_base(config, base, revision):
        assert not full, 'A full release must not require a downloaded base'
        assert (base, revision)==('author/pinned-base','frozen-revision')
        return '/frozen/base'
    monkeypatch.setattr(service, 'local_base', local_base)
    runtime=service.KevNative(dict(weights_path='.local/weights', max_length=8192, max_state_tokens=4096))
    assert checkpoint.meta.base==('/frozen/kev/weights' if full else '/frozen/base')
    assert checkpoint.meta.base_revision is None
    assert (runtime.tokenizer, runtime.model)==('author-tokenizer','author-model')
    assert calls==[('cuda',dict(dtype=torch.bfloat16, merge=True, backend='torch', fused=False, cuda_graphs=False))]

@pytest.mark.parametrize('fails', [False, True])
def test_lev_native_gate_shapes_keep_variants_and_restore_after_capacity_failure(monkeypatch, fails):
    from jevlight.backends.native_row_batch import enable_lev_native_gate_shapes
    class Tensor:
        ndim=3
        def __init__(self, rows):self.rows,self.shape=rows,(len(rows),len(rows[0]),1)
        def __getitem__(self,index):return Tensor([row[index[1]] for row in self.rows[index[0]]])
        def contiguous(self):return self
    calls=[]
    class Linear:
        bias=None
        def forward(self,inputs):calls.append(inputs.rows);return inputs
    torch=ModuleType('torch')
    torch.nn=SimpleNamespace(Linear=Linear,functional=SimpleNamespace(pad=lambda inputs,sizes:
        Tensor([row+[0]*sizes[3] for row in inputs.rows])))
    torch.cat=lambda tensors,dim:Tensor([row for tensor in tensors for row in tensor.rows])
    monkeypatch.setitem(sys.modules,'torch',torch)
    block=type('Qwen3_5GatedDeltaNet',(),{})()
    block.in_proj_a,block.in_proj_b=Linear(),Linear()
    unrelated=Linear()
    rows=[[11,12,13,0,0],[21,22,23,0,0],[31,32,33,34,35]]
    def forward(prepared):
        assert block.in_proj_a.forward(Tensor(rows)).rows==rows
        if fails and len(prepared)>1:raise RuntimeError('native capacity failure')
        return prepared
    engine=SimpleNamespace(config=SimpleNamespace(prefix_mode='single',compile=False),
        _decoder=SimpleNamespace(modules=lambda:[block,unrelated]),_forward=forward)
    runtime=SimpleNamespace(engine=engine)
    prepared=[SimpleNamespace(rows=2,width=3),SimpleNamespace(rows=1,width=5)]
    enable_lev_native_gate_shapes(runtime);wrapped=engine._forward
    enable_lev_native_gate_shapes(runtime);assert engine._forward is wrapped
    if fails:
        with pytest.raises(RuntimeError,match='native capacity failure'):engine._forward(prepared)
    else:assert engine._forward(prepared)==prepared
    assert calls==[[[11,12,13],[21,22,23]],[[31,32,33,34,35]]]
    # Single requests, even with several variants, retain their author GEMM.
    engine._forward(prepared[:1]);assert calls[-1]==rows
    unrelated.forward(Tensor(rows));assert calls[-1]==rows

@pytest.mark.parametrize('all_layers', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_lev_native_shapes_keep_whole_forward_and_restore_scalar_scope(monkeypatch, fails, all_layers):
    from jevlight.backends import native_row_batch as batch
    scopes=[]
    class Linear:
        def forward(self,inputs):return inputs
    torch=ModuleType('torch');torch.nn=SimpleNamespace(Linear=Linear)
    monkeypatch.setitem(sys.modules,'torch',torch)
    def projection(module,inputs,context):
        scopes.append((context.layout,context.indices))
        if context.layout is not None:context.indices='temporary GPU indices'
        return module._step_native_packed_projection(inputs)
    monkeypatch.setattr(batch,'_native_packed_projection',projection)
    first,later=Linear(),Linear()
    layers=[SimpleNamespace(modules=lambda:[first]),SimpleNamespace(modules=lambda:[later])]
    calls=[]
    def forward(requests):
        calls.append(requests);first.forward('complete decoder input');later.forward('complete decoder input')
        if fails and len(requests)>1:raise RuntimeError('capacity failure')
        return requests
    engine=SimpleNamespace(config=SimpleNamespace(prefix_mode='single',compile=False),
        _decoder=SimpleNamespace(layers=layers,modules=lambda:[first,later]),_forward=forward)
    runtime=SimpleNamespace(engine=engine)
    requests=[SimpleNamespace(rows=2,width=3),SimpleNamespace(rows=1,width=5)]
    enable=batch.enable_lev_native_shapes if all_layers else batch.enable_lev_native_first_layer
    enable(runtime);wrapped=engine._forward
    enable(runtime);assert engine._forward is wrapped
    if fails:
        with pytest.raises(RuntimeError,match='capacity failure'):engine._forward(requests)
    else:assert engine._forward(requests)==requests
    expected=[([(2,3),(1,5)],None)]
    if all_layers:expected.append(([(2,3),(1,5)],'temporary GPU indices'))
    assert calls==[requests] and scopes==expected
    assert hasattr(later,'_step_native_packed_projection') is all_layers
    engine._forward(requests[:1]);assert scopes[-1]==(None,None)

@pytest.mark.parametrize('all_layers', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_clef_native_shapes_keep_author_head_media_and_scalar_scope(monkeypatch, fails, all_layers):
    from jevlight.backends import native_row_batch as batch
    scopes=[]
    class Linear:
        bias=None
        def forward(self,inputs):return inputs
    torch=ModuleType('torch');torch.nn=SimpleNamespace(Linear=Linear)
    monkeypatch.setitem(sys.modules,'torch',torch)
    def projection(module,inputs,context):
        scopes.append((context.layout,context.indices))
        if context.layout is not None:context.indices='temporary GPU indices'
        return module._step_native_packed_projection(inputs)
    monkeypatch.setattr(batch,'_native_packed_projection',projection)
    block=type('Qwen3_5GatedDeltaNet',(),{})();block.in_proj_a,block.in_proj_b=Linear(),Linear()
    text=SimpleNamespace(modules=lambda:[block,block.in_proj_a,block.in_proj_b]);head=object()
    def forward(value):
        block.in_proj_b.forward('complete text tower input')
        if fails and len(value['records'])>1 and not value.get('media'):raise RuntimeError('capacity failure')
        return head
    model=SimpleNamespace(language_model=SimpleNamespace(model=text),head=head,forward=forward)
    runtime=SimpleNamespace(model=model)
    enable=batch.enable_clef_native_shapes if all_layers else batch.enable_clef_native_beta_shapes
    enable(runtime);wrapped=model.forward
    enable(runtime);assert model.forward is wrapped
    value=dict(records=[SimpleNamespace(input_ids=[1,2,3]),SimpleNamespace(input_ids=[4,5])])
    if fails:
        with pytest.raises(RuntimeError,match='capacity failure'):model.forward(value)
    else:assert model.forward(value) is head
    assert scopes==[([(1,3),(1,2)],None)]
    assert model.head is head and hasattr(block.in_proj_a,'_step_native_packed_projection') is all_layers
    assert model.forward(dict(records=value['records'][:1])) is head
    assert scopes[-1]==(None,None)
    assert model.forward(dict(value,media={'image':'author media input'})) is head
    assert scopes[-1]==(None,None)

@pytest.mark.parametrize('fails', [False, True])
def test_batch_convolution_restores_author_kernel_and_does_not_wrap_twice(monkeypatch, fails):
    from jevlight.backends.native_row_batch import enable_native_batch_conv
    author = ModuleType('transformers.models.qwen3_5.modeling_qwen3_5')
    original, candidate = object(), object()
    author.causal_conv1d_fn = original
    package = ModuleType('transformers.models.qwen3_5')
    package.modeling_qwen3_5 = author
    monkeypatch.setitem(sys.modules, 'transformers.models.qwen3_5', package)
    kernels = ModuleType('jevlight.backends.causal_conv')
    kernels.enable_batch_conv = lambda runtime: setattr(author, 'causal_conv1d_fn', candidate)
    monkeypatch.setitem(sys.modules, 'jevlight.backends.causal_conv', kernels)
    calls = []
    def batch(rows, size):
        assert author.causal_conv1d_fn is candidate
        calls.append((rows, size))
        if fails:
            raise RuntimeError('native capacity failure')
        return rows
    runtime = SimpleNamespace(native_batch=batch)
    enable_native_batch_conv(runtime)
    wrapped = runtime.native_batch
    enable_native_batch_conv(runtime)
    assert runtime.native_batch is wrapped and author.causal_conv1d_fn is original
    if fails:
        with pytest.raises(RuntimeError, match='native capacity failure'):
            runtime.native_batch([1, 2, 3], 196)
    else:
        assert runtime.native_batch([1, 2, 3], 196) == [1, 2, 3]
    assert calls == [([1, 2, 3], 196)]
    assert author.causal_conv1d_fn is original

def test_phase_acceptance_does_not_check_probability_or_confidence_values():
    from scripts.experiments.batch_acceptance import compare_phases
    req = dict(questions={'phase': dict(type='choice', criteria={'A': '', 'B': ''})})
    before = dict(answers={'phase': dict(type='choice', choice='A', probabilities={'A': .8, 'B': .2})})
    after = dict(answers={'phase': dict(type='choice', choice='A', probabilities={'A': 7, 'B': 4}, confidence=-2)})
    compare_phases([req], [before], [after])
    with pytest.raises(ValueError, match='phase disagreement'):
        compare_phases([req], [before], [dict(answers={'phase': dict(type='choice', choice='B')})])
    with pytest.raises(ValueError, match='answer IDs'):
        compare_phases([req], [before], [dict(answers={'other': dict(type='choice', choice='A')})])

def test_relaxed_phase_policy_still_rejects_lost_mapping_and_invalid_candidates():
    from scripts.experiments.batch_acceptance import compare_phases
    req = dict(questions={'phase': dict(type='choice', criteria={'A': '', 'B': ''})})
    before = dict(answers={'phase': dict(type='choice', choice='A')})
    changed = dict(answers={'phase': dict(type='choice', choice='B')})
    compare_phases([req], [before], [changed], require_equal=False)
    for invalid in (dict(answers={}),
                    dict(answers={'phase': dict(type='choice', choice='C')}),
                    dict(answers={'phase': dict(type='noul', choice='B')})):
        with pytest.raises(ValueError):
            compare_phases([req], [before], [invalid], require_equal=False)

@pytest.mark.parametrize('fails', [False, True])
def test_valen_native_projection_shapes_preserve_full_rows_and_reset_after_failure(monkeypatch, fails):
    from jevlight.backends.native_row_batch import enable_valen_native_shapes
    class Tensor:
        ndim = 3
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]), 1)
        def __getitem__(self, index):
            row, tokens = index
            return Tensor([values[tokens] for values in self.rows[row]])
    calls = []
    class Linear:
        def forward(self, inputs):
            calls.append(inputs.rows)
            return inputs
    def pad(inputs, sizes):
        return Tensor([values + [0]*sizes[3] for values in inputs.rows])
    torch = ModuleType('torch')
    torch.nn = SimpleNamespace(Linear=Linear, functional=SimpleNamespace(pad=pad))
    torch.cat = lambda rows, dim: Tensor([values for row in rows for values in row.rows])
    monkeypatch.setitem(sys.modules, 'torch', torch)
    linear = Linear()
    def forward(states, max_tokens):
        result = linear.forward(Tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]]))
        if fails:
            raise RuntimeError('actual capacity failure')
        return result
    model = SimpleNamespace(backbone=SimpleNamespace(modules=lambda: [linear]), forward_state_batch=forward)
    runtime = SimpleNamespace(model=model)
    states = [SimpleNamespace(questions=[1], inputs={'input_ids': SimpleNamespace(ndim=2, shape=(1, length))})
              for length in (3, 5)]
    enable_valen_native_shapes(runtime)
    wrapped = model.forward_state_batch
    enable_valen_native_shapes(runtime)
    assert model.forward_state_batch is wrapped
    if fails:
        with pytest.raises(RuntimeError, match='actual capacity failure'):
            model.forward_state_batch(states, max_tokens=196)
    else:
        assert model.forward_state_batch(states, max_tokens=196).rows == [[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]]
    assert calls == [[[11, 12, 13]], [[21, 22, 23, 24, 25]]]
    # A later native call retains its original whole matrix, even after OOM.
    linear.forward(Tensor([[31, 32, 33, 34], [41, 42, 43, 44]]))
    assert calls[-1] == [[31, 32, 33, 34], [41, 42, 43, 44]]

@pytest.mark.parametrize('fails', [False, True])
def test_open_native_shapes_keep_variable_candidate_groups_and_author_head(monkeypatch, fails):
    from jevlight.backends.native_row_batch import enable_open_native_shapes
    class Tensor:
        def __init__(self, rows, ndim=3):
            self.rows, self.ndim = rows, ndim
            self.shape = (len(rows), len(rows[0]), 1) if ndim == 3 else (len(rows), 1)
        def __getitem__(self, index):
            if isinstance(index, tuple):
                return Tensor([values[index[1]] for values in self.rows[index[0]]], self.ndim)
            return Tensor(self.rows[index], self.ndim)
        def contiguous(self): return self
    projection_calls, head_calls = [], []
    class Linear:
        def forward(self, inputs):
            projection_calls.append(inputs.rows)
            return inputs
    class Head:
        def forward(self, inputs):
            head_calls.append(inputs.rows)
            return inputs
    torch = ModuleType('torch')
    torch.nn = SimpleNamespace(Linear=Linear, functional=SimpleNamespace(pad=lambda inputs, sizes:
        Tensor([values+[0]*sizes[3] for values in inputs.rows])))
    torch.cat = lambda rows, dim: Tensor([values for row in rows for values in row.rows], rows[0].ndim)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    linear, head = Linear(), Head()
    rows = [[11,12,13,0,0],[21,22,0,0,0],[31,32,33,34,35],[41,42,43,0,0],[51,52,53,54,0]]
    def author(requests, size):
        assert linear.forward(Tensor(rows)).rows == rows
        assert head.forward(Tensor([[1],[2],[3],[4],[5]], 2)).rows == [[1],[2],[3],[4],[5]]
        if fails: raise RuntimeError('native GPU capacity failure')
        return requests
    runtime = SimpleNamespace(model=SimpleNamespace(backbone=SimpleNamespace(modules=lambda: [linear]), head=head),
        native_batch=author, encode=lambda req: [dict(candidate_path_tokens=req['lengths'])])
    requests = [dict(id='first', lengths=[3,2]), dict(id='second', lengths=[5,3,4])]
    enable_open_native_shapes(runtime)
    wrapped = runtime.native_batch
    enable_open_native_shapes(runtime)
    assert runtime.native_batch is wrapped
    if fails:
        with pytest.raises(RuntimeError, match='native GPU capacity failure'):
            runtime.native_batch(requests, 196)
    else:
        assert runtime.native_batch(requests, 196) == requests
    assert projection_calls == [[[11,12,13],[21,22,0]], rows[2:]]
    assert head_calls == [[[1],[2]], [[3],[4],[5]]]
    linear.forward(Tensor(rows)); head.forward(Tensor([[1],[2],[3],[4],[5]], 2))
    assert projection_calls[-1] == rows and head_calls[-1] == [[1],[2],[3],[4],[5]]

@pytest.mark.parametrize('fails', [False, True])
def test_decider_shapes_keep_native_64_padding_multiple_questions_and_restore(monkeypatch, fails):
    from jevlight.backends.native_row_batch import enable_decider_native_shapes
    class Tensor:
        ndim = 3
        def __init__(self, rows): self.rows, self.shape = rows, (len(rows), len(rows[0]), 1)
        def __getitem__(self, index): return Tensor([row[index[1]] for row in self.rows[index[0]]])
        def contiguous(self): return self
    calls = []
    class Linear:
        def forward(self, inputs): calls.append(inputs.rows); return inputs
    torch = ModuleType('torch')
    torch.nn = SimpleNamespace(Linear=Linear, functional=SimpleNamespace(pad=lambda inputs, sizes:
        Tensor([row+[0]*sizes[3] for row in inputs.rows])))
    torch.cat = lambda tensors, dim: Tensor([row for tensor in tensors for row in tensor.rows])
    monkeypatch.setitem(sys.modules, 'torch', torch)
    linear=Linear(); rows=[[11]*64+[0]*64, [22]*64+[0]*64, [33]*128]
    def author(requests, size):
        assert linear.forward(Tensor(rows)).rows == rows
        if fails: raise RuntimeError('native capacity failure')
        return requests
    def items(state, max_state_tokens):
        return {}, {}, [dict(ids=[1]*length, slots=[0]) for length in state]
    inner=SimpleNamespace(modules=lambda:[linear])
    model=SimpleNamespace(lm=SimpleNamespace(model=inner), slot_logits=lambda *args:'native letter head')
    runtime=SimpleNamespace(model=model, engine=SimpleNamespace(_system_one_items=items),
        max_state=262144, native_batch=author)
    enable_decider_native_shapes(runtime); wrapped=runtime.native_batch
    enable_decider_native_shapes(runtime); assert runtime.native_batch is wrapped
    requests=[dict(state=[49,63]), dict(state=[65])]
    if fails:
        with pytest.raises(RuntimeError,match='native capacity failure'):runtime.native_batch(requests,196)
    else:assert runtime.native_batch(requests,196)==requests
    assert calls == [[[11]*64,[22]*64],[[33]*128]]
    linear.forward(Tensor(rows)); assert calls[-1]==rows
    assert model.slot_logits(None,None,None,None,None)=='native letter head'

@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('family', ['open-jev', 'lev'])
def test_native_flash_only_prepares_selected_batches_and_restores_native_calls(monkeypatch, fails, family):
    from jevlight.backends.native_row_batch import enable_open_co_native_flash,enable_lev_native_flash
    enable = enable_open_co_native_flash if family == 'open-jev' else enable_lev_native_flash
    attribute = '_step_open_flash_meta' if family == 'open-jev' else '_step_lev_flash_meta'
    class Registry(dict):
        def register(self, key, value): self[key]=value
    registry=Registry(sdpa=lambda *args, **kwargs:'author attention')
    transformers=ModuleType('transformers.modeling_utils')
    transformers.ALL_ATTENTION_FUNCTIONS=registry
    monkeypatch.setitem(sys.modules,'transformers.modeling_utils',transformers)
    class Mask: ndim=2
    torch=ModuleType('torch');torch.is_tensor=lambda value:isinstance(value,Mask)
    monkeypatch.setitem(sys.modules,'torch',torch)
    calls=[]
    metadata=ModuleType('jevlight.backends.native_flash_batch')
    def prepare(mask): calls.append(mask);return {'complete_native_mask':mask}
    metadata._metadata=prepare;metadata._pack=lambda value,meta:value
    monkeypatch.setitem(sys.modules,'jevlight.backends.native_flash_batch',metadata)
    class Qwen3_5Attention: pass
    module=Qwen3_5Attention();hooks=[]
    base=SimpleNamespace(modules=lambda:[module],register_forward_pre_hook=lambda fn,with_kwargs:hooks.append(fn))
    def author(requests,size):
        hooks[0](base,(),{'attention_mask':Mask()})
        if family == 'lev' or all('cooperative_view' in r['state'] for r in requests):
            assert getattr(module,attribute) is not None
        else:assert getattr(module,attribute) is None
        if fails:raise RuntimeError('actual CUDA failure')
        return requests
    runtime=SimpleNamespace(model=SimpleNamespace(backbone=base),engine=SimpleNamespace(_decoder=base),native_batch=author)
    enable(runtime);wrapped=runtime.native_batch
    enable(runtime);assert runtime.native_batch is wrapped and len(hooks)==1
    co=[dict(state={'cooperative_view':{}})]*2
    if fails:
        with pytest.raises(RuntimeError,match='actual CUDA failure'):runtime.native_batch(co,196)
    else:assert runtime.native_batch(co,196)==co
    assert len(calls)==1
    # A native request can have four candidate GPU rows; it must still use
    # the author attention and avoid the candidate's CPU mask preparation.
    hooks[0](base,(),{'attention_mask':Mask()})
    assert getattr(module,attribute) is None and len(calls)==1
    assert registry['sdpa'](module,None,None,None,None)=='author attention'
    local=[dict(state={})]*2
    if fails:
        with pytest.raises(RuntimeError,match='actual CUDA failure'):runtime.native_batch(local,196)
    else:assert runtime.native_batch(local,196)==local
    assert len(calls)==(2 if family == 'lev' else 1)
    mixed=[co[0],local[0]]
    if fails:
        with pytest.raises(RuntimeError,match='actual CUDA failure'):runtime.native_batch(mixed,196)
    else:assert runtime.native_batch(mixed,196)==mixed
    assert len(calls)==(3 if family == 'lev' else 1)

def test_clef_batch_probe_observes_exact_author_text_tower_without_replacing_head():
    from jevlight.backends.native_row_batch import enable_clef_text_batch_probe
    tower,head=object(),object()
    base=SimpleNamespace(model=SimpleNamespace(language_model=tower))
    language=SimpleNamespace(get_base_model=lambda:base)
    model=SimpleNamespace(language_model=language,head=head)
    runtime=SimpleNamespace(model=model)
    enable_clef_text_batch_probe(runtime);enable_clef_text_batch_probe(runtime)
    assert model.encoder is tower and model.language_model is language and model.head is head
    model.encoder=object()
    with pytest.raises(ValueError,match='existing encoder'):enable_clef_text_batch_probe(runtime)

def test_gliner_batch_restores_repeated_ids_and_independent_schemas(monkeypatch):
    from jevlight.backends.decision_model_service import GlinerNative
    from jevlight.backends.gliner_batch import enable_gliner_batch
    module = ModuleType('gliner2.classification')
    module.ClassificationConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'gliner2.classification', module)
    runtime = GlinerNative.__new__(GlinerNative)
    inputs = [request(i) for i in range(3)]
    inputs = [dict(state=r['state'], questions={
        'phase': r['questions']['phase_'+str(i)],
        'congestion': r['questions']['congested_'+str(i)]}) for i, r in enumerate(inputs)]
    inputs[1]['questions']['phase']['criteria'] = {'B': 'second', 'A': 'first'}
    runtime.rows = lambda req: ((qid, str(req['state']['id']), q) for qid, q in req['questions'].items())
    calls = []
    def score(texts, schemas, **config):
        calls.append((texts, schemas, SimpleNamespace(**config)))
        return [SimpleNamespace(value=lambda qid, i=int(text): 'A' if i != 1 else 'B',
                    probabilities=lambda qid, i=int(text): {'true': i / 10, 'false': 1-i/10} if qid == 'congestion' else {'A': .6, 'B': .4},
                    confidence=lambda qid: .6) for text in texts]
    runtime.classifier = SimpleNamespace(compile_schema=lambda schema: schema,
        scorer=SimpleNamespace(batch_score=score), decode=lambda score, schema, **kwargs: score)
    enable_gliner_batch(runtime)
    outputs = runtime.native_batch(inputs, 2)
    assert [o['answers']['phase']['choice'] for o in outputs] == ['A', 'B', 'A']
    assert [o['answers']['congestion']['noul'] for o in outputs] == [0, .1, .2]
    assert calls[0][2].max_len is None and calls[0][2].batch_size == 6
    assert list(calls[0][1][2]['criteria']) == ['B', 'A']
    assert runtime.native_batch([], 2) == []
    with pytest.raises(ValueError, match='must be positive'):
        runtime.native_batch(inputs, 0)
    runtime.classifier.scorer.batch_score = lambda *args, **kwargs: []
    with pytest.raises(ValueError, match='result count mismatch'):
        runtime.native_batch(inputs, 2)

def test_open_jev_batches_records_and_restores_each_request(monkeypatch):
    runtime=OpenJevNative.__new__(OpenJevNative)
    api=ModuleType('jev.api'); metrics=ModuleType('jev.metrics')
    api.compile_request=lambda state,questions: [(state['id'],qid) for qid in questions]
    api.format_response=lambda records,probabilities: {'answers':{qid:result(i)['answers'][qid] for i,qid in records}}
    metrics.softmax=lambda values,temperature: values
    monkeypatch.setitem(sys.modules,'jev',ModuleType('jev'))
    monkeypatch.setitem(sys.modules,'jev.api',api); monkeypatch.setitem(sys.modules,'jev.metrics',metrics)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    class Logit:
        def float(self): return self
        def cpu(self): return self
        def tolist(self): return [0.6,0.4]
    calls=[]
    def forward(records):
        calls.append(records);return [Logit() for _ in records]
    runtime.model=forward; runtime.temperature=1.0
    inputs=[request(i) for i in range(3)]
    outputs=runtime.native_batch(inputs,2)
    assert [len(records) for records in calls]==[4,2]
    assert [list(o['answers']) for o in outputs]==[list(r['questions']) for r in inputs]

def test_clef_keeps_request_and_candidate_mappings(monkeypatch):
    runtime=ClefNative.__new__(ClefNative)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    calls=[]
    class Logit:
        def float(self): return self
        def softmax(self,dim): return self
        def tolist(self): return [0.6,0.4]
    class Model:
        def parameters(self): return iter([SimpleNamespace(device='cuda')])
        def __call__(self,records):
            calls.append(len(records));return [[Logit() for q in r.questions] for r in records]
    def encode(tokenizer,req,**kwargs):
        return SimpleNamespace(input_ids=[1]*12,questions=[SimpleNamespace(question_id=qid,
            option_ids=['A','B'] if q['type']=='choice' else ['true','false']) for qid,q in req['questions'].items()])
    def answer(question,probabilities):
        if question['type']=='noul':return {'type':'noul','noul':probabilities['true']}
        return {'type':'choice','choice':'A','probabilities':probabilities,'confidence':0.6}
    runtime.model=Model();runtime.model_id='clef';runtime.max_length=100
    runtime.processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    runtime.module=SimpleNamespace(encode_record=encode,collate_records=lambda records,*args:records,systemone_answer=answer)
    inputs=[request(i) for i in range(3)]
    outputs=runtime.native_batch(inputs,2)
    assert calls==[2,1]
    assert [list(o['answers']) for o in outputs]==[list(r['questions']) for r in inputs]

def test_agent_request_batch_uses_and_restores_native_path_microbatch():
    from jevlight.backends.decision_model_service import AgentNative
    runtime = AgentNative.__new__(AgentNative)
    calls = []
    engine = SimpleNamespace(trained_config=None, path_batch=1)
    def evaluate(payload):
        calls.append((engine.path_batch, len(payload['requests'])))
        return {'usage': {'truncated_inputs': 0}, 'results': [{'answers': [
            {'id': q['id'], 'type': q['type'], **({'probability': 0.2} if q['type'] == 'boolean' else
                {'value': 'A', 'distribution': {'A': 0.6, 'B': 0.4}, 'top_probability': 0.6})}
            for q in row['questions']]} for row in payload['requests']]}
    engine.evaluate = evaluate
    runtime.engine = engine
    inputs = [request(i) for i in range(3)]
    outputs = runtime.native_batch(inputs, 2)
    assert calls == [(2, 2), (2, 1)]
    assert engine.path_batch == 1
    assert [list(row['answers']) for row in outputs] == [list(row['questions']) for row in inputs]
