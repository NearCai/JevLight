"""GPU native decision batching, with capacity learned from actual GPU work."""
import gc
import importlib
import json
import math
import os
import time
from contextlib import contextmanager
from types import MethodType

from jevlight.backends.step_batch import _backbone, _bucket
from jevlight.step_batch_support import prepare_step_backend


def configure_execution():
    """Avoid compiling huge per-shape masks and oversubscribing host threads."""
    from jevlight.backends.cuda_runtime import configure_cpu_threads
    configure_cpu_threads()
    os.environ.update(TORCHDYNAMO_DISABLE='1',
                      TORCHINDUCTOR_COMPILE_THREADS='1',TOKENIZERS_PARALLELISM='false')
    import torch
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch._dynamo.config.disable=True


def prepare_native_runtime(backend):
    """Let CUDA allocation failures reach capacity selection, never move to CPU."""
    if backend.config['family']!='laya':
        return
    agent=backend.runtime.agent
    if hasattr(agent,'_native_original_infer'):
        return
    if agent._compiled or agent._fast is not None or agent.backend!='eager':
        raise ValueError('GPU Laya GPU inference requires the unchanged eager author head')
    agent._native_original_infer=agent._infer
    def gpu_infer(self,b):
        from laya.agent import _amp_context
        if self.device.type!='cuda':
            raise ValueError('GPU Laya runtime unexpectedly left the GPU')
        extras={name:b[name].to(self.device) for name in ('position_ids','option_ids') if name in b}
        with self._gate.read_lock(), _amp_context(self.device,self.dtype,
                self._amp_enabled_for(b['input_ids'].shape[0])):
            out=self.model(*(b[name].to(self.device) for name in
                ('input_ids','attention_mask','marker_pos','marker_mask','qtype')),**extras)
        self._amp_failures=0
        return out
    agent._infer=MethodType(gpu_infer,agent)


@contextmanager
def original_attention(backend):
    """Compare SUMO and timing against the original author's attention kernels."""
    import torch
    runtime=backend.runtime
    previous_mha=torch.backends.mha.get_fastpath_enabled()
    restored=[]
    try:
        if backend.config['family']=='laya':
            torch.backends.mha.set_fastpath_enabled(runtime._native_reference_mha)
        elif backend.config['family']=='gliner2':
            for module in runtime.model.modules():
                for name,original in (('forward','_native_forward'),('get_attention_mask','_deberta_attention_mask'),
                                      ('get_rel_pos','_native_relative_pos')):
                    if hasattr(module,original):
                        restored.append((module,name,getattr(module,name)))
                        setattr(module,name,getattr(module,original))
        elif backend.config['family']=='decider':
            if hasattr(runtime,'_native_native_conv'):
                from transformers.models.qwen3_5 import modeling_qwen3_5 as module
                restored.append((module,'causal_conv1d_fn',module.causal_conv1d_fn))
                module.causal_conv1d_fn=runtime._native_native_conv
        yield
    finally:
        for module,name,method in restored:
            setattr(module,name,method)
        torch.backends.mha.set_fastpath_enabled(previous_mha)


def _phase_answers(request, result, **kwargs):
    """Acceptance currently validates phase/mapping, never probability values."""
    if set(result['answers'])!=set(request['questions']):
        raise ValueError('Native answer IDs do not match the complete request')
    for qid,question in request['questions'].items():
        answer=result['answers'][qid]
        if answer.get('type')!=question['type']:
            raise ValueError('Native answer type changed')
        if question['type']=='choice' and answer.get('choice') not in question['criteria']:
            raise ValueError('Native phase is outside the original candidates')
    return result


def _agent_batch(self,requests,batch_size):
    from jev_service.contract import prepare, encode_paths, answer
    prepared=[]
    for request in requests:
        prepared.extend(prepare(self.payload(request)))
    config=self.engine.trained_config
    if config:
        previous=config.get('eval_batch_states')
        try:
            config['eval_batch_states']=len(requests)
            native=self.engine._evaluate_trained(prepared)
        finally:
            if previous is None:config.pop('eval_batch_states',None)
            else:config['eval_batch_states']=previous
    else:
        # Published safetensors use the author's path encoder rather than the
        # optional training-bundle collator. Retain its encoding/head/calibration
        # and lift only the JSON/state and path scheduling limits.
        from jev_service.prefix import encode
        torch=self.engine.torch
        paths,locations,questions=encode_paths(prepared,self.engine.tokenizer,self.engine.max_tokens)
        previous=self.engine.path_batch
        try:
            self.engine.path_batch=len(paths)
            with self.engine.lock, torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                vectors,mask,usage=encode(self.engine,paths,locations,questions,self.engine.encoder)
                logits=self.model._score(vectors,mask).float().masked_fill(~mask,float('-inf'))
                if self.engine.temperatures:
                    scaling=torch.tensor([self.engine.temperatures.get(q['type'],1.) for q in questions],
                                         device=self.engine.device)
                    logits=logits/scaling[:,None]
                probabilities=torch.softmax(logits,dim=-1).cpu().tolist()
        finally:
            self.engine.path_batch=previous
        rows=[];index=0
        for request in prepared:
            answers=[]
            for question in request['questions']:
                answers.append(answer(question,probabilities[index][:len(question['candidates'])]));index+=1
            rows.append(dict(id=request['id'],answers=answers))
        native=dict(results=rows,usage=dict(usage,truncated_inputs=0))
    if native['usage'].get('truncated_inputs'):
        raise ValueError('AgentJev truncated a complete batch input')
    results=[]
    for row in native['results']:
        answers={answer['id']:(dict(type='noul',noul=answer['probability']) if answer['type']=='boolean' else
            dict(type='choice',choice=answer['value'],probabilities=answer['distribution'],confidence=answer['top_probability']))
            for answer in row['answers']}
        results.append(dict(answers=answers,usage={}))
    return results


def _nimble_batch(self, requests, batch_size):
    import torch
    rows, locations = [], []
    for index, request in enumerate(requests):
        schema = self.schema(request)
        prepared = self.module.prepare_prompts(self.scorer.tokenizer,
            json.dumps(request['state'], ensure_ascii=False), schema, self.scorer.contract['max_length'])
        for i, qid in enumerate(prepared.names):
            row = dict(input_ids=prepared.full_ids[i], candidate_ids=prepared.candidate_ids[i], choices=prepared.choices[i])
            if schema[qid]['type'] == 'boolean':
                row['kind'] = 'noul'
            rows.append(row); locations.append((index, qid))
    results = [dict(answers={}, usage={'output_tokens': 0}) for _ in requests]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = {k:v.to('cuda') for k,v in self.scorer.collator(rows).items()}
        logits = self.module.candidate_logits(self.model, inputs).cpu()
    for row, (index, qid), z in zip(rows, locations, logits, strict=True):
        result = self.module.decision_result(row, z, temperature=self.scorer.temperature)
        answer = (dict(type='noul', noul=result['probability_true']) if row.get('kind') == 'noul' else
                  dict(type='choice', choice=result['prediction'], probabilities=result['probabilities'],
                       confidence=max(result['probabilities'].values())))
        results[index]['answers'][qid] = answer
    return results


def _von_batch(self, requests, batch_size):
    import torch
    from von.backends.option_marker_backend import _format_state
    texts = [];fit_keys=set()
    for request in requests:
        state = _format_state(request['state'])
        for definition in request['questions'].values():
            if definition['type'] == 'choice':
                descriptions = [description.strip() if description else name.strip()
                                for name, description in definition['criteria'].items()]
            elif definition['type'] == 'noul':
                descriptions = [(definition.get('criteria') or {}).get('true') or 'Yes, condition holds true.',
                                (definition.get('criteria') or {}).get('false') or 'No, condition is false.']
            else:
                raise ValueError('Unsupported Von decision question')
            if self.engine._fit_state(self.model, state, definition['instructions'], descriptions) != state:
                raise ValueError('Von shortened a batch state')
            fit_keys.add((state,definition['instructions'],tuple(descriptions)))
            texts.append(self.model.pack_sequence(state, definition['instructions'], descriptions))
    encoded = self.model.tokenizer(texts, return_tensors='pt', padding=True).to(self.engine.device)
    positions = [(row == self.model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
                 for row in encoded['input_ids']]
    with torch.inference_mode():
        logits = self.model(**encoded, mask_positions=positions, independent_options=self.engine._independent_options)
    original = self.model.forward
    cursor = 0
    def replay(**kwargs):
        nonlocal cursor
        if cursor >= len(logits):
            raise ValueError('Von native decoder made an unexpected extra forward')
        expected = encoded['input_ids'][cursor][encoded['attention_mask'][cursor].bool()]
        actual = kwargs['input_ids'][0][kwargs['attention_mask'][0].bool()]
        if not torch.equal(actual, expected):
            raise ValueError('Von native decoder input/mapping changed')
        result = [logits[cursor]]; cursor += 1
        return result
    try:
        # The exact author decoder retains its temperature, Noul prior and
        # confidence calibration; GPU logits are produced once for all rows.
        self.model.forward = replay
        from jevlight.backends.von_batch_cache import reuse_native_inputs
        with reuse_native_inputs(self,texts,encoded,fit_keys):
            results = [self.native(request) for request in requests]
        if cursor != len(logits):
            raise ValueError('Von native decoder did not consume every batch row')
        return results
    finally:
        self.model.forward = original


def _valen_batch(self, requests, batch_size):
    import torch
    answer = importlib.import_module(type(self.model).__module__.rsplit('.',1)[0]+'.responses').answer
    compiled = [self.compiler.compile({'request':self.request(request)}) for request in requests]
    # Author max_tokens is a scheduling budget, not the native context limit.
    # Capacity is governed by measured GPU memory instead of a fixed 32K total.
    budget = max(32768, len(compiled) * max(state.compute_tokens for state in compiled))
    with torch.inference_mode():
        rows = self.model.forward_state_batch(compiled, max_tokens=budget)
    results = []
    for state, logits in zip(compiled, rows, strict=True):
        answers = {q.qid:answer(q,z,1.0) for q,z in zip(state.questions,logits,strict=True)}
        results.append(dict(model=self.model.model_name, answers=answers,
                            usage={'input_tokens':state.logical_tokens,'output_tokens':0}))
    return results


def _resource_error(error):
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'canUse32BitIndexMath' in str(error))


def _try(runtime, group):
    import torch
    torch.cuda.reset_peak_memory_stats()
    observations = []
    backbone = _backbone(runtime)
    def record(module, args, kwargs):
        tensor = kwargs.get('input_ids')
        if tensor is None:
            tensor = next((x for x in args if torch.is_tensor(x) and x.is_cuda and x.ndim >= 2), None)
        if tensor is not None and tensor.is_cuda:
            observations.append(int(tensor.shape[0]))
    handle = backbone.register_forward_pre_hook(record, with_kwargs=True) if backbone is not None else None
    started = time.monotonic()
    try:
        result = runtime._native_author_batch(group, len(group))
        torch.cuda.synchronize()
        free,total = torch.cuda.mem_get_info()
        reserved,allocated = torch.cuda.memory_reserved(),torch.cuda.max_memory_allocated()
        # Cached allocator blocks belong to this process and are reclaimable.
        # Count peak live allocations and external processes, then retain margin.
        external = max(0, total-free-reserved)
        available = total-external-allocated
        margin = max(4 * 1024**3,total*.1)
        report = dict(status='ready', requests=len(group), gpu_forward_rows=observations,
            wall_ms=(time.monotonic()-started)*1000, peak_allocated_mib=allocated/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            available_after_peak_mib=available/1024**2, required_margin_mib=margin/1024**2)
        if len(result) != len(group):
            raise ValueError('GPU native batch result count mismatch')
        if available < margin:
            return None,dict(report,status='insufficient_memory_margin')
        return result,report
    except Exception as error:
        if not _resource_error(error):
            raise
        # No exception/traceback escapes this function retaining CUDA tensors.
        return None,dict(status='cuda_oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'kernel_index_limit',
                         requests=len(group),reason=str(error),gpu_forward_rows=observations)
    finally:
        if handle is not None: handle.remove()


def _capacity(runtime, requests, requested):
    import torch
    bucket = _bucket(requests)
    capacity = min(len(requests),requested,runtime._step_capacity.get(bucket,requested))
    evidence = []
    while capacity:
        result, report = _try(runtime, requests[:capacity])
        evidence.append(report)
        if result is not None:
            # A short final tail must not lower a learned capacity for the step.
            if len(evidence)>1: runtime._step_capacity[bucket]=capacity
            return result,capacity,evidence
        gc.collect(); torch.cuda.empty_cache()
        if capacity==1:
            raise RuntimeError('A complete GPU request cannot retain memory margin: '+report.get('reason',report['status']))
        capacity=max(1,capacity//2)
    raise RuntimeError('No GPU batch capacity fits')


def _probe(self, requests, requested):
    # Qualification starts from the whole step even when an earlier short
    # phase check learned a smaller capacity in the same length bucket.
    bucket=_bucket(requests)
    previous=self._step_capacity.pop(bucket,None)
    try:
        return _capacity(self,requests,requested)
    finally:
        if previous is not None:
            self._step_capacity[bucket]=min(previous,self._step_capacity.get(bucket,requested))


def _step(self, requests, batch_size):
    if type(batch_size) is not int or batch_size<1:
        raise ValueError('batch_size must be positive')
    requests=list(requests)
    results, evidence, forwards = [],[],[]
    started=time.monotonic()
    while len(results)<len(requests):
        group,capacity,attempts=_capacity(self,requests[len(results):],batch_size)
        results.extend(group);evidence.extend(attempts);forwards.extend(attempts[-1]['gpu_forward_rows'])
        notify=getattr(self._step_progress,'notify',None)
        if notify:notify()
    report=dict(request_count=len(requests),requested_capacity=batch_size,
        gpu_forward_rows=forwards,whole_step_one_forward=len(forwards)==1,
        capacity_attempts=evidence,wall_ms=(time.monotonic()-started)*1000,
        capacity_cache=[dict(co=k[0],char_bucket=k[1],capacity=v) for k,v in self._step_capacity.items()])
    self.last_step_batch=report
    for result in results:result.setdefault('usage',{})['step_batch']=report
    return results


def install(backend, *, attention='native'):
    from jevlight.backends.gliner_batch import enable_gliner_batch
    from jevlight.backends import decision_model_service
    runtime=backend.runtime
    if hasattr(runtime,'_native_author_batch'):
        return
    family=backend.config['family']
    prepare_native_runtime(backend)
    enable_gliner_batch(runtime)
    if family=='von': runtime.native_batch=MethodType(_von_batch,runtime)
    elif family=='nimble':
        runtime.native_batch=MethodType(_nimble_batch,runtime)
        base=runtime.model.model
        runtime.model.encoder=getattr(base,'language_model',base)
        from jevlight.backends.native_flash_batch import enable_native_flash
        enable_native_flash(runtime.model.encoder)
    elif family=='valen':runtime.native_batch=MethodType(_valen_batch,runtime)
    elif family=='agent-jev':runtime.native_batch=MethodType(_agent_batch,runtime)
    elif family=='kev':
        from jevlight.backends.kev_prefix_batch import enable_kev_prefix_batch
        align_states=backend.config['model_id']=='jaredpalmer/kev-0.8b'
        enable_kev_prefix_batch(runtime,align_states=align_states)
        from jevlight.backends.native_flash_batch import enable_native_flash
        enable_native_flash(getattr(runtime.model.lm,'model',runtime.model.lm),
            pack_single=(lambda:runtime._native_prefix_meta is not None) if align_states else None)
        runtime._native_varlen_flash=True
    elif family=='decider':
        from jevlight.backends.native_row_batch import enable_decider_full_rows
        enable_decider_full_rows(runtime)
        # slot_logits calls lm.model directly, bypassing the parent LM's hook.
        # Observe the very same decoder through an alias; no forward changes.
        runtime.model.encoder=runtime.model.lm.model
        from jevlight.backends.causal_conv import enable_batch_conv
        enable_batch_conv(runtime)
    elif family=='clef':
        from jevlight.backends.batched_linear import enable_sequence_linears
        base=runtime.model.language_model
        base=base.get_base_model() if hasattr(base,'get_base_model') else base
        decoder=base.model
        decoder=getattr(decoder,'language_model',decoder)
        enable_sequence_linears(decoder,include=('in_proj_a','in_proj_b'))
        from jevlight.backends.causal_conv import enable_batch_conv
        enable_batch_conv(runtime)
        from jevlight.backends.native_flash_batch import enable_native_flash
        enable_native_flash(decoder)
        runtime.model.encoder=decoder
    if not hasattr(runtime,'native_batch'):
        raise ValueError(f'{family} has no independent-input GPU batch')
    if attention=='laya_sdpa' and family=='laya':
        import torch
        runtime._native_reference_mha=torch.backends.mha.get_fastpath_enabled()
        # The fused TransformerEncoderLayer path expands a quadratic padding
        # mask. The author's existing SDPA path uses the same weights and mask.
        torch.backends.mha.set_fastpath_enabled(False)
    elif attention=='deberta_tiled':
        from jevlight.backends.deberta_attention import enable_deberta_attention
        if not enable_deberta_attention(runtime.model):
            raise ValueError('No compatible DeBERTa attention layers')
    elif attention!='native':
        raise ValueError('Unsupported attention engine')
    runtime._native_author_batch=runtime.native_batch
    prepare_step_backend(backend)
    runtime._step_reporting_native=MethodType(_step,runtime)
    runtime._step_reporting_probe=MethodType(_probe,runtime)
    backend.config['batch_size']=int(backend.config.get('batch_size',196))
    # Validation is scoped to this serving process.
    backend.answer_validator=_phase_answers
