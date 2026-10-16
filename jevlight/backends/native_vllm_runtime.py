"""Native choice heads and complete author tokens through one local vLLM GPU."""
import dataclasses
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]


def validate_choice(request,value):
    if list(value['answers'])!=list(request['questions']):raise ValueError('Engine question mapping changed')
    for qid,definition in request['questions'].items():
        answer=value['answers'][qid]
        if answer.get('type')!='choice' or answer.get('choice') not in definition['criteria']:
            raise ValueError('Engine returned an invalid native phase')
    return value

def observe_gpu(worker,path):
    import torch
    from vllm.forward_context import get_forward_context
    worker._native_engine_blocks=[]
    def observe(module,args,kwargs,result):
        def counts(metadata):
            if isinstance(metadata,dict):return [v for item in metadata.values() for v in counts(item)]
            if isinstance(metadata,(list,tuple)):return [v for item in metadata for v in counts(item)]
            lengths=getattr(metadata,'seq_lens',None)
            starts=getattr(metadata,'query_start_loc',None)
            if starts is not None:return [int((starts[1:]>starts[:-1]).sum().item())]
            if lengths is not None:return [int((lengths>0).sum().item())]
            return []
        values=counts(get_forward_context().attn_metadata)
        ids=kwargs.get('input_ids',args[0] if args else None)
        if not values or not torch.is_tensor(ids) or not ids.is_cuda:return
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        item=dict(completed_at=time.time(),gpu_rows=max(values),input_tokens=ids.numel(),
            memory_allocated_mib=torch.cuda.memory_allocated()/2**20,
            free_memory_mib=free/2**20,total_memory_mib=total/2**20)
        worker._native_engine_blocks.append(item)
        callback=getattr(worker,'_native_engine_completed',None)
        if callback:callback(item)
        with open(path,'a') as stream:stream.write(json.dumps(item)+'\n')
    model=worker.model_runner.model
    # vLLM's compile decorator replaces Module.__call__ and bypasses ordinary
    # PyTorch hooks even in eager mode. Observe the actual callable instead.
    cls=type(model);original_call=cls.__call__;original_forward=model.forward
    def call(self,*args,**kwargs):
        if self is not model:return original_call(self,*args,**kwargs)
        self._native_engine_inside_call=True
        try:
            result=original_call(self,*args,**kwargs)
            observe(self,args,kwargs,result)
            return result
        finally:self._native_engine_inside_call=False
    def forward(*args,**kwargs):
        result=original_forward(*args,**kwargs)
        if not getattr(model,'_native_engine_inside_call',False):observe(model,args,kwargs,result)
        return result
    cls.__call__=call;model.forward=forward
    return dict(device=str(worker.device),installed=True)

def blocks(worker,clear=False):
    values=list(worker._native_engine_blocks)
    if clear:worker._native_engine_blocks.clear()
    return values

class Head:
    def __init__(self,config,checkpoint):
        self.family=config['family']
        self.config=config
        self.checkpoint=checkpoint
        if self.family=='laya':
            from vllm_jev.laya import LayaService
            self.service=LayaService(None,checkpoint,config['model_id'],max_inflight=int(config.get('batch_size',196)))
        elif self.family in ('decider','kev'):
            from vllm_jev.decision import DecisionService
            self.service=DecisionService(None,checkpoint,config['model_id'],config['max_length'],max_inflight=int(config.get('batch_size',196)))
        elif self.family=='clef':
            from vllm_jev.clef import ClefService
            self.service=ClefService(None,checkpoint,config['model_id'],config['max_length'],max_inflight=int(config.get('batch_size',196)))
        elif self.family=='agent-jev':
            import torch
            from safetensors.torch import load_file
            from jevlight.backends.decision_model_service import add_author_source
            add_author_source('agent-jev')
            from agentjev.model import AgentJevModel,CandidateSetEncoder,ScalarScorer
            self.head=AgentJevModel.__new__(AgentJevModel);torch.nn.Module.__init__(self.head)
            weights=load_file(str(checkpoint/'agent_head.safetensors'))
            width,hidden=weights['proj_in.weight'].shape
            self.head.proj_in=torch.nn.Linear(hidden,width)
            self.head.set_encoder=CandidateSetEncoder(width,4,2)
            self.head.proj_out=torch.nn.Linear(width,hidden)
            self.head.scorer=ScalarScorer(hidden,width)
            self.head.load_state_dict(weights,strict=True);self.head.to('cuda',dtype=torch.bfloat16).eval()
        elif self.family=='nimble':
            from safetensors import safe_open
            from jevlight.backends.decision_model_service import author_file
            source=ROOT/config['weights_path']
            __import__('sys').path.insert(0,str(source))
            self.module=author_file(source/'inference.py','native_vllm_native_nimble_head')
            index=json.loads((checkpoint/'model.safetensors.index.json').read_text())
            with safe_open(checkpoint/index['weight_map']['lm_head.weight'],framework='pt',device='cpu') as file:
                self.vocabulary=file.get_tensor('lm_head.weight')

    def rows(self,entry):
        encoded=entry['encoded']
        if self.family in ('laya','decider'):return [item['ids'] for item in encoded['items']]
        if self.family in ('kev','nimble'):return [row['ids'] for row in encoded['rows']]
        if self.family=='agent-jev':return encoded['paths']
        return [list(encoded['input_ids'])]

    def answer(self,entry,hidden):
        import torch
        request=entry['request'];encoded=entry['encoded']
        if self.family=='laya':
            from laya.common import collate_items
            items=encoded['items'];batch=collate_items([items],self.service.protocol.tok.pad_token_id)
            tensor=torch.zeros((len(items),batch['input_ids'].shape[1],hidden[0].shape[-1]),device='cuda')
            for i,h in enumerate(hidden):tensor[i,:len(items[i]['ids'])]=h.to('cuda',dtype=tensor.dtype)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits,action=self.service.head(tensor,*(batch[name].to('cuda') for name in
                    ('attention_mask','marker_pos','marker_mask','qtype')))
            answers=self.service.protocol._decode_answers(logits.float().cpu().numpy(),
                torch.softmax(action.float(),-1).cpu().numpy(),items,encoded['identifiers'],encoded['internal'],0)
        elif self.family=='decider':
            answers={}
            for qid,item,h in zip(request['questions'],encoded['items'],hidden,strict=True):
                if len(item['slots'])!=1 or item['slots'][0]!=len(item['ids'])-1:
                    raise ValueError('Decider native schema requires a dedicated packed-slot adapter')
                keys=list(request['questions'][qid]['criteria'])
                p=self.service._probabilities(h,[item['slots'][0]],len(keys),'choice')
                answers[qid]=dict(type='choice',choice=keys[max(range(len(keys)),key=p.__getitem__)],probabilities=dict(zip(keys,p)))
        elif self.family=='kev':
            answers={}
            for qid,row,h in zip(request['questions'],encoded['rows'],hidden,strict=True):
                keys=list(request['questions'][qid]['criteria'])
                positions=row['readouts'][1:]+row['readouts'][:1]
                p=self.service._probabilities(h,positions,len(keys),'choice')
                answers[qid]=dict(type='choice',choice=keys[max(range(len(keys)),key=p.__getitem__)],probabilities=dict(zip(keys,p)))
        elif self.family=='agent-jev':
            locations=encoded['locations'];questions=encoded['questions']
            mask=torch.zeros((len(questions),max(len(q['keys']) for q in questions)),device='cuda',dtype=torch.bool)
            vectors=torch.zeros((*mask.shape,hidden[0].numel()),device='cuda',dtype=torch.bfloat16)
            for h,(qi,ci) in zip(hidden,locations,strict=True):vectors[qi,ci]=h.to('cuda');mask[qi,ci]=True
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=self.head._score(vectors,mask).float()
                from jev_service.contract import answer
                temperatures_path=self.checkpoint/'temperatures.json'
                temperatures=json.loads(temperatures_path.read_text()) if temperatures_path.exists() else {}
                if temperatures:
                    scale=torch.tensor([temperatures.get(q['type'],{}).get('temperature',1.)
                        for q in questions],device='cuda')
                    logits=logits/scale[:,None]
                probabilities=logits.softmax(-1).cpu().tolist()
            native=[answer(q,p[:len(q['keys'])]) for q,p in zip(questions,probabilities,strict=True)]
            answers={a['id']:dict(type='choice',choice=a['value'],probabilities=a['distribution']) for a in native}
        elif self.family=='nimble':
            answers={}
            for row,h in zip(encoded['rows'],hidden,strict=True):
                weight=self.vocabulary[row['candidate_ids']].to('cuda',dtype=torch.bfloat16)
                logits=torch.nn.functional.linear(h.to('cuda',dtype=torch.bfloat16),weight).float().cpu()
                value=self.module.decision_result(row,logits,temperature=self.module.DEFAULT_TEMPERATURE)
                answers[row['question_id']]=dict(type='choice',choice=value['prediction'],probabilities=value['probabilities'])
        else:
            from vllm_jev.clef import EncodedQuestion,answer
            questions=[EncodedQuestion(**q) for q in encoded['questions']]
            lexical=[]
            for question in questions:
                lexical.append(torch.stack([self.service.output_embedding[
                    encoded['input_ids'][start:end]].mean(0) for start,end in question.option_spans]).to('cuda'))
            with torch.inference_mode():logits=self.service.head(hidden[0].to('cuda',dtype=torch.bfloat16),questions,lexical)
            answers={q.question_id:answer(request['questions'][q.question_id],
                dict(zip(q.option_ids,z.float().softmax(-1).tolist()))) for q,z in zip(questions,logits,strict=True)}
        value=dict(answers=answers,usage={'truncated':False})
        validate_choice(request,value)
        return value


class VllmDecisionRuntime:
    emits_gpu_chunk_progress=True

    def __init__(self,config,selection):
        import torch
        from jevlight.backends.cuda_runtime import configure_cpu_threads
        configure_cpu_threads()
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
        from vllm import LLM
        self.config=config;self.family=config['family'];self.selection=selection
        self.checkpoint=Path(selection['checkpoint'])
        self.head=Head(config,self.checkpoint)
        self._step_progress=threading.local()
        self.capacity=int(selection['capacity'])
        self.window=int(config['max_length'])
        self.task='embed' if self.family in ('agent-jev','nimble') else 'token_embed'
        pooler=dict(task=self.task,use_activation=False)
        pooler['seq_pooling_type' if self.task=='embed' else 'tok_pooling_type']='LAST' if self.task=='embed' else 'ALL'
        longest=max(v['largest_encoded_tokens'] for v in selection['native_modes'].values())
        multiplier=4 if self.family=='agent-jev' else 1
        self.engine=LLM(model=str(self.checkpoint),runner='pooling',convert='none' if self.family in ('laya','clef') else 'embed',
            pooler_config=pooler,dtype='bfloat16',max_model_len=self.window,
            max_num_seqs=self.capacity*multiplier,max_num_batched_tokens=self.capacity*multiplier*longest,
            enable_chunked_prefill=self.family not in ('laya',),
            gpu_memory_utilization=float(selection.get('gpu_memory_utilization',.80)),enable_prefix_caching=False,enforce_eager=True,
            trust_remote_code=False,mamba_ssm_cache_dtype='float32',limit_mm_per_prompt={'image':0,'video':0})
        self.progress_path=str(selection['progress_path'])
        self.engine.collective_rpc(observe_gpu,args=(self.progress_path,))
        def listener(worker,runtime):
            def completed(event):
                notify=getattr(runtime._step_progress,'notify',None)
                if notify:notify()
            worker._native_engine_completed=completed
        self.engine.collective_rpc(listener,args=(self,))
        self._step_author_batch=self.native_batch
        self._initialize_codec()

    def _initialize_codec(self):
        from transformers import AutoTokenizer
        from jevlight.backends.decision_model_service import add_author_source,author_file
        source=ROOT/self.config['weights_path']
        if self.family=='laya':
            self.codec=self.head.service.protocol
        elif self.family=='decider':
            import sys
            sys.path.insert(0,str(source))
            from decider.infer import Decider
            cfg=json.loads((source/'decider_config.json').read_text())
            self.codec=Decider.__new__(Decider)
            self.codec.m=SimpleNamespace(tok=AutoTokenizer.from_pretrained(source,local_files_only=True))
            self.codec.schema_first=False;self.codec.isolated_levels=bool(cfg.get('isolated_levels',False))
            self.codec.neutralize_none=bool(cfg.get('neutralize_none',True));self.codec.chat=None
            if cfg.get('layout','plain')!='plain':raise ValueError('Frozen Decider chat layout needs its original chat compiler')
        elif self.family=='kev':
            add_author_source('kev')
            self.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        elif self.family=='agent-jev':
            add_author_source('agent-jev')
            self.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        elif self.family=='nimble':
            self.module=self.head.module
            self.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
            self.contract=json.loads((source/'schema_config.json').read_text())
        else:
            self.module=author_file(source/'joint_schema_model.py','native_vllm_original_clef_compiler')
            self.processor=self.head.service.processor

    def _compile(self,request,unlimited=False):
        if any(q['type']!='choice' for q in request['questions'].values()):
            raise ValueError('This qualified experiment adapter supports the original phase-only protocol')
        limit=1_000_000 if unlimited else self.window
        if self.family=='laya':
            ids=list(request['questions']);internal={qid:self.codec._to_internal(request['questions'][qid]) for qid in ids}
            from laya.agent import Agent
            items=Agent._encode_state(self.codec,request['state'],ids,internal,max_len=limit,
                head_max_len=1_000_000 if unlimited else self.config['head_max_length'])
            return dict(items=items,identifiers=ids,internal=internal)
        if self.family=='decider':
            rendered,index,items=self.codec._system_one_items(**request,
                max_state_tokens=1_000_000 if unlimited else self.config['max_state_tokens'])
            return dict(items=items,rendered=rendered,index=index)
        if self.family=='kev':
            from kev.api import SystemOneRequest,to_record
            from kev.model import encode,rows_of
            record,meta=to_record(SystemOneRequest(**request))
            encoded=encode(self.tokenizer,record,max_state=1_000_000 if unlimited else self.config['max_state_tokens'],
                max_branch=limit,strict=True,option_isolation=False)
            state,positions,rows=rows_of(encoded)
            return dict(rows=[dict(ids=state+r['ids'],readouts=[len(state)+r['decide']]+[len(state)+p for p in r['opts']]) for r in rows],meta=meta)
        if self.family=='agent-jev':
            from jev_service.contract import prepare,encode_paths
            from jevlight.backends.decision_model_service import AgentNative
            paths,locations,questions=encode_paths(prepare(AgentNative.payload(None,request)),self.tokenizer,limit)
            return dict(paths=paths,locations=locations,questions=questions)
        if self.family=='nimble':
            schema={qid:dict(type='enum',description=q['instructions'],choices=list(q['criteria']),choice_descriptions=q['criteria']) for qid,q in request['questions'].items()}
            prepared=self.module.prepare_prompts(self.tokenizer,json.dumps(request['state'],ensure_ascii=False),schema,limit)
            return dict(rows=[dict(ids=prepared.full_ids[i],candidate_ids=prepared.candidate_ids[i],choices=prepared.choices[i],question_id=qid) for i,qid in enumerate(prepared.names)])
        record=self.module.encode_record(self.processor.tokenizer,dict(request,model=self.config['model_id']),max_length=limit,processor=self.processor)
        encoded=dataclasses.asdict(record)
        encoded['input_ids']=list(encoded['input_ids'])
        return encoded

    def entry(self,request):
        encoded=self._compile(request);entry=dict(request=request,encoded=encoded)
        full=dict(request=request,encoded=self._compile(request,unlimited=True))
        if self.head.rows(entry)!=self.head.rows(full):raise ValueError('Engine compiler would truncate original input')
        if any(not ids or len(ids)>self.window for ids in self.head.rows(entry)):raise ValueError('Complete input exceeds qualified engine window')
        return entry

    def encode(self,request):
        entry=self.entry(request)
        return [dict(encoded_tokens=len(ids),max_length=self.window,truncated=False) for ids in self.head.rows(entry)]

    def native(self,request):return self.native_batch([request],int(self.config.get('batch_size',196)))[0]

    def native_batch(self,requests,batch_size):
        import torch
        from vllm import PoolingParams
        entries=[self.entry(request) for request in requests]
        rows=[ids for entry in entries for ids in self.head.rows(entry)]
        self.engine.collective_rpc(blocks,args=(True,))
        outputs=self.engine.encode([dict(prompt_token_ids=ids) for ids in rows],
            PoolingParams(task=self.task,use_activation=False),pooling_task=self.task,use_tqdm=False)
        states=[]
        if len(outputs)!=len(rows):raise ValueError('Engine lost independent native rows')
        for ids,output in zip(rows,outputs,strict=True):
            if output.prompt_token_ids!=ids:raise ValueError('Engine changed original input token mapping')
            value=torch.as_tensor(output.outputs.data)
            if self.task=='token_embed' and value.shape[0]!=len(ids):raise ValueError('Engine omitted complete native states')
            states.append(value)
        results=[];offset=0
        for entry in entries:
            count=len(self.head.rows(entry));results.append(self.head.answer(entry,states[offset:offset+count]));offset+=count
        evidence=self.engine.collective_rpc(blocks,args=(True,))[0]
        self.last_step_batch=dict(request_count=len(requests),requested_capacity=batch_size,
            gpu_forward_rows=[row['gpu_rows'] for row in evidence],engine='vllm',engine_capacity=self.capacity,
            gpu_blocks=evidence,whole_step_one_forward=len(evidence)==1)
        for result in results:result['usage']['step_batch']=self.last_step_batch
        return results

    def probe_step_capacity(self,requests,batch_size):
        results=self.native_batch(requests,batch_size)
        return results,self.capacity,self.last_step_batch['gpu_blocks']


def backend(config,selection):
    from jevlight.backends.decision_model_service import NativeBackend
    from jevlight.backends import decision_model_service

    value=NativeBackend.__new__(NativeBackend)
    value.config=dict(config,batch_size=int(config.get('batch_size',196)),inference_engine='vllm_native_heads')
    value.lock=threading.Lock();value.reported_model=config['model_id'];value.log_path=None
    value.answer_validator=validate_choice
    value.runtime=VllmDecisionRuntime(value.config,selection)
    return value
