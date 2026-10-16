"""Complete author marker encoding and native vocabulary readout through vLLM."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class NativeMarkerPooling:
    emits_gpu_chunk_progress = True

    def __new__(cls,config):
        if config['family']=='decider':return super().__new__(cls)
        import importlib
        delegates={'lev':('jevlight.backends.lev_vllm_runtime','LevPoolingNative'),
                   'valen':('jevlight.backends.valen_vllm_runtime','ValenPoolingNative'),
                   'kev':('jevlight.backends.kev_vllm_runtime','KevPoolingNative'),
                   'clef':('jevlight.backends.clef_vllm_runtime','ClefPoolingNative')}
        if config['family']not in delegates:raise ValueError('Unsupported native marker head')
        module,name=delegates[config['family']]
        return getattr(importlib.import_module(module),name)(config)

    def __init__(self, config):
        if config['family'] != 'decider':
            raise ValueError('This qualified marker runtime currently supports Decider')
        import torch
        from jevlight.backends.cuda_runtime import prepare_cuda_device, configure_cpu_threads
        prepare_cuda_device()
        configure_cpu_threads(default=8)
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, ModelRegistry
        from vllm.config import PoolerConfig
        from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
        AutoConfig.register('qwen3_5_text', Qwen3_5TextConfig, exist_ok=True)
        ModelRegistry.register_model('NativeMarkerEngine', 'jevlight.backends.marker_vllm_models:NativeMarkerEngine')
        self.config = config
        self.source = ROOT/config['weights_path']
        self.max_length = int(config.get('engine_max_length', config['max_length']))
        if self.max_length > config['max_length']:
            raise ValueError('The engine window exceeds the model window')
        self.max_state = config['max_state_tokens']
        sys.path.insert(0, str(self.source))
        from decider.infer import Decider
        from decider.prompt import letter_ids
        calibration = json.loads((self.source/'decider_config.json').read_text())
        if calibration.get('layout', 'plain') != 'plain':
            raise ValueError('The frozen plain compiler is required')
        self.codec = Decider.__new__(Decider)
        self.codec.m = SimpleNamespace(tok=AutoTokenizer.from_pretrained(self.source, local_files_only=True))
        self.codec.schema_first = False
        self.codec.isolated_levels = bool(calibration.get('isolated_levels', False))
        self.codec.neutralize_none = bool(calibration.get('neutralize_none', True))
        self.codec.chat = None
        self.temperature = calibration.get('temperature', 1.)
        self.temperatures = calibration.get('temperature_by_type', {})
        letters = letter_ids(self.codec.m.tok)
        from safetensors import safe_open
        index_path = self.source/'model.safetensors.index.json'
        if index_path.exists():
            index = json.loads(index_path.read_text())['weight_map']
            name = next((n for n in ['lm_head.weight', 'model.embed_tokens.weight',
                                    'model.language_model.embed_tokens.weight'] if n in index), '')
            shard = self.source/index[name]
        else:
            shard = self.source/'model.safetensors'
            name = None
        with safe_open(shard, framework='pt', device='cpu') as archive:
            name = name or next((n for n in ['lm_head.weight', 'model.embed_tokens.weight',
                                            'model.language_model.embed_tokens.weight'] if n in archive.keys()), '')
            if name != 'lm_head.weight' and not json.loads((self.source/'config.json').read_text()).get('tie_word_embeddings'):
                raise ValueError('An untied vocabulary projection is missing')
            self.weight = archive.get_tensor(name)[letters].clone()
        if self.weight.dtype != torch.bfloat16:
            raise ValueError('The released vocabulary projection precision differs')
        self.engine = LLM(model=str(self.source), runner='pooling', convert='embed',
            hf_overrides={'architectures':['NativeMarkerEngine']},
            pooler_config=PoolerConfig(task='token_embed',tok_pooling_type='ALL',use_activation=False),
            skip_tokenizer_init=True,dtype='bfloat16',tensor_parallel_size=1,max_model_len=self.max_length,
            max_num_seqs=196,max_num_batched_tokens=int(config['engine_token_budget']),
            gpu_memory_utilization=float(config.get('engine_gpu_memory_utilization',.85)),
            enable_prefix_caching=False,enforce_eager=True)
        self.weight = self.weight.to('cuda')
        self.model = torch.nn.Module()
        self.model.backbone = self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        self.last_engine_forward_rows = []
        self.engine_forward_evidence = []
        self.model.backbone.register_forward_hook(self._completed_forward,with_kwargs=True)

    def _completed_forward(self,module,args,kwargs,value):
        import torch
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        values = list(metadata.values()) if isinstance(metadata,dict) else [metadata]
        rows = max((int(v.query_start_loc.numel()-1) for v in values
                    if getattr(v,'query_start_loc',None) is not None),default=0)
        if rows < 1:
            raise ValueError('Actual GPU sequence mapping is unavailable')
        torch.cuda.synchronize()
        self.last_engine_forward_rows.append(rows)
        self.engine_forward_evidence.append(dict(rows=rows,input_tokens=int(kwargs['input_ids'].numel())))
        notify = getattr(getattr(self,'_step_progress',None),'notify',None)
        if notify:
            notify()

    def _compile(self, request):
        if any(q['type'] != 'choice' for q in request['questions'].values()):
            raise ValueError('The qualified experiment uses the phase-only native protocol')
        _,_,rows = self.codec._system_one_items(**request,max_state_tokens=self.max_state)
        _,_,unlimited = self.codec._system_one_items(**request,max_state_tokens=1_000_000)
        if rows != unlimited or len(rows) != len(request['questions']):
            raise ValueError('Original independent rows would truncate or change mapping')
        if any(len(row['ids'])>self.max_length or len(row['slots'])!=1 for row in rows):
            raise ValueError('Complete native choice row exceeds the verified window or slot contract')
        return rows

    def encode(self, request):
        return [dict(encoded_tokens=len(row['ids']),max_length=self.max_length,truncated=False)
                for row in self._compile(request)]

    def native(self, request):
        return self.native_batch([request],1)[0]

    def native_batch(self, requests, batch_size):
        import torch
        from vllm import PoolingParams
        from jevlight.backends.decision_model_service import validate_answers
        packed = [self._compile(request) for request in requests]
        rows = [row for group in packed for row in group]
        self.last_engine_forward_rows.clear()
        self.engine_forward_evidence.clear()
        with torch.inference_mode():
            values = self.engine.encode([dict(prompt_token_ids=row['ids'])for row in rows],
                pooling_params=PoolingParams(task='token_embed',use_activation=False),
                pooling_task='token_embed',use_tqdm=False)
            if len(values)!=len(rows):
                raise ValueError('The engine changed native row mapping')
            results=[];cursor=0
            for request,group in zip(requests,packed,strict=True):
                answers={}
                for qid,row in zip(request['questions'],group,strict=True):
                    value=values[cursor];cursor+=1
                    if value.prompt_token_ids!=row['ids']:
                        raise ValueError('The engine omitted native tokens')
                    hidden=torch.as_tensor(value.outputs.data)
                    if hidden.ndim!=2 or hidden.shape[0]!=len(row['ids']):
                        raise ValueError('The engine omitted native token states')
                    h=hidden[row['slots']].to('cuda',dtype=torch.bfloat16)
                    keys=list(request['questions'][qid]['criteria'])
                    if row['nopts'] != [len(keys)]:
                        raise ValueError('Native candidate order or count changed')
                    logits=torch.nn.functional.linear(h,self.weight)[0].float()[:len(keys)]
                    temperature=self.temperatures.get('choice',self.temperature)
                    if temperature<=0 or not torch.isfinite(logits).all():
                        raise ValueError('Native head/calibration is invalid')
                    p=(logits/temperature).softmax(-1).cpu().tolist()
                    confidence=1. if len(p)==1 else max(0.,(len(p)*max(p)-1)/(len(p)-1))
                    answers[qid]=dict(type='choice',choice=keys[max(range(len(p)),key=p.__getitem__)],
                        probabilities=dict(zip(keys,p,strict=True)),confidence=confidence)
                result=dict(model=self.config['model_id'],answers=answers,
                            usage=dict(input_tokens=sum(len(r['ids'])for r in group),output_tokens=0))
                results.append(validate_answers(request,result))
        if not self.last_engine_forward_rows:
            raise ValueError('The batch did not execute on the real GPU')
        return results
