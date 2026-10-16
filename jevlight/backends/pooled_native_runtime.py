"""Shared guarded GPU pooling and progress for native decision runtimes."""
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


class PoolingNativeBase:
    emits_gpu_chunk_progress=True

    def initialize_engine(self,config,source,*,task='token_embed',architecture='NativeMarkerEngine',**options):
        import torch
        from jevlight.backends.cuda_runtime import prepare_cuda_device, configure_cpu_threads
        prepare_cuda_device()
        configure_cpu_threads(default=8)
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        if options.get('enable_lora'):
            from jevlight.backends.vllm_lora import install_fp32_lora_linear
            install_fp32_lora_linear()
        from transformers import AutoConfig
        from vllm import LLM,ModelRegistry
        from vllm.config import PoolerConfig
        from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig,Qwen3_5Config
        class ValenConfig(Qwen3_5Config):model_type='valen_qwen'
        AutoConfig.register('qwen3_5_text',Qwen3_5TextConfig,exist_ok=True)
        AutoConfig.register('valen_qwen',ValenConfig,exist_ok=True)
        ModelRegistry.register_model(architecture,'jevlight.backends.marker_vllm_models:'+architecture)
        self.config=config;self.source=Path(source);self.task=task
        self.max_length=int(config.get('engine_max_length',config['max_length']))
        if self.max_length>config['max_length']:raise ValueError('Engine window exceeds the model window')
        pooler=(PoolerConfig(seq_pooling_type='LAST',use_activation=False)if task=='embed'
                else PoolerConfig(task=task,tok_pooling_type='ALL',use_activation=False))
        self.engine=LLM(model=str(source),runner='pooling',convert='embed',
            hf_overrides={'architectures':[architecture]},pooler_config=pooler,skip_tokenizer_init=True,
            dtype='bfloat16',tensor_parallel_size=1,max_model_len=self.max_length,
            max_num_seqs=int(config.get('engine_max_sequences',196)),max_num_batched_tokens=int(config['engine_token_budget']),
            gpu_memory_utilization=float(config.get('engine_gpu_memory_utilization',.85)),
            enable_prefix_caching=False,enforce_eager=True,**options)
        self.model=torch.nn.Module();self.model.backbone=self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        self.last_engine_forward_rows=[];self.engine_forward_evidence=[]
        self.model.backbone.register_forward_hook(self._completed_forward,with_kwargs=True)

    def _completed_forward(self,module,args,kwargs,value):
        import torch
        from vllm.forward_context import get_forward_context
        metadata=get_forward_context().attn_metadata;values=list(metadata.values())if isinstance(metadata,dict)else[metadata]
        rows=max((int(v.query_start_loc.numel()-1)for v in values if getattr(v,'query_start_loc',None)is not None),default=0)
        if rows<1:raise ValueError('Actual GPU sequence mapping is unavailable')
        torch.cuda.synchronize();self.last_engine_forward_rows.append(rows)
        self.engine_forward_evidence.append(dict(rows=rows,input_tokens=int(kwargs['input_ids'].numel())))
        notify=getattr(getattr(self,'_step_progress',None),'notify',None)
        if notify:notify()

    def pool(self,rows,*,lora=None):
        from vllm import PoolingParams
        if any(not ids or len(ids)>self.max_length for ids in rows):
            raise ValueError('Complete native tokens exceed the verified engine window')
        self.last_engine_forward_rows.clear();self.engine_forward_evidence.clear()
        values=self.engine.encode([dict(prompt_token_ids=ids)for ids in rows],
            PoolingParams(task=self.task,use_activation=False),pooling_task=self.task,lora_request=lora,use_tqdm=False)
        if len(values)!=len(rows)or any(value.prompt_token_ids!=ids for value,ids in zip(values,rows,strict=True)):
            raise ValueError('The engine omitted tokens or changed native row mapping')
        if not self.last_engine_forward_rows:raise ValueError('No actual GPU execution')
        return values

    def native(self,request):
        return self.native_batch([request],1)[0]
