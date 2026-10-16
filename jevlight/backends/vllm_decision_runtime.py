"""Complete Open-Jev candidate paths through a frozen pooling engine."""
import json
import os
from pathlib import Path
import subprocess


class OpenJevPoolingNative:
    emits_gpu_chunk_progress = True

    def __new__(cls, config):
        if cls is OpenJevPoolingNative and config.get('engine_adapter') == 'author_merged':
            from jevlight.backends.open_jev_merged_vllm_runtime import MergedOpenJevPoolingNative
            return object.__new__(MergedOpenJevPoolingNative)
        return object.__new__(cls)

    def __init__(self, config):
        import torch
        from jevlight.backends.decision_model_service import ROOT, add_author_source, local_base
        from jevlight.backends.cuda_runtime import prepare_cuda_device
        prepare_cuda_device()
        add_author_source('open-jev')
        self.checkpoint = ROOT/config['weights_path']/'package/checkpoint'
        metadata = json.loads((self.checkpoint/'model.json').read_text())
        base = str(local_base(config, metadata['model_id'], metadata['revision']))
        self.max_length = int(config.get('engine_max_length',config['max_length']))
        if self.max_length > config['max_length']:
            raise ValueError('Engine window exceeds the actual model window')
        self.temperature = json.loads((self.checkpoint/'temperature.json').read_text())['temperature']
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        from vllm import LLM, ModelRegistry
        from vllm.config import PoolerConfig
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        from jevlight.backends.vllm_lora import install_fp32_lora_linear
        install_fp32_lora_linear()
        ModelRegistry.register_model('OpenJevEngineForEmbedding',
            'jevlight.backends.vllm_decision_models:OpenJevEngineForEmbedding')
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.engine = LLM(model=base, runner='pooling', convert='embed',
            hf_overrides={'architectures': ['OpenJevEngineForEmbedding']},
            pooler_config=PoolerConfig(seq_pooling_type='LAST', use_activation=False),
            skip_tokenizer_init=True, dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=self.max_length, enable_lora=True, max_lora_rank=metadata['lora_rank'],
            max_loras=1, lora_dtype=torch.float32,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'o_proj', 'in_proj_qkv', 'out_proj'],
            gpu_memory_utilization=float(config.get('engine_gpu_memory_utilization', .85)),
            enable_prefix_caching=False, enforce_eager=True, max_num_seqs=784,
            max_num_batched_tokens=int(config['engine_token_budget']))
        backbone = self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        state = torch.load(self.checkpoint/'head.pt', map_location='cuda:0', weights_only=True)
        if state['weight'].dtype != torch.float32 or state['bias'].dtype != torch.float32:
            raise ValueError('Published decision head precision changed')
        self.model = torch.nn.Module()
        self.model.backbone = backbone
        self.model.head = torch.nn.Linear(state['weight'].shape[1], 1, dtype=torch.float32, device='cuda:0')
        self.model.head.load_state_dict(state)
        self.model.eval()
        self.lora = LoRARequest('frozen_author', 1, str(self.checkpoint/'adapter'))
        self.last_engine_forward_rows = []
        self.engine_forward_evidence = []
        self.adapter_verified = False
        backbone.register_forward_hook(self._completed_forward, with_kwargs=True)

    def _completed_forward(self, module, args, kwargs, result):
        import torch
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        values = list(metadata.values()) if isinstance(metadata, dict) else [metadata]
        rows = max((int(value.query_start_loc.numel()-1) for value in values
                    if getattr(value, 'query_start_loc', None) is not None), default=0)
        if rows <= 0:
            raise ValueError('Engine GPU sequence mapping unavailable')
        torch.cuda.synchronize()
        self.last_engine_forward_rows.append(rows)
        self.engine_forward_evidence.append(dict(sequence_rows=rows,
            input_tokens=int(kwargs['input_ids'].numel())))
        progress = getattr(self, '_step_progress', None)
        notify = getattr(progress, 'notify', None)
        if notify is not None:
            notify()

    def _compile(self, request):
        from jev.api import compile_request, candidate_prompts
        records = compile_request(**request)
        rendered = [self.tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for record in records for prompt in candidate_prompts(record)]
        paths = self.tokenizer(rendered, padding=False, truncation=False)['input_ids']
        if not paths or any(not path or len(path) > self.max_length for path in paths):
            raise ValueError('Complete native candidate path exceeds the actual model window')
        return records, paths

    def encode(self, request):
        _, paths = self._compile(request)
        return [dict(candidate_path_tokens=[len(path) for path in paths],
                     max_length=self.max_length, truncated=False)]

    def _verify_adapter(self):
        from jevlight.backends.vllm_lora import verify_adapter_coverage
        self.adapter_tensor_count = verify_adapter_coverage(self.model.backbone, self.checkpoint)
        self.adapter_verified = True

    def native(self, request):
        return self.native_batch([request], 1)[0]

    def native_batch(self, requests, batch_size):
        import torch
        from vllm import PoolingParams
        from jev.api import candidate_prompts, format_response
        from jev.metrics import softmax
        from jevlight.backends.decision_model_service import validate_answers
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Engine request batch size must be positive')
        compiled = [self._compile(request) for request in requests]
        paths = [dict(prompt_token_ids=path) for _, rows in compiled for path in rows]
        self.last_engine_forward_rows = []
        self.engine_forward_evidence = []
        if not paths:
            return []
        try:
            outputs = self.engine.encode(paths, PoolingParams(task='embed'), pooling_task='embed',
                                        lora_request=self.lora, use_tqdm=False)
        except Exception:
            # A failed forward leaves scheduled requests in the in-process engine.
            # Remove them before the capacity controller retries a smaller block.
            core = self.engine.llm_engine
            pending = list(core.output_processor.request_states)
            if pending:
                core.abort_request(pending, internal=True)
            raise
        if len(outputs) != len(paths) or any(output.prompt_token_ids != path['prompt_token_ids']
                for output, path in zip(outputs, paths, strict=True)):
            raise ValueError('Engine changed independent complete path mapping')
        if not self.adapter_verified:
            self._verify_adapter()
        with torch.inference_mode():
            hidden = torch.stack([output.outputs.data for output in outputs]).to('cuda:0').float()
            scores = self.model.head(hidden).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise ValueError('Published decision head returned nonfinite scores')
        scores = scores.cpu(); cursor = 0; results = []
        for records, _ in compiled:
            probabilities = []
            for record in records:
                count = len(candidate_prompts(record))
                values = scores[cursor:cursor+count]; cursor += count
                if record['kind'] == 'noul':
                    values = torch.stack([torch.zeros_like(values[0]), values[0]])
                probabilities.append(softmax(values.tolist(), temperature=self.temperature))
            results.append(format_response(records, probabilities))
        if cursor != len(paths):
            raise ValueError('Engine candidate and question reconstruction differs')
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]
