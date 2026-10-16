"""Author-supported merged BF16 backbone with unchanged Open-Jev FP32 head."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from jevlight.backends.vllm_decision_runtime import OpenJevPoolingNative


class MergedOpenJevPoolingNative(OpenJevPoolingNative):
    def __init__(self, config):
        import torch
        from jevlight.backends.decision_model_service import ROOT, add_author_source
        from jevlight.backends.cuda_runtime import prepare_cuda_device
        prepare_cuda_device()
        add_author_source('open-jev')
        self.checkpoint = ROOT/config['weights_path']/'package/checkpoint'
        source = ROOT/config['engine_weights_path']
        manifest = json.loads((source/'engine_export_manifest.json').read_text())
        if manifest['status'] != 'ready' or Path(manifest['checkpoint']) != self.checkpoint:
            raise ValueError('The merged export does not match the frozen native checkpoint')
        for name, evidence in manifest['files'].items():
            stat = (source/name).stat()
            if stat.st_size != evidence['size'] or stat.st_mtime_ns != evidence['mtime_ns']:
                raise ValueError(f'Merged export weight identity changed: {name}')
        for name, expected in [('adapter/adapter_model.safetensors', 'source_adapter_sha256'),
                               ('head.pt', 'source_head_sha256'), ('temperature.json', 'source_temperature_sha256')]:
            with (self.checkpoint/name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest[expected]:
                    raise ValueError(f'Frozen native source changed: {name}')
        if manifest['adapter_source_dtype'] != 'float32' or manifest['merged_backbone_dtype'] != 'bfloat16':
            raise ValueError('The reviewed author merge precision differs')
        self.adapter_tensor_count = manifest['released_adapter_tensors_verified']
        if self.adapter_tensor_count != 2*len(manifest['merged_targets']):
            raise ValueError('Not every frozen adapter target was merged')
        self.adapter_verified = True
        self.temperature = json.loads((self.checkpoint/'temperature.json').read_text())['temperature']
        self.max_length = int(config.get('engine_max_length', config['max_length']))
        if self.max_length > config['max_length']:
            raise ValueError('Engine window exceeds verified model capability')
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        from vllm import LLM, ModelRegistry
        from vllm.config import PoolerConfig
        from transformers import AutoTokenizer
        ModelRegistry.register_model('OpenJevEngineForEmbedding',
                                    'jevlight.backends.vllm_decision_models:OpenJevEngineForEmbedding')
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.engine = LLM(model=str(source), runner='pooling', convert='embed',
            hf_overrides={'architectures': ['OpenJevEngineForEmbedding']},
            pooler_config=PoolerConfig(seq_pooling_type='LAST', use_activation=False),
            skip_tokenizer_init=True, dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=self.max_length, max_num_seqs=784,
            max_num_batched_tokens=int(config['engine_token_budget']),
            gpu_memory_utilization=float(config['engine_gpu_memory_utilization']),
            enable_prefix_caching=False,
            enforce_eager=bool(config.get('engine_enforce_eager', True)))
        state = torch.load(self.checkpoint/'head.pt', map_location='cuda:0', weights_only=True)
        if state['weight'].dtype != torch.float32 or state['bias'].dtype != torch.float32:
            raise ValueError('The published decision head precision changed')
        self.model = torch.nn.Module()
        self.model.backbone = self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        self._verify_merged_targets(source, manifest)
        self.model.head = torch.nn.Linear(state['weight'].shape[1], 1, device='cuda:0', dtype=torch.float32)
        self.model.head.load_state_dict(state); self.model.eval()
        self.lora = None
        self.last_engine_forward_rows = []
        self.engine_forward_evidence = []
        self.model.backbone.register_forward_hook(self._completed_forward, with_kwargs=True)

    def _verify_merged_targets(self, source, manifest):
        import torch
        from safetensors import safe_open
        index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
        parameters = dict(self.model.backbone.named_parameters())
        checked = 0
        for shard in sorted({index[name] for name in manifest['merged_targets']}):
            with safe_open(source/shard, framework='pt', device='cpu') as archive:
                for name in manifest['merged_targets']:
                    if index[name] != shard:
                        continue
                    expected = archive.get_tensor(name)
                    target = name.replace('model.language_model.', 'model.', 1)
                    leaf = target.rsplit('.', 2)[-2]
                    offset = 0
                    if leaf in ('q_proj', 'k_proj', 'v_proj'):
                        prefix = name.rsplit('.', 2)[0]
                        for prior in ('q_proj', 'k_proj', 'v_proj')[:('q_proj', 'k_proj', 'v_proj').index(leaf)]:
                            original = prefix+'.'+prior+'.weight'
                            with safe_open(source/index[original], framework='pt', device='cpu') as other:
                                offset += other.get_slice(original).get_shape()[0]
                        target = target.rsplit('.', 2)[0]+'.qkv_proj.weight'
                    elif leaf == 'in_proj_qkv':
                        target = target.replace('.in_proj_qkv.weight', '.in_proj_qkvz.weight')
                    if target not in parameters:
                        raise ValueError(f'Merged native target was not loaded: {name}')
                    actual = parameters[target].detach()[offset:offset+expected.shape[0]].cpu()
                    if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                        raise ValueError(f'Actual GPU merged target differs from author export: {name}')
                    checked += 1
        if checked != len(manifest['merged_targets']):
            raise ValueError('Actual GPU merge coverage is incomplete')
        self.verified_merged_target_count = checked
