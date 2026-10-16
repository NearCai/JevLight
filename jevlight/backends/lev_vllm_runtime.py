"""Original Lev routing, option-order averaging and calibration through vLLM."""
import json
from pathlib import Path
from jevlight.backends.pooled_native_runtime import PoolingNativeBase,ROOT


def native_codec(source):
    from jevlight.backends.decision_model_service import add_author_source
    from transformers import AutoTokenizer
    add_author_source('lev','packages/lev/src')
    from lev.model import DecisionEngine,EngineConfig
    from lev.calibrate import CalibrationProfile
    release=json.loads((source/'lev_release.json').read_text())
    engine=DecisionEngine(None,AutoTokenizer.from_pretrained(source,local_files_only=True),
        EngineConfig(model_id=release['base_model'],prompt_style=release['prompt_style'],compile=False),
        CalibrationProfile.load(source/'calibration.json'),mode_b_head=object())
    engine.checkpoint=source
    return engine


def verify_adapter(model,path):
    import torch
    from safetensors import safe_open
    config=json.loads((path/'adapter_config.json').read_text())
    if config.get('rank_pattern')or config.get('alpha_pattern')or config.get('trainable_token_indices'):
        raise ValueError('Native adapter needs independently reviewed nonuniform or embedding coverage')
    modules=dict(model.named_modules());scale=config['lora_alpha']/config['r'];count=0
    with safe_open(path/'adapter_model.safetensors',framework='pt',device='cpu')as archive:
        for name in archive.keys():
            stem,kind=name.removeprefix('base_model.model.').rsplit('.lora_',1)
            stem=stem.removeprefix('model.language_model.').removeprefix('model.')
            if not stem.startswith('layers.')or kind not in ('A.weight','B.weight'):
                raise ValueError(f'Unreviewed native adapter tensor: {name}')
            stem='model.'+stem;part=0;leaf=stem.rsplit('.',1)[-1]
            if leaf in ('q_proj','k_proj','v_proj'):
                part=('q_proj','k_proj','v_proj').index(leaf);stem=stem.rsplit('.',1)[0]+'.qkv_proj'
            elif leaf in ('gate_proj','up_proj'):
                part=('gate_proj','up_proj').index(leaf);stem=stem.rsplit('.',1)[0]+'.gate_up_proj'
            module=modules.get(stem)
            if module is None:raise ValueError(f'Released adapter module missing: {stem}')
            values=getattr(module,'lora_a_stacked'if kind=='A.weight'else'lora_b_stacked')
            actual=values[part][0,0].detach().cpu();expected=archive.get_tensor(name)
            if actual.dtype!=torch.float32 or expected.dtype!=torch.float32:
                raise ValueError('The published FP32 adapter precision changed')
            if kind=='B.weight':expected=expected*scale
            if actual.shape!=expected.shape or not torch.equal(actual,expected):
                raise ValueError(f'Incomplete released adapter tensor: {name}')
            count+=1
    return count


class LevPoolingNative(PoolingNativeBase):
    def __init__(self,config):
        import torch
        source=ROOT/config['weights_path'];self.codec=native_codec(source)
        self.codec_source=source
        manifest=json.loads((source.parent/'manifest.json').read_text())
        base=next(b for b in manifest['bases']if b['model_id']==self.codec.config.model_id)
        backbone=Path(base['weights_path'])
        from lev.readout.mode_a import LabelTokenReadout
        from lev.prompt import label_prefix,Style
        readout=LabelTokenReadout(self.codec.tokenizer,prefix=label_prefix(Style(self.codec.config.prompt_style)))
        self.labels=readout.candidate_ids(['A','B','C','D'])
        from safetensors import safe_open
        index=json.loads((backbone/'model.safetensors.index.json').read_text())['weight_map']
        name=next((n for n in ['lm_head.weight','model.language_model.embed_tokens.weight','model.embed_tokens.weight']if n in index),'')
        metadata=json.loads((backbone/'config.json').read_text())
        if name!='lm_head.weight'and not metadata.get('tie_word_embeddings',metadata.get('text_config',{}).get('tie_word_embeddings')):
            raise ValueError('A trained untied vocabulary head cannot be substituted')
        with safe_open(backbone/index[name],framework='pt',device='cpu')as archive:
            vocabulary=archive.get_tensor(name);weight=vocabulary[self.labels].clone().to(torch.bfloat16)
            self.vocabulary_size=vocabulary.shape[0]
        adapter=json.loads((source/'adapter_config.json').read_text())
        self.initialize_engine(config,backbone,task='embed',enable_lora=True,max_lora_rank=adapter['r'],max_loras=1,
            lora_dtype=torch.float32,lora_target_modules=['q_proj','k_proj','v_proj','qkv_proj','o_proj',
                                                        'gate_proj','up_proj','gate_up_proj','down_proj'])
        from vllm.lora.request import LoRARequest
        self.lora=LoRARequest('frozen_author',1,str(source));self.weight=weight.to('cuda');self.adapter_verified=False

    def _prepare(self,request):
        from lev.router import Mode
        from lev.readout.mode_a import LabelTokenReadout
        from lev.prompt import label_prefix,Style
        prepared=self.codec.prepare(**request)
        if any(q.type!='choice'for q in prepared.questions.values()):raise ValueError('Qualified phase-only protocol required')
        readout=LabelTokenReadout(self.codec.tokenizer,prefix=label_prefix(Style(self.codec.config.prompt_style)))
        for route in prepared.routes.values():
            if route.mode is not Mode.LABEL_TOKEN or any(t not in self.labels for t in readout.candidate_ids(route.codes)):
                raise ValueError('This native route needs an independently qualified head')
        rows=[prepared.prefix_ids+v.suffix_ids for v in prepared.variants]
        if any(len(ids)>self.max_length for ids in rows):raise ValueError('Complete native row exceeds the engine window')
        return prepared,rows

    def encode(self,request):
        _,rows=self._prepare(request)
        return [dict(encoded_tokens=len(ids),max_length=self.max_length,truncated=False)for ids in rows]

    def native_batch(self,requests,batch_size):
        import torch
        from jevlight.backends.decision_model_service import validate_answers
        packed=[self._prepare(r)for r in requests];rows=[ids for _,group in packed for ids in group]
        with torch.inference_mode():
            values=self.pool(rows,lora=self.lora)
            if not self.adapter_verified:
                self.released_adapter_tensors_verified=verify_adapter(self.model.backbone,self.codec_source);self.adapter_verified=True
            cursor=0;results=[]
            for request,(prepared,group)in zip(requests,packed,strict=True):
                logits=[]
                for ids in group:
                    h=torch.as_tensor(values[cursor].outputs.data).to('cuda',dtype=torch.bfloat16);cursor+=1
                    score=torch.nn.functional.linear(h,self.weight).float()
                    complete=torch.zeros(self.vocabulary_size,device='cuda',dtype=torch.float32);complete[self.labels]=score;logits.append(complete)
                result=self.codec._respond(prepared,torch.stack(logits),None).model_dump(mode='json')
                results.append(validate_answers(request,result))
        return results
