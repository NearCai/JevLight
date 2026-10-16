"""Original shared-state compiler and FP32 Mixer head with GPU pooling."""
import importlib
import json
import sys
from types import ModuleType
from jevlight.backends.pooled_native_runtime import PoolingNativeBase,ROOT


def native_module(source,name):
    package='_jevlight_frozen_valen'
    if package not in sys.modules:
        module=ModuleType(package);module.__path__=[str(source)];sys.modules[package]=module
    return importlib.import_module(package+'.'+name)


class TextProcessor:
    """The frozen processor's text-only chat-template path; no media is admitted."""
    def __init__(self,tokenizer):self.tokenizer=tokenizer

    def apply_chat_template(self,messages,**kwargs):
        for message in messages:
            if not isinstance(message['content'],str)and any(part['type']!='text'for part in message['content']):
                raise ValueError('The qualified benchmark uses complete text-only inputs')
        kwargs.pop('processor_kwargs',None)
        return self.tokenizer.apply_chat_template(messages,**kwargs)


def native_compiler(config):
    from transformers import AutoTokenizer,PreTrainedTokenizerFast
    source=ROOT/config['weights_path'];published=json.loads((source/'config.json').read_text())
    tokenizer_config=json.loads((source/'tokenizer_config.json').read_text())
    tokenizer=(PreTrainedTokenizerFast.from_pretrained(source,local_files_only=True)
               if tokenizer_config.get('tokenizer_class')=='TokenizersBackend'
               else AutoTokenizer.from_pretrained(source,local_files_only=True))
    processor=TextProcessor(tokenizer)
    compiler=native_module(source,'compiler').Compiler(processor,max_length=config['max_length'],
        media_kwargs=published.get('valen_media_kwargs',{}),execution='shared_state')
    return compiler


def compile_request(compiler,request):
    state=compiler.compile(dict(request=dict(state=json.dumps(request['state'],ensure_ascii=False),questions=request['questions'])))
    if state.inputs is None or state.media:raise ValueError('Complete native text-only shared-state encoding required')
    for question in state.questions:
        if question.kind!='choice'or question.keys!=list(request['questions'][question.qid]['criteria'])or len(question.branches)!=1:
            raise ValueError('Native phase question/candidate mapping differs')
    return state


class ValenPoolingNative(PoolingNativeBase):
    def __init__(self,config):
        import torch
        from safetensors import safe_open
        source=ROOT/config['weights_path'];published=json.loads((source/'config.json').read_text())
        self.compiler=native_compiler(config);self.responses=native_module(source,'responses')
        head=native_module(source,'heads').build_head(published['text_config']['hidden_size'],published['valen_head']).float().eval()
        index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map'];state={}
        for shard in sorted({v for n,v in index.items()if n.startswith('head.')}):
            with safe_open(source/shard,framework='pt',device='cpu')as archive:
                for name in archive.keys():
                    if name.startswith('head.'):
                        value=archive.get_tensor(name)
                        if value.dtype!=torch.float32:raise ValueError('The released Mixer precision changed')
                        state[name.removeprefix('head.')]=value
        head.load_state_dict(state,strict=True);self.released_head_tensors_verified=len(state)
        self.initialize_engine(config,source,architecture='ValenMarkerEngine')
        self.model.head=head.to('cuda').eval()

    def _compile(self,request):
        state=compile_request(self.compiler,request)
        if state.logical_tokens>self.max_length:raise ValueError('Complete native input exceeds the engine window')
        return state

    def encode(self,request):
        state=self._compile(request)
        return [dict(logical_tokens=state.logical_tokens,compute_tokens=state.compute_tokens,
                     max_length=self.max_length,truncated=False)]

    def native_batch(self,requests,batch_size):
        import torch
        from jevlight.backends.decision_model_service import validate_answers
        states=[self._compile(r)for r in requests];rows=[s.inputs['input_ids'][0].tolist()for s in states]
        with torch.inference_mode():
            values=self.pool(rows);results=[]
            for request,state,ids,value in zip(requests,states,rows,values,strict=True):
                hidden=torch.as_tensor(value.outputs.data)
                if hidden.ndim!=2 or hidden.shape[0]!=len(ids):raise ValueError('The engine omitted native token states')
                answers={}
                for question in state.questions:
                    features=self.model.head.extract_features(hidden.unsqueeze(0),question.branches[0]).to('cuda')
                    logits=self.model.head.score_features(features)
                    answers[question.qid]=self.responses.answer(question,logits,1.)
                result=dict(model=self.config['model_id'],answers=answers,
                    usage=dict(input_tokens=state.logical_tokens,output_tokens=0),
                    internal_usage=dict(compute_tokens=state.compute_tokens))
                results.append(validate_answers(request,result))
        return results
