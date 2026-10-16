"""Frozen full Kev backbone and native FP32 pointer/calibration through vLLM."""
import json
import ast
from types import ModuleType
from jevlight.backends.pooled_native_runtime import PoolingNativeBase,ROOT


def load_primitives():
    path=ROOT/'.local/model_sources/kev/kev/model.py'
    tree=ast.parse(path.read_text())
    names={'ContextOverflow','PointerHead','user_tokens','encode','rows_of'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            module=getattr(node,'module','')or''
            if not module.startswith('transformers'):nodes.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            # The frozen encoder's delimiter/context constants and regex only.
            if node.lineno<87:nodes.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in names:
            nodes.append(node)
    module=ModuleType('jevlight_frozen_kev_primitives')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module


class KevPoolingNative(PoolingNativeBase):
    def __init__(self,config):
        import torch
        from transformers import AutoTokenizer,PreTrainedTokenizerFast
        from jevlight.backends.decision_model_service import add_author_source
        source=ROOT/config['weights_path']
        if not(source/'config.json').exists():raise ValueError('An unmerged adapter needs separately qualified complete LoRA coverage')
        add_author_source('kev')
        self.primitives=load_primitives()
        metadata=torch.load(source/'head.pt',map_location='cpu',weights_only=True)
        if metadata.get('option_isolation',False):raise ValueError('Native isolated option positions need their own engine adapter')
        tokenizer_config=json.loads((source/'tokenizer_config.json').read_text())
        self.tokenizer=(PreTrainedTokenizerFast.from_pretrained(source,local_files_only=True)
                        if tokenizer_config.get('tokenizer_class')=='TokenizersBackend'
                        else AutoTokenizer.from_pretrained(source,local_files_only=True))
        weights=metadata['head']
        if any(v.dtype!=torch.float32 for v in weights.values()):raise ValueError('Published pointer head precision differs')
        head=self.primitives.PointerHead(weights['q.weight'].shape[1],weights['q.weight'].shape[0]);head.load_state_dict(weights,strict=True)
        head.temperature=metadata.get('temperature',1.)
        if head.temperature<=0:raise ValueError('Native calibration is invalid')
        self.max_state=config['max_state_tokens'];self.released_head_tensors_verified=len(weights)
        self.initialize_engine(config,source);self.model.head=head.to('cuda',dtype=torch.float32).eval()

    def _compile(self,request):
        from kev.api import SystemOneRequest,to_record
        encode,rows_of=self.primitives.encode,self.primitives.rows_of
        if any(q['type']!='choice'for q in request['questions'].values()):raise ValueError('Qualified phase-only protocol required')
        record,meta=to_record(SystemOneRequest(**request))
        encoded=encode(self.tokenizer,record,max_state=self.max_state,max_branch=self.max_length,strict=True,option_isolation=False)
        full=encode(self.tokenizer,record,max_state=1_000_000,max_branch=1_000_000,strict=True,option_isolation=False)
        if encoded['ids']!=full['ids']:raise ValueError('Native complete pointer encoding would truncate')
        prefix,positions,branches=rows_of(encoded)
        rows=[dict(ids=prefix+r['ids'],decide=len(prefix)+r['decide'],options=[len(prefix)+i for i in r['opts']])for r in branches]
        if len(rows)!=len(meta):raise ValueError('Native question-row mapping differs')
        for row,entry in zip(rows,meta,strict=True):
            if len(row['ids'])>self.max_length or len(row['options'])!=len(request['questions'][entry['id']]['criteria']):
                raise ValueError('Complete native branch/candidate mapping exceeds the engine contract')
        return encoded,meta,rows

    def encode(self,request):
        encoded,_,rows=self._compile(request)
        return [dict(encoded_tokens=max(len(r['ids'])for r in rows),state_tokens=encoded['state_tokens'],
                     max_length=self.max_length,max_state_tokens=self.max_state,truncated=False)]

    def native_batch(self,requests,batch_size):
        import torch
        from kev.api import to_answers
        from jevlight.backends.decision_model_service import validate_answers
        packed=[self._compile(r)for r in requests];rows=[row for _,_,group in packed for row in group]
        with torch.inference_mode():
            values=self.pool([r['ids']for r in rows]);results=[];cursor=0
            for request,(encoded,meta,group)in zip(requests,packed,strict=True):
                probabilities=[]
                for row in group:
                    h=torch.as_tensor(values[cursor].outputs.data);cursor+=1
                    if h.ndim!=2 or h.shape[0]!=len(row['ids']):raise ValueError('Engine omitted native token states')
                    positions=[row['decide'],*row['options']];selected=h[positions].to('cuda',dtype=torch.float32)
                    logits=self.model.head(selected[0],selected[1:])
                    if not torch.isfinite(logits).all():raise ValueError('Native head returned nonfinite logits')
                    probabilities.append(logits.softmax(-1).cpu())
                result=dict(answers=to_answers(probabilities,meta),usage=dict(input_tokens=len(encoded['ids']),output_tokens=0))
                results.append(validate_answers(request,result))
        return results
