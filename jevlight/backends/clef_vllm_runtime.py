"""Frozen Clef compiler, joint schema head and lexical embeddings via vLLM."""
import json
from jevlight.backends.pooled_native_runtime import PoolingNativeBase,ROOT


def native_parts(config):
    from jevlight.backends.decision_model_service import author_file
    from transformers import AutoTokenizer,PreTrainedTokenizerFast
    source=ROOT/config['weights_path']
    module=author_file(source/'joint_schema_model.py','jevlight_frozen_clef_pooling')
    metadata=json.loads((source/'tokenizer_config.json').read_text())
    tokenizer=(PreTrainedTokenizerFast.from_pretrained(source,local_files_only=True)
               if metadata.get('tokenizer_class')=='TokenizersBackend'
               else AutoTokenizer.from_pretrained(source,local_files_only=True))
    return module,tokenizer


class ClefPoolingNative(PoolingNativeBase):
    def __init__(self,config):
        import torch
        from safetensors.torch import load_file
        from safetensors import safe_open
        source=ROOT/config['weights_path'];self.module,self.tokenizer=native_parts(config)
        head=self.module.JointSchemaHead(**json.loads((source/'joint_head_config.json').read_text()))
        state=load_file(str(source/'joint_head.safetensors'));head.load_state_dict(state,strict=True)
        self.released_head_tensors_verified=len(state)
        index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
        if 'lm_head.weight'not in index:raise ValueError('The original lexical embedding projection is missing')
        with safe_open(source/index['lm_head.weight'],framework='pt',device='cpu')as archive:
            vocabulary=archive.get_tensor('lm_head.weight')
        self.initialize_engine(config,source)
        self.model.head=head.to('cuda',dtype=torch.bfloat16).eval()
        self.output_embedding=vocabulary.to('cuda',dtype=torch.bfloat16)

    def _compile(self,request):
        if any(q['type']!='choice'for q in request['questions'].values()):raise ValueError('Qualified phase-only native protocol required')
        record=dict(request,model=self.config['model_id'])
        actual=self.module.encode_record(self.tokenizer,record,max_length=self.max_length)
        unlimited=self.module.encode_record(self.tokenizer,record,max_length=1_000_000)
        if actual.input_ids!=unlimited.input_ids or actual.media:
            raise ValueError('Native complete text encoding would truncate or introduce media')
        if len(actual.questions)!=len(request['questions']):raise ValueError('Native question mapping differs')
        for q in actual.questions:
            if q.option_ids!=tuple(sorted(str(k)for k in request['questions'][q.question_id]['criteria'])):
                raise ValueError('Native candidate order differs')
        return actual

    def encode(self,request):
        record=self._compile(request)
        return [dict(encoded_tokens=len(record.input_ids),max_length=self.max_length,truncated=False)]

    def native_batch(self,requests,batch_size):
        import torch
        from jevlight.backends.decision_model_service import validate_answers
        records=[self._compile(r)for r in requests];rows=[list(r.input_ids)for r in records]
        with torch.inference_mode():
            values=self.pool(rows);results=[]
            for request,record,ids,value in zip(requests,records,rows,values,strict=True):
                hidden=torch.as_tensor(value.outputs.data).to('cuda',dtype=torch.bfloat16)
                if hidden.ndim!=2 or hidden.shape[0]!=len(ids):raise ValueError('The engine omitted native token states')
                input_ids=torch.tensor([ids],device='cuda')
                logits=self.model.head(hidden.unsqueeze(0),input_ids,
                    torch.ones_like(input_ids),[record],self.output_embedding)[0]
                answers={}
                for question,row in zip(record.questions,logits,strict=True):
                    if not torch.isfinite(row).all():raise ValueError('Native joint head returned nonfinite logits')
                    probabilities=dict(zip(question.option_ids,row.float().softmax(-1).cpu().tolist(),strict=True))
                    answers[question.question_id]=self.module.systemone_answer(request['questions'][question.question_id],probabilities)
                result=dict(model=self.config['model_id'],answers=answers,
                    usage=dict(input_tokens=len(ids),output_tokens=0))
                results.append(validate_answers(request,result))
        return results
