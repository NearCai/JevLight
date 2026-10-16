"""Reuse verified packed inputs while retaining Von's calibrated author decoder."""
from contextlib import contextmanager


@contextmanager
def reuse_native_inputs(runtime,texts,encoded,fit_keys):
    from transformers import BatchEncoding
    tokenizer=runtime.model.tokenizer
    fit=runtime.engine._fit_state
    lengths=encoded['attention_mask'].sum(-1).cpu().tolist()
    rows={text:BatchEncoding({key:value[i:i+1,:int(length)] for key,value in encoded.items()})
          for i,(text,length) in enumerate(zip(texts,lengths,strict=True))}
    class CachedTokenizer:
        def __getattr__(self,name):return getattr(tokenizer,name)
        def __call__(self,text,*args,**kwargs):
            if isinstance(text,str) and text in rows and not args and kwargs=={'return_tensors':'pt'}:
                return rows[text]
            return tokenizer(text,*args,**kwargs)
    def fitted(model,state,question,descriptions):
        if (state,question,tuple(descriptions)) in fit_keys:return state
        return fit(model,state,question,descriptions)
    try:
        runtime.model.tokenizer=CachedTokenizer()
        runtime.engine._fit_state=fitted
        yield
    finally:
        runtime.model.tokenizer=tokenizer
        runtime.engine._fit_state=fit
