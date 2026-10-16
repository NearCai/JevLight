"""Run padded causal batches through PyTorch's native variable-length Flash kernel."""
import numpy as np
import torch


def _metadata(mask):
    valid=mask.detach().cpu().numpy().astype(bool)
    batch,length=valid.shape
    lengths=valid.sum(-1).tolist()
    if not all(lengths):raise ValueError('Native causal batch contains an empty token sequence')
    indices=np.concatenate([np.flatnonzero(row)+b*length for b,row in enumerate(valid)])
    return dict(indices=torch.from_numpy(indices).to(mask.device),
        cumulative=torch.tensor([0,*np.cumsum(lengths).tolist()],dtype=torch.int32,device=mask.device),
        maximum=max(lengths),batch=batch,length=length)


def _pack(tensor,meta):
    batch,heads,length,width=tensor.shape
    return tensor.transpose(1,2).reshape(batch*length,heads,width).index_select(0,meta['indices'])


def enable_native_flash(model,pack_single=None):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    modules=[module for module in model.modules() if type(module).__name__=='Qwen3_5Attention']
    if not modules:raise ValueError('No compatible native Qwen3.5 attention layers')
    if any(getattr(module,'sliding_window',None) for module in modules):
        raise ValueError('Native variable-length adapter requires full causal attention')
    def before(module,args,kwargs):
        for attention in modules:attention._native_varlen=None
        if getattr(model,"_native_dense_prefix",False):return
        cache=kwargs.get('past_key_values')
        # Preserve the original masked rectangular kernel for cached branches.
        if cache is not None and cache.get_seq_length()>0:return
        mask=kwargs.get('attention_mask')
        if isinstance(mask,dict):mask=mask.get('linear_attention')
        if not torch.is_tensor(mask) or mask.ndim!=2:return
        if mask.shape[0]<=1 and not (pack_single is not None and pack_single()):return
        meta=_metadata(mask)
        for attention in modules:attention._native_varlen=meta
    model.register_forward_pre_hook(before,with_kwargs=True)
    def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        meta=getattr(module,'_native_varlen',None)
        if (meta is None or query.shape[2]!=key.shape[2] or query.dtype not in (torch.bfloat16,torch.float16)
                or dropout or kwargs.get('output_attentions')):
            return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        q,k,v=[_pack(tensor,meta) for tensor in (query,key,value)]
        causal=kwargs.get('is_causal',getattr(module,'is_causal',True))
        output=torch.ops.aten._flash_attention_forward(q,k,v,meta['cumulative'],meta['cumulative'],
            meta['maximum'],meta['maximum'],0.,causal,False,scale=scaling)[0]
        padded=output.new_zeros((meta['batch']*meta['length'],query.shape[1],query.shape[-1]))
        padded.index_copy_(0,meta['indices'],output)
        return padded.view(meta['batch'],meta['length'],query.shape[1],query.shape[-1]),None
    # Process-local, marked modules only. Scalar author calls keep their exact
    # original SDPA implementation; unrelated attention modules are unchanged.
    ALL_ATTENTION_FUNCTIONS.register('sdpa',attention)
    return len(modules)
