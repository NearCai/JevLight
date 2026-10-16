"""Stream native FP32 relative attention with preprojected positional rows."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q,K,V,C2P,P2C,Positions,Mask,Out,L,Scale,
               HEADS:tl.constexpr,D:tl.constexpr,P:tl.constexpr,
               BQ:tl.constexpr,BK:tl.constexpr):
    bh=tl.program_id(1).to(tl.int64)
    b=bh//HEADS
    qi=tl.program_id(0)*BQ+tl.arange(0,BQ)
    d=tl.arange(0,D)
    q=tl.load(Q+bh*L*D+qi[:,None]*D+d[None,:],qi[:,None]<L,0)
    query_valid=tl.load(Mask+b*L+qi,qi<L,0).to(tl.int1)
    maximum=tl.full((BQ,),float('-inf'),tl.float32)
    denominator=tl.zeros((BQ,),tl.float32)
    accumulator=tl.zeros((BQ,D),tl.float32)
    for start in range(0,tl.cdiv(L,BK)):
        ki=start*BK+tl.arange(0,BK)
        key=tl.load(K+bh*L*D+ki[None,:]*D+d[:,None],ki[None,:]<L,0)
        value=tl.load(V+bh*L*D+ki[:,None]*D+d[None,:],ki[:,None]<L,0)
        valid=(qi[:,None]<L)&(ki[None,:]<L)
        indices=tl.load(Positions+qi[:,None]-ki[None,:]+L-1,valid,0)
        c2p=tl.load(C2P+bh*L*P+qi[:,None]*P+indices,valid,0)/Scale
        p2c=tl.load(P2C+bh*L*P+ki[None,:]*P+indices,valid,0)/Scale
        scores=tl.dot(q,key,input_precision='ieee')/Scale+c2p+p2c
        key_valid=tl.load(Mask+b*L+ki,ki<L,0).to(tl.int1)
        scores=tl.where(query_valid[:,None]&key_valid[None,:],scores,-3.4028234663852886e38)
        scores=tl.where(ki[None,:]<L,scores,float('-inf'))
        new_maximum=tl.maximum(maximum,tl.max(scores,axis=1))
        alpha=tl.exp(maximum-new_maximum)
        probability=tl.exp(scores-new_maximum[:,None])
        accumulator=accumulator*alpha[:,None]+tl.dot(probability,value,input_precision='ieee')
        denominator=denominator*alpha+tl.sum(probability,axis=1)
        maximum=new_maximum
    result=accumulator/denominator[:,None]
    tl.store(Out+bh*L*D+qi[:,None]*D+d[None,:],result,qi[:,None]<L)


def flash_forward(self,hidden_states,attention_mask,output_attentions=False,
                  query_states=None,relative_pos=None,rel_embeddings=None):
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position
    if (self.training or output_attentions or query_states is not None or
            hidden_states.dtype!=torch.float32 or not self.share_att_key or
            set(self.pos_att_type)!={'c2p','p2c'}):
        from jevlight.backends.deberta_attention import _forward
        return _forward(self,hidden_states,attention_mask,output_attentions,
                        query_states,relative_pos,rel_embeddings)
    batch,length,_=hidden_states.shape;heads=self.num_attention_heads
    query=self.transpose_for_scores(self.query_proj(hidden_states),heads).contiguous()
    key=self.transpose_for_scores(self.key_proj(hidden_states),heads).contiguous()
    value=self.transpose_for_scores(self.value_proj(hidden_states),heads).contiguous()
    span=self.pos_ebd_size
    embeddings=self.pos_dropout(rel_embeddings)[:2*span].unsqueeze(0)
    positional_query=self.transpose_for_scores(self.query_proj(embeddings),heads).contiguous()
    positional_key=self.transpose_for_scores(self.key_proj(embeddings),heads).contiguous()
    # cuBLAS projects each positional row once. Keeping these native FP32
    # tensors trades measured capacity for fewer registers/spills in attention.
    c2p=torch.matmul(query.view(batch,heads,length,-1),positional_key.transpose(-1,-2)[None]).contiguous()
    p2c=torch.matmul(key.view(batch,heads,length,-1),positional_query.transpose(-1,-2)[None]).contiguous()
    distances=torch.arange(1-length,length,device=query.device)
    if self.position_buckets>0:
        distances=make_log_bucket_position(distances,self.position_buckets,self.max_relative_positions)
    positions=(distances.long()+span).clamp(0,2*span-1)
    if attention_mask.ndim!=2:
        raise ValueError('Fused DeBERTa requires the complete native 2D token mask')
    output=torch.empty_like(query);scale=math.sqrt(query.shape[-1]*3)
    _attention[(triton.cdiv(length,32),batch*heads)](query,key,value,c2p,p2c,positions,
        attention_mask.contiguous(),output,length,scale,HEADS=heads,D=query.shape[-1],P=2*span,
        BQ=32,BK=64,num_warps=4,num_stages=1)
    context=output.view(batch,heads,length,-1).permute(0,2,1,3).contiguous()
    return context.view(batch,length,-1),None
