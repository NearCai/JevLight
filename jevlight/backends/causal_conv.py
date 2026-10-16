"""Batch-only depthwise causal convolution with native FP32 accumulation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv(X,W,Bias,Y,L,C,SB,SC,ST,N,
          WIDTH:tl.constexpr,HAS_BIAS:tl.constexpr,SILU:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    valid=index<N
    b=index//(C*L);c=(index//L)%C;t=index%L
    value=tl.zeros((BLOCK,),tl.float32)
    for j in tl.static_range(WIDTH):
        source=t-WIDTH+1+j
        x=tl.load(X+b*SB+c*SC+source*ST,valid&(source>=0),0).to(W.dtype.element_ty).to(tl.float32)
        weight=tl.load(W+c*WIDTH+j,valid,0).to(tl.float32)
        value+=x*weight
    if HAS_BIAS:value+=tl.load(Bias+c,valid,0).to(tl.float32)
    # cuDNN returns a convolution rounded to the weight dtype before SiLU.
    value=value.to(W.dtype.element_ty).to(tl.float32)
    if SILU:value=(value*tl.sigmoid(value)).to(W.dtype.element_ty)
    tl.store(Y+index,value,valid)


def batch_causal_conv(hidden_states,weight,bias=None,activation=None,**kwargs):
    if activation not in (None,'silu'):
        raise ValueError('Unsupported native causal-convolution activation')
    b,c,length=hidden_states.shape
    if bias is not None:
        # cuDNN's bias rounding differs from adding it to the unrounded FP32
        # accumulator. Keep that native path; published Qwen3.5 has no bias.
        output=torch.nn.functional.conv1d(hidden_states.to(weight.dtype),weight[:,None],bias,
            padding=weight.shape[-1]-1,groups=c)[:,:,:length]
    else:
        output=torch.empty(hidden_states.shape,dtype=weight.dtype,device=hidden_states.device)
        _conv[(triton.cdiv(b*c*length,256),)](hidden_states,weight,bias,output,length,c,*hidden_states.stride(),b*c*length,
            WIDTH=weight.shape[-1],HAS_BIAS=False,SILU=False,BLOCK=256,enable_fp_fusion=False)
    # Keep ATen's original activation and its rounding, rather than substituting
    # a fused sigmoid approximation near low-precision rounding boundaries.
    if activation=='silu':output=torch.nn.functional.silu(output)
    return output.to(hidden_states.dtype)


def enable_batch_conv(runtime):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    original=module.causal_conv1d_fn
    def convolution(hidden_states,weight,bias=None,activation=None,**kwargs):
        if (hidden_states.is_cuda and hidden_states.shape[0]>1 and weight.is_contiguous() and bias is None
                and weight.dtype==torch.bfloat16 and hidden_states.dtype==torch.bfloat16
                and activation in (None,'silu') and not any(value is not None for value in kwargs.values())):
            return batch_causal_conv(hidden_states,weight,bias,activation)
        return original(hidden_states,weight,bias,activation=activation,**kwargs)
    runtime._native_native_conv=original
    module.causal_conv1d_fn=convolution
