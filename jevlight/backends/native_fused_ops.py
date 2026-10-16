"""Batch-only native BF16 activation/product and convolution fusion."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from types import MethodType
import threading


@triton.jit
def native_div(a,b):
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def native_mul(a,b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def silu_product(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32);u=tl.load(U+i,mask,0).to(tl.float32)
    activation=native_div(g,1.+libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,native_mul(activation,u),mask)

def fused(g,u):
    if g.dtype!=torch.bfloat16 or u.dtype!=torch.bfloat16 or not g.is_contiguous() or not u.is_contiguous() or g.shape!=u.shape:raise ValueError('Fusion requires complete equal contiguous BF16 tensors')
    y=torch.empty_like(g)
    silu_product[(triton.cdiv(g.numel(),256),)](g,u,y,g.numel(),BLOCK=256,enable_fp_fusion=False)
    return y

@triton.jit
def fused_conv(X,W,Y,L,C,SB,SC,ST,N,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    b=i//(C*L);c=(i//L)%C;t=i%L;value=tl.zeros((BLOCK,),tl.float32)
    for j in tl.static_range(WIDTH):
        source=t-WIDTH+1+j
        x=tl.load(X+b*SB+c*SC+source*ST,mask&(source>=0),0).to(tl.float32)
        weight=tl.load(W+c*WIDTH+j,mask,0).to(tl.float32)
        value+=x*weight
    value=value.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,native_div(value,1.+libdevice.exp(-value)),mask)


def enable_native_fused_ops(runtime, family):
    """Preserve author scalar/media paths and restore all scopes after failure."""
    if hasattr(runtime, '_step_fused_ops_batch'):
        return
    if family == 'lev':
        backbone = runtime.engine._decoder
    elif family == 'clef':
        backbone = runtime.model.encoder
    else:
        raise ValueError('Native fused operations support Lev and Clef text batches')
    context = threading.local()
    mlps = [module for module in backbone.modules() if type(module).__name__ == 'Qwen3_5MLP']
    if not mlps or any(type(m.act_fn).__name__ not in ('SiLU', 'SiLUActivation', 'SiluActivation') for m in mlps):
        raise ValueError('Native fused MLP requires the original Qwen3.5 SiLU operation')
    for module in mlps:
        module._step_fused_mlp_forward = module.forward
        def mlp(self, inputs):
            if (not getattr(context, 'active', False) or not inputs.is_cuda
                    or inputs.dtype != torch.bfloat16):
                return self._step_fused_mlp_forward(inputs)
            return self.down_proj(fused(self.gate_proj(inputs), self.up_proj(inputs)))
        module.forward = MethodType(mlp, module)
    runtime._step_fused_ops_batch = runtime.native_batch

    def batch(self, requests, batch_size):
        from jevlight.backends import causal_conv
        previous = getattr(context, 'active', False)
        original_conv = causal_conv.batch_causal_conv
        context.active = len(requests) > 1 and not any(r.get('images') or r.get('videos') for r in requests)
        def convolution(hidden_states, weight, bias=None, activation=None, **kwargs):
            if (not context.active or hidden_states.ndim != 3 or hidden_states.shape[0] <= 1
                    or bias is not None or hidden_states.dtype != torch.bfloat16
                    or weight.dtype != torch.bfloat16 or activation != 'silu'
                    or any(value is not None for value in kwargs.values())):
                return original_conv(hidden_states, weight, bias, activation=activation, **kwargs)
            output = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
            b, channels, length = hidden_states.shape
            fused_conv[(triton.cdiv(hidden_states.numel(), 256),)](
                hidden_states, weight, output, length, channels, *hidden_states.stride(),
                hidden_states.numel(), WIDTH=weight.shape[-1], BLOCK=256, enable_fp_fusion=False)
            return output
        try:
            causal_conv.batch_causal_conv = convolution
            return self._step_fused_ops_batch(requests, batch_size)
        finally:
            context.active = previous
            causal_conv.batch_causal_conv = original_conv
    runtime.native_batch = MethodType(batch, runtime)
