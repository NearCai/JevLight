"""Screen FLA's fused DeltaNet while retaining the native FP32 intermediates."""


def enable_fp32_delta(runtime):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    original=module.torch_chunk_gated_delta_rule
    runtime._native_native_delta=original
    runtime._native_delta_gpu_calls=0
    def delta(query,key,value,g,beta,chunk_size=64,initial_state=None,
              output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        if not query.is_cuda or query.shape[0]<=1 or kwargs.get('cu_seqlens') is not None:
            return original(query,key,value,g=g,beta=beta,chunk_size=chunk_size,
                initial_state=initial_state,output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        dtype=query.dtype
        q,k,v=[tensor.float().contiguous() for tensor in (query,key,value)]
        if use_qk_l2norm_in_kernel:
            q=module.l2norm(q,dim=-1,eps=1e-6)
            k=module.l2norm(k,dim=-1,eps=1e-6)
        # Apply scaling at the same stage as the native implementation. No
        # BF16 normalization, TF32 or reduced-precision recurrent state is used.
        q=q*(q.shape[-1]**-.5)
        out,state=chunk_gated_delta_rule(q,k,v,g.float().contiguous(),beta.float().contiguous(),
            scale=1.,initial_state=initial_state,output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,chunk_size=chunk_size)
        runtime._native_delta_gpu_calls+=1
        return out.to(dtype),state
    module.torch_chunk_gated_delta_rule=delta
