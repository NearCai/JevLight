"""Author CUDA state/branch passes sized by measured whole-step capacity."""
import gc
from contextlib import contextmanager
from types import MethodType


@contextmanager
def _limits(module,settings):
    previous={key:getattr(module,key) for key in settings}
    try:
        for key,value in settings.items():setattr(module,key,value)
        yield
    finally:
        for key,value in previous.items():setattr(module,key,value)


def native_prefix_batch(runtime,requests):
    import torch
    from kev import cuda_graphs as author
    from kev.model import rows_of
    from kev.api import to_answers
    packed=[runtime.encode_native(request) for request in requests]
    splits=[rows_of(encoded) for encoded,_ in packed]
    states=max(len(state) for state,_,_ in splits)
    branches=max(len(row['ids']) for _,_,rows in splits for row in rows)
    row_count=sum(len(rows) for _,_,rows in splits)
    bank_width=author.pow2(max(64 if getattr(runtime,'_native_align_prefix_states',False) else 16,states))
    needed=(len(splits),row_count,bank_width,branches)
    cache=getattr(runtime,'_native_prefix_engine',None)
    if cache is not None and any(a>b for a,b in zip(needed,runtime._native_prefix_shape,strict=True)):
        runtime._native_prefix_engine=None
        del cache
        gc.collect();torch.cuda.empty_cache()
    settings=dict(GRAPH_STATES=needed[0],GRAPH_ROWS=needed[1],BANK_WIDTH=bank_width,
        GRAPH_STATE=states,GRAPH_ROW=branches,GRAPH_TOKENS=row_count*(bank_width+branches),
        bucket=lambda n,steps=8,floor=16:n,
        length_groups=lambda lengths,cap:[list(range(len(lengths)))])
    if getattr(runtime,'_native_prefix_engine',None) is None:
        with _limits(author,settings):
            engine=author.CudaGraphs(runtime.model.lm,runtime.model.pad_id)
        if getattr(runtime,'_native_align_prefix_states',False):
            from jevlight.backends.kev_aligned_prefix import install_aligned_states
            install_aligned_states(engine,runtime)
        runtime._native_prefix_engine=engine
        runtime._native_prefix_shape=needed
        runtime._native_prefix_settings=settings
    engine=runtime._native_prefix_engine
    group=[author.Request(state,positions,[(row['ids'],row['pos']) for row in rows],None,False,
                         [[row['decide'],*row['opts']] for row in rows]) for state,positions,rows in splits]
    with torch.inference_mode(),_limits(author,runtime._native_prefix_settings):
        hidden,_=engine.run(group)
        # Retain the original FP32 pointer readout and native API decoder.
        probabilities=iter(runtime.model._readout_many(hidden,[len(row['opts']) for _,_,rows in splits for row in rows]))
        results=[]
        for (encoded,meta),(_,_,rows) in zip(packed,splits,strict=True):
            results.append(dict(answers=to_answers([next(probabilities) for _ in rows],meta),
                usage=dict(input_tokens=len(encoded['ids']),output_tokens=0)))
        # Variable-length packing reads padding metadata before GPU kernels;
        # use the author's eager cached-prefix passes for this candidate.
        if not getattr(runtime,'_native_varlen_flash',False) and engine.capture_due(False):
            engine.capture_pending(limit=1)
        free,total=torch.cuda.mem_get_info()
        external=max(0,total-free-torch.cuda.memory_reserved())
        if total-external-torch.cuda.max_memory_allocated()<max(4*1024**3,total*.1):
            # The outer controller retains peak evidence and rejects this tier.
            # Release its bank so the smaller retry can actually use less memory.
            runtime._native_prefix_engine=None
    return results


def enable_kev_prefix_batch(runtime,align_states=False):
    if not runtime.model.hybrid:
        raise ValueError('GPU prefix batch requires the published hybrid Kev backbone')
    runtime._native_align_prefix_states=align_states
    if align_states:
        from jevlight.backends.kev_aligned_prefix import enable_aligned_prefix
        enable_aligned_prefix(runtime)
    def batch(self,requests,batch_size):
        import torch
        try:
            return native_prefix_batch(self,requests)
        except Exception as error:
            if isinstance(error,torch.cuda.OutOfMemoryError) or 'canUse32BitIndexMath' in str(error):
                # Successful construction can leave a large bank resident when
                # its forward OOMs. The smaller tier needs a fresh smaller bank.
                self._native_prefix_engine=None
            raise
    runtime.native_batch=MethodType(batch,runtime)
