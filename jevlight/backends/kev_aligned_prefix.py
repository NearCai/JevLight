"""Keep native DeltaNet chunk boundaries in independent padded state passes."""
from types import MethodType


def enable_aligned_prefix(runtime, *, projection_names=None, native_delta_shapes=True):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    original=module.torch_chunk_gated_delta_rule
    runtime._native_prefix_meta=None
    runtime._native_prefix_active=False
    runtime._native_gate_ranges=None
    runtime.model.lm._native_dense_prefix=True
    enable_cached_attention(runtime)
    enable_native_projections(runtime, names=projection_names)
    def delta(query,key,value,g,beta,**kwargs):
        meta=runtime._native_prefix_meta
        if meta is not None:
            valid=meta['valid']
            if tuple(valid.shape)!=tuple(query.shape[:2]):
                raise ValueError('Aligned state mask does not match native DeltaNet tokens')
            mask=valid[:,:,None,None]
            query=query.masked_fill(~mask,0);key=key.masked_fill(~mask,0);value=value.masked_fill(~mask,0)
            g=g.masked_fill(~valid[:,:,None],0);beta=beta.masked_fill(~valid[:,:,None],0)
        ranges=runtime._native_gate_ranges if runtime._native_prefix_active and native_delta_shapes else None
        if ranges is None:return original(query,key,value,g=g,beta=beta,**kwargs)
        grouped={}
        for row,(left,end) in enumerate(ranges):
            if left:raise ValueError('Native DeltaNet rows require right padding')
            grouped.setdefault(end,[]).append(row)
        output=value.new_zeros(value.shape)
        state=None
        if kwargs.get('output_final_state'):
            state=torch.empty((value.shape[0],value.shape[2],key.shape[3],value.shape[3]),device=value.device,dtype=torch.float32)
        for length,rows in grouped.items():
            indices=torch.tensor(rows,device=query.device)
            tensors=[tensor.index_select(0,indices)[:,:length].contiguous() for tensor in (query,key,value,g,beta)]
            options=dict(kwargs)
            if options.get('initial_state') is not None:options['initial_state']=options['initial_state'].index_select(0,indices)
            result,final=original(*tensors[:3],g=tensors[3],beta=tensors[4],**options)
            output[indices,:length]=result
            if state is not None:state[indices]=final
        return output,state
    module.torch_chunk_gated_delta_rule=delta
    for layer in runtime.model.lm.modules():
        if type(layer).__name__!='Qwen3_5GatedDeltaNet':continue
        def tail(projection,args,output,owner=layer):
            meta=runtime._native_prefix_meta
            if meta is None:return
            positions=meta['end'][:,None]-owner.conv_kernel_size+torch.arange(owner.conv_kernel_size,device=output.device)
            captured=output.gather(1,positions.clamp_min(0)[:,:,None].expand(-1,-1,output.shape[-1]))
            owner._native_conv_tail=captured.masked_fill((positions<meta['left'][:,None])[:,:,None],0).transpose(1,2).contiguous()
        layer.in_proj_qkv.register_forward_hook(tail)
        def restore(owner,args,kwargs,output):
            if runtime._native_prefix_meta is None:return
            cache=kwargs['cache_params']
            cache.layers[owner.layer_idx].conv_states[0].copy_(owner._native_conv_tail)
            owner._native_conv_tail=None
        layer.register_forward_hook(restore,with_kwargs=True)


def install_aligned_states(engine,runtime):
    import torch
    from kev import cuda_graphs as author
    def states(self,items,Sb):
        # Every sequence begins at its native chunk origin. Freeze only the
        # trailing dummy tokens, including any extra whole empty chunks.
        Sb=((Sb+63)//64)*64
        count=len(items)
        lengths=torch.tensor([len(ids) for ids,_,_ in items],device=self.device)
        right=Sb-lengths
        left=torch.zeros_like(lengths)
        ids=torch.full((count,Sb),self.pad_id,dtype=torch.long,device=self.device)
        positions=torch.zeros_like(ids)
        for index,(tokens,pos,keep) in enumerate(items):
            if keep:raise ValueError('Aligned GPU state passes require native ephemeral prefixes')
            offset=0
            ids[index,offset:offset+len(tokens)]=torch.tensor(tokens,device=self.device)
            positions[index,offset:offset+len(tokens)]=torch.tensor(pos,device=self.device)
        at=torch.arange(Sb,device=self.device)
        valid=(at[None]>=left[:,None])&(at[None]<left[:,None]+lengths[:,None])
        allow=((at[None,None,:]<=at[None,:,None])&valid[:,None,:])|(at[None,None,:]==at[None,:,None])
        views=[(a[:count,:,author.BANK_WIDTH-Sb:],b[:count,:,author.BANK_WIDTH-Sb:]) if attention else (a[:count],b[:count])
               for (attention,*_),(a,b) in zip(self.bank.slots,self.bank_views,strict=True)]
        ranges=[(0,len(tokens)) for tokens,_,_ in items]
        runtime._native_prefix_meta=dict(valid=valid,left=left,end=left+lengths,ranges=ranges)
        try:
            self._forward(ids,positions,self._mask(allow),valid.long(),self._cache(views,0,False))
        finally:
            runtime._native_prefix_meta=None
        # The original author branch pass expects real K/V at the bank's end.
        # Relocate only these padding addresses; retain all native positions.
        source=(at[None]-right[:,None]).clamp_min(0)
        real=at[None]>=Sb-lengths[:,None]
        for (attention,*_),(a,b) in zip(self.bank.slots,self.bank_views,strict=True):
            if not attention:continue
            for tensor in (a,b):
                target=tensor[:count,:,author.BANK_WIDTH-Sb:]
                index=source[:,None,:,None].expand(count,target.shape[1],Sb,target.shape[-1])
                moved=target.gather(2,index).masked_fill(~real[:,None,:,None],0)
                target.copy_(moved)
        return [None]*count
    engine.states=MethodType(states,engine)
    original_run=engine.run
    def run(self,requests):
        previous=runtime._native_prefix_active
        runtime._native_prefix_active=True
        try:return original_run(requests)
        finally:runtime._native_prefix_active=previous
    engine.run=MethodType(run,engine)


def enable_cached_attention(runtime):
    """Remove only bank padding from the native cached causal attention."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from jevlight.backends.native_flash_split import forward as split_forward,native_splits,load_kernel
    load_kernel()
    sms=torch.cuda.get_device_properties(next(runtime.model.lm.parameters()).device).multi_processor_count
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    modules=[module for module in runtime.model.lm.modules() if type(module).__name__=='Qwen3_5Attention']
    runtime._native_cached_attention_gpu_calls=0
    def before(module,args,kwargs):
        for attention in modules:attention._native_cached_varlen=None
        runtime._native_gate_ranges=None
        if not runtime._native_prefix_active:return
        if runtime._native_prefix_meta is not None:
            runtime._native_gate_ranges=runtime._native_prefix_meta['ranges']
        cache=kwargs.get('past_key_values')
        if cache is None or cache.get_seq_length()<=0:return
        masks=kwargs.get('attention_mask')
        if not isinstance(masks,dict):raise ValueError('Native cached pass lost its complete masks')
        query_mask=masks.get('linear_attention');full=masks.get('full_attention')
        if not torch.is_tensor(query_mask) or query_mask.ndim!=2 or not torch.is_tensor(full) or full.ndim!=4:
            raise ValueError('Native cached pass masks have unsupported shapes')
        last=query_mask.long().sum(-1)-1
        if (last<0).any():raise ValueError('Native cached branch is empty')
        query_lengths=query_mask.long().sum(-1).cpu().tolist()
        runtime._native_gate_ranges=[(0,int(length)) for length in query_lengths]
        key_mask=full[torch.arange(full.shape[0],device=full.device),0,last,:]==0
        key_positions=[[i for i,valid in enumerate(row) if valid] for row in key_mask.cpu().tolist()]
        grouped={}
        for row,(qlen,positions) in enumerate(zip(query_lengths,key_positions,strict=True)):
            grouped.setdefault((qlen,len(positions)),[]).append(row)
        groups=[dict(query_length=qlen,rows=torch.tensor(rows,device=full.device),
            keys=torch.tensor([key_positions[row] for row in rows],device=full.device))
            for (qlen,klen),rows in grouped.items()]
        meta=dict(groups=groups)
        for attention in modules:attention._native_cached_varlen=meta
    runtime.model.lm.register_forward_pre_hook(before,with_kwargs=True)
    def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if runtime._native_prefix_active and runtime._native_prefix_meta is not None:
            # Flash's dense and variable-length forms differ near BF16 rounding
            # boundaries. Keep native sequence length/layout for this sensitive
            # checkpoint, sharing a causal kernel across equal-length states.
            ranges=runtime._native_prefix_meta['ranges']
            grouped={}
            for row,(left,end) in enumerate(ranges):
                if left:raise ValueError('Native prefix attention requires right padding')
                grouped.setdefault(end,[]).append(row)
            output=query.new_zeros((query.shape[0],query.shape[2],query.shape[1],query.shape[3]))
            for length,rows in grouped.items():
                indices=torch.tensor(rows,device=query.device)
                tensors=[tensor.index_select(0,indices)[:,:,:length].transpose(1,2).contiguous().transpose(1,2)
                         for tensor in (query,key,value)]
                splits=native_splits(length,query.shape[1],sms,query.shape[-1])
                if splits>1:
                    part=split_forward(*tensors,splits,scaling)
                else:
                    part,_=original(module,*tensors,None,dropout=0.,scaling=scaling,is_causal=True)
                output[indices,:length]=part
            return output,None
        meta=getattr(module,'_native_cached_varlen',None)
        if meta is None or dropout or kwargs.get('output_attentions'):
            return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        # Preserve the scalar author's dense additive mask as well as its
        # efficient kernel. Implicit causal varlen changes BF16 rounding.
        padded=query.new_zeros((query.shape[0],query.shape[2],query.shape[1],query.shape[3]))
        for group in meta['groups']:
            rows,positions,qlen=group['rows'],group['keys'],group['query_length']
            q=query.index_select(0,rows)[:,:,:qlen].contiguous()
            index=positions[:,None,:,None].expand(-1,key.shape[1],-1,key.shape[-1])
            k=key.index_select(0,rows).gather(2,index).contiguous()
            v=value.index_select(0,rows).gather(2,index).contiguous()
            mask=attention_mask.index_select(0,rows)[:,:,:qlen].gather(-1,
                positions[:,None,None,:].expand(-1,1,qlen,-1)).contiguous()
            output,_=original(module,q,k,v,mask,dropout=0.,scaling=scaling,is_causal=False)
            padded[rows,:qlen]=output
        runtime._native_cached_attention_gpu_calls+=1
        return padded,None
    ALL_ATTENTION_FUNCTIONS.register('sdpa',attention)


def enable_native_projections(runtime, *, names=None):
    """Keep native GEMM shapes inside the shared attention/DeltaNet forward."""
    import torch
    runtime._native_native_gate_projection_rows=0
    def projection(self,inputs):
        ranges=runtime._native_gate_ranges
        if not runtime._native_prefix_active or ranges is None:
            return self._native_native_gate_forward(inputs)
        if inputs.ndim!=3 or len(ranges)!=inputs.shape[0]:
            raise ValueError('Native gate ranges do not match independent decoder rows')
        output=inputs.new_zeros((*inputs.shape[:2],self.weight.shape[0]))
        for row,(left,end) in enumerate(ranges):
            if not 0<=left<end<=inputs.shape[1]:raise ValueError('Incomplete native gate input')
            output[row,left:end]=self._native_native_gate_forward(inputs[row:row+1,left:end].contiguous())[0]
        runtime._native_native_gate_projection_rows+=len(ranges)
        return output
    for name,module in runtime.model.lm.named_modules():
        if type(module) is torch.nn.Linear and (names is None or any(part in name for part in names)):
            module._native_native_gate_forward=module.forward
            module.forward=MethodType(projection,module)
