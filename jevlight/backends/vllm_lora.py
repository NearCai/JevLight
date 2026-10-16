"""Published FP32 LoRA for a single frozen vLLM adapter."""

def install_fp32_lora_linear():
    """Keep the published FP32 adapter where this engine's Triton ops require half."""
    import torch
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    original=PunicaWrapperGPU.add_lora_linear
    def linear(self,y,x,lora_a_stacked,lora_b_stacked,scale,output_slices,*,buffer=None,**kwargs):
        if not lora_a_stacked or lora_a_stacked[0].dtype!=torch.float32:
            return original(self,y,x,lora_a_stacked,lora_b_stacked,scale,output_slices,
                            buffer=buffer,**kwargs)
        if any(value.shape[0]!=1 for value in (*lora_a_stacked,*lora_b_stacked)):
            raise ValueError('Experimental FP32 adapter supports one frozen LoRA only')
        values=x.reshape(-1,x.shape[-1])
        mask=(self.token_lora_indices==0).reshape(-1,1)
        if mask.shape[0]!=values.shape[0]:raise ValueError('FP32 LoRA token mapping differs')
        if self.no_lora:
            return None
        output=y.reshape(-1,y.shape[-1]);offset=0
        for a,b,width in zip(lora_a_stacked,lora_b_stacked,output_slices,strict=True):
            if a.ndim!=4 or b.ndim!=4 or a.shape[1]!=1 or b.shape[1]!=1:
                raise ValueError('FP32 LoRA packed layout differs from reviewed release')
            free,total=torch.cuda.mem_get_info()
            free += torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
            margin=max(4096*2**20,total*.1)
            scratch=min(free*.25,max(0,free-margin))
            if scratch < 2**20:
                raise torch.cuda.OutOfMemoryError(
                    'FP32 adapter scratch cannot retain the required GPU memory margin; '
                    'reduce the token capacity or cache reservation')
            per_row=4*(values.shape[-1]+a.shape[-2]+width)+output.element_size()*width
            rows=max(1,min(values.shape[0],int(scratch//per_row)))
            for start in range(0,values.shape[0],rows):
                end=min(start+rows,values.shape[0])
                delta=torch.nn.functional.linear(
                    torch.nn.functional.linear(values[start:end].float(),a[0,0]),b[0,0])
                delta.mul_(scale).mul_(mask[start:end])
                output[start:end,offset:offset+width].add_(delta.to(output.dtype))
                del delta
            offset+=width
        return None
    PunicaWrapperGPU.add_lora_linear=linear


def verify_adapter_coverage(model, checkpoint):
    import json
    import torch
    from safetensors import safe_open
    configuration = json.loads((checkpoint / 'adapter/adapter_config.json').read_text())
    if configuration.get('rank_pattern') or configuration.get('alpha_pattern'):
        raise ValueError('Experimental adapter verification requires the published uniform rank/alpha')
    scale = configuration['lora_alpha'] / configuration['r']
    modules = dict(model.named_modules())
    checked = 0
    with safe_open(checkpoint / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as archive:
        for name in archive.keys():
            stem, kind = name.removeprefix('base_model.model.').rsplit('.lora_', 1)
            if not stem.startswith('layers.') or kind not in ('A.weight', 'B.weight'):
                raise ValueError(f'Unreviewed released LoRA tensor mapping: {name}')
            stem = 'model.' + stem
            part = 0
            leaf = stem.rsplit('.', 1)[-1]
            if leaf in ('q_proj', 'k_proj', 'v_proj'):
                part = ('q_proj', 'k_proj', 'v_proj').index(leaf)
                stem = stem.rsplit('.', 1)[0] + '.qkv_proj'
            module = modules.get(stem)
            if module is None:
                raise ValueError(f'Released adapter module not loaded: {stem}')
            values = getattr(module, 'lora_a_stacked' if kind == 'A.weight' else 'lora_b_stacked')
            expected = archive.get_tensor(name)
            if kind == 'B.weight':
                expected = expected * scale
            if leaf == 'in_proj_qkv' and len(values) > 1:
                pieces = [value[0, 0].float().cpu() for value in values]
                if kind == 'A.weight':
                    if any((not torch.equal(piece, expected) for piece in pieces)):
                        raise ValueError(f'Combined GDN adapter A is not repeated completely: {name}')
                    actual = pieces[0]
                else:
                    actual = torch.cat(pieces, dim=0)
            else:
                actual = values[part][0, 0].float().cpu()
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                diagnostics = dict(name=name, actual_shape=list(actual.shape), expected_shape=list(expected.shape), published_scale=scale)
                if actual.shape == expected.shape:
                    nonzero = expected != 0
                    diagnostics.update(max_abs_weight_difference=float((actual - expected).abs().max()), median_weight_ratio=float((actual[nonzero] / expected[nonzero]).median()), same_value_multiset=torch.equal(actual.flatten().sort().values, expected.flatten().sort().values))
                raise ValueError(f'Released adapter tensor missing, changed or mispacked: {diagnostics}')
            checked += 1
    return checked
