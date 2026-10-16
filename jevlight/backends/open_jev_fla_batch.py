"""Batch-only Open-Jev DeltaNet with isolated FLA dependencies."""
import importlib
import sys
import threading
from pathlib import Path
from types import MethodType


def load_batch_fla(package_path):
    """Import only pinned batch extras after constructing the author runtime."""
    import torch
    import transformers
    import triton
    package_path = Path(package_path).resolve()
    if not (package_path/'fla').is_dir() or not (package_path/'einops').is_dir():
        raise ValueError('Pinned isolated FLA batch packages are missing')
    original = (torch, transformers, triton)
    sys.path.append(str(package_path))
    try:
        fla = importlib.import_module('fla')
        if fla.__version__ != '0.5.2' or not Path(fla.__file__).resolve().is_relative_to((package_path/'fla').resolve()):
            raise ValueError('FLA batch implementation differs from the pinned isolated package')
        function = importlib.import_module('fla.ops.gated_delta_rule').chunk_gated_delta_rule
    finally:
        sys.path.remove(str(package_path))
    if original != tuple(sys.modules[name] for name in ('torch', 'transformers', 'triton')):
        raise ValueError('Batch dependency import replaced the author runtime')
    return function


def enable_packed_open_mlp(runtime, context):
    """Keep pure native ordinary MLP matrices packed until their final output."""
    import inspect
    import torch
    from jevlight.backends.native_fused_ops import fused
    from jevlight.backends.native_packed_scatter import scatter_native_rows
    native_context = None
    for projection in runtime.model.backbone.modules():
        if hasattr(projection, '_step_open_fla_projection'):
            native_context = inspect.getclosurevars(projection._step_open_fla_projection.__func__).nonlocals['context']
            break
    if native_context is None:
        raise ValueError('Packed Open MLP requires the complete native candidate projection context')
    mlps = [m for m in runtime.model.backbone.modules() if type(m).__name__ == 'Qwen3_5MLP']
    if not mlps or any(type(getattr(m, n)) is not torch.nn.Linear or getattr(m, n).bias is not None
                       for m in mlps for n in ('gate_proj', 'up_proj', 'down_proj')):
        raise ValueError('Packed Open MLP requires unchanged bias-free native Linear modules')
    for module in mlps:
        module._step_packed_open_mlp = module.forward
        def mlp(self, inputs):
            layout = getattr(native_context, 'layout', None)
            if (not getattr(context, 'active', False) or getattr(context, 'cooperative', False)
                    or not layout or not inputs.is_cuda or inputs.dtype != torch.bfloat16):
                return self._step_packed_open_mlp(inputs)
            if (inputs.ndim != 3 or inputs.shape[0] != sum(r for r, _ in layout)
                    or inputs.shape[1] != max(t for _, t in layout)):
                raise ValueError('Packed Open MLP lost complete native candidate rows')
            count, hidden = sum(r*t for r, t in layout), self.gate_proj.weight.shape[0]
            gate = inputs.new_empty((count, hidden))
            up = torch.empty_like(gate)
            cursor = offset = 0
            for rows, width in layout:
                native = inputs[cursor:cursor+rows, :width].contiguous()
                torch.nn.functional.linear(native, self.gate_proj.weight, None,
                    out=gate[offset:offset+rows*width].view(rows, width, hidden))
                torch.nn.functional.linear(native, self.up_proj.weight, None,
                    out=up[offset:offset+rows*width].view(rows, width, hidden))
                cursor += rows
                offset += rows*width
            activated = fused(gate, up)
            del gate, up
            packed = inputs.new_empty((count, self.down_proj.weight.shape[0]))
            offset = 0
            for rows, width in layout:
                torch.nn.functional.linear(activated[offset:offset+rows*width].view(rows, width, hidden),
                    self.down_proj.weight, None,
                    out=packed[offset:offset+rows*width].view(rows, width, packed.shape[-1]))
                offset += rows*width
            return scatter_native_rows(packed, layout, inputs.shape[1], native_context)
        module.forward = MethodType(mlp, module)


def enable_open_fla_batch(runtime, package_path, *, packed_mlp=False,
                          all_modes=False, full_linear=False):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    from jevlight.backends.causal_conv import batch_causal_conv
    if hasattr(runtime, '_step_open_fla_batch'):
        return
    blocks = [m for m in runtime.model.backbone.modules() if type(m).__name__ == 'Qwen3_5GatedDeltaNet']
    if not blocks or any(m.chunk_gated_delta_rule is not module.torch_chunk_gated_delta_rule
                         or type(m.norm).__name__ != 'Qwen3_5RMSNormGated'
                         or m.causal_conv1d_fn is not None for m in blocks):
        raise ValueError('Open FLA batch requires the unchanged author Torch scalar runtime')
    chunk = load_batch_fla(package_path)
    context = threading.local()
    original = module.torch_chunk_gated_delta_rule
    def delta(query, key, value, g, beta, chunk_size=64, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if (not getattr(context, 'active', False)
                or (not all_modes and not getattr(context, 'cooperative', False))
                or kwargs.get('cu_seqlens') is not None):
            return original(query, key, value, g=g, beta=beta, chunk_size=chunk_size,
                initial_state=initial_state, output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            # The pinned author normalizes in BF16 before its FP32 promotion.
            query = module.l2norm(query, dim=-1, eps=1e-6)
            key = module.l2norm(key, dim=-1, eps=1e-6)
        q = (query.float()*(query.shape[-1]**-.5)).contiguous()
        output, state = chunk(q, key.float().contiguous(), value.float().contiguous(),
            g.float().contiguous(), beta.float().contiguous(), scale=1.,
            initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False, chunk_size=chunk_size)
        return output.to(dtype), state
    for block in blocks:
        block.chunk_gated_delta_rule = delta
    native_convs = [block.causal_conv1d_fn for block in blocks]
    def convolution(*, x, weight, bias=None, activation=None, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise ValueError('Unsupported native Open batch convolution parameters')
        return batch_causal_conv(x, weight, bias, activation)
    # The original candidate head, FP32 calibration, norm and complete
    # author encoding remain in place. Ordinary projections retain native
    # candidate matrix shapes; Co uses its measured original batch shapes.
    for projection in [m for m in runtime.model.backbone.modules() if type(m) is torch.nn.Linear]+[runtime.model.head]:
        if not hasattr(projection, '_step_open_linear') and not hasattr(projection, '_step_open_head'):
            raise ValueError('Open FLA batch requires native-shaped ordinary projections')
        projection._step_open_fla_projection = projection.forward
        def choose(self, inputs):
            if getattr(context, 'active', False) and (full_linear or getattr(context, 'cooperative', False)):
                return self._step_open_head(inputs) if hasattr(self, '_step_open_head') else self._step_open_linear(inputs)
            return self._step_open_fla_projection(inputs)
        projection.forward = MethodType(choose, projection)
    if packed_mlp:
        enable_packed_open_mlp(runtime, context)
    runtime._step_open_fla_batch = runtime.native_batch
    def batch(self, requests, batch_size):
        previous = (getattr(context, 'active', False), getattr(context, 'cooperative', False))
        context.active = len(requests) > 1
        context.cooperative = bool(requests) and all('cooperative_view' in r.get('state', {}) for r in requests)
        try:
            for block in blocks:
                block.causal_conv1d_fn = convolution if context.active else None
            return self._step_open_fla_batch(requests, batch_size)
        finally:
            context.active, context.cooperative = previous
            for block, original_conv in zip(blocks, native_convs, strict=True):
                block.causal_conv1d_fn = original_conv
    runtime.native_batch = MethodType(batch, runtime)
