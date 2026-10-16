"""Keep author projections packed through a complete independent-request MLP."""
import threading
from types import MethodType


def enable_lev_packed_mlp(runtime):
    import torch
    engine = runtime.engine
    native_context = getattr(runtime, '_step_lev_projection_context', None)
    if native_context is None:
        raise ValueError('Packed Lev MLP requires complete native projection shapes')
    if hasattr(engine, '_step_packed_mlp_forward'):
        return
    mlps = [m for m in engine._decoder.modules() if type(m).__name__ == 'Qwen3_5MLP']
    if not mlps:
        raise ValueError('Packed Lev MLP requires the native Qwen3.5 MLP')
    context = threading.local()
    for module in mlps:
        module._step_packed_mlp = module.forward
        def mlp(self, inputs):
            layout = getattr(context, 'layout', None)
            if not layout or not inputs.is_cuda or inputs.dtype != torch.bfloat16:
                return self._step_packed_mlp(inputs)
            if (inputs.ndim != 3 or inputs.shape[0] != sum(r for r, _ in layout)
                    or inputs.shape[1] != max(t for _, t in layout)):
                raise ValueError('Packed MLP lost complete independent native matrix shapes')
            from jevlight.backends.native_fused_ops import fused
            from jevlight.backends.native_packed_scatter import scatter_native_rows
            hidden = self.gate_proj.weight.shape[0]
            gates, ups, cursor = [], [], 0
            previous = getattr(native_context, 'layout', None)
            try:
                # Keep the complete author/PEFT projections, including their
                # low-rank additions and dtype transitions, at native shapes.
                native_context.layout = None
                for rows, width in layout:
                    native = inputs[cursor:cursor+rows, :width].contiguous()
                    gates.append(self.gate_proj(native).reshape(-1, hidden))
                    ups.append(self.up_proj(native).reshape(-1, hidden))
                    cursor += rows
                activated = fused(torch.cat(gates), torch.cat(ups))
                del gates, ups
                outputs, offset = [], 0
                for rows, width in layout:
                    native = activated[offset:offset+rows*width].view(rows, width, hidden)
                    outputs.append(self.down_proj(native).reshape(-1, inputs.shape[-1]))
                    offset += rows*width
                return scatter_native_rows(torch.cat(outputs), layout, inputs.shape[1], context)
            finally:
                native_context.layout = previous
        module.forward = MethodType(mlp, module)
    engine._step_packed_mlp_forward = engine._forward
    def forward(self, requests):
        previous = getattr(context, 'layout', None)
        try:
            context.layout = [(r.rows, r.width) for r in requests] if len(requests) > 1 else None
            return self._step_packed_mlp_forward(requests)
        finally:
            context.layout = previous
    engine._forward = MethodType(forward, engine)
