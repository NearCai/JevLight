"""Use the frozen Kev author's fused serving kernels without a new environment."""
import importlib.util
from pathlib import Path


def enable_kev_author_fusion(runtime, package_path):
    if hasattr(runtime, '_step_author_fused_kev'):
        return
    from jevlight.backends.open_jev_fla_batch import load_batch_fla
    package_path = Path(package_path)
    if importlib.util.find_spec('fla') is None:
        load_batch_fla(package_path)
    else:
        import fla
        if fla.__version__ != '0.5.2':
            raise ValueError('Frozen Kev fusion requires FLA0.5.2')
        actual, pinned = Path(fla.__file__).parent, package_path/'fla'
        for source in pinned.rglob('*.py'):
            relative = source.relative_to(pinned)
            if not (actual/relative).is_file() or source.read_bytes() != (actual/relative).read_bytes():
                raise ValueError(f'Existing FLA differs from frozen kernel source: {relative}')
    if hasattr(runtime.model.lm, 'peft_config'):
        raise ValueError('Author fusion requires a full or already merged backbone')
    from kev.fused_qwen35 import fuse
    fuse(runtime.model.lm)
    # FLA's normalization kernels include signed 32-bit flattened offsets.
    # Reject an oversized padded group before any
    # kernel runs, so the normal capacity tiers can retry in the same context.
    # An illegal access after launch would poison that CUDA context instead.
    widths = [max(getattr(module, 'key_dim', 0),getattr(module, 'value_dim', 0))
              for module in runtime.model.lm.modules()
              if type(module).__name__ == 'Qwen3_5GatedDeltaNet']
    key_width = max([*widths,getattr(runtime.model.lm.config,'hidden_size',0)])
    def check_kernel_index(module, args, kwargs):
        tensor = kwargs.get('input_ids')
        if tensor is None:
            tensor = kwargs.get('inputs_embeds')
        if tensor is None and args:
            tensor = args[0]
        if tensor is None or tensor.ndim < 2:
            return
        elements = int(tensor.shape[0]) * int(tensor.shape[1]) * key_width
        if elements >= 2**31:
            raise RuntimeError(
                'canUse32BitIndexMath(input) && canUse32BitIndexMath(output): '
                f'author FLA normalization preflight: {elements} elements '
                f'({tensor.shape[0]} rows, {tensor.shape[1]} tokens, '
                f'{key_width} features) exceed signed 32-bit offsets; '
                'reduce the GPU group without truncating inputs')
    runtime.model.lm.register_forward_pre_hook(check_kernel_index, with_kwargs=True)
    runtime._step_author_fused_kev = True
