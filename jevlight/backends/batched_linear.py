"""Keep independent sequence GEMM shapes for low-precision native batches."""
from types import MethodType


def _forward(self,inputs):
    import torch
    if inputs.ndim!=3 or inputs.shape[0]<=1 or not inputs.is_cuda or self.bias is not None:
        return self._native_linear_forward(inputs)
    # Native single-sequence linears multiply [tokens,hidden]. Flattening the
    # request axis into tokens changes the low-precision GEMM chosen by CUDA.
    # A strided batched GEMM keeps each original matrix and shares the weights.
    return torch.bmm(inputs,self.weight.T.unsqueeze(0).expand(inputs.shape[0],-1,-1))


def enable_sequence_linears(model,include=None):
    import torch
    count=0
    for name,module in model.named_modules():
        if include is not None and not any(part in name.split('.') for part in include):
            continue
        if type(module) is torch.nn.Linear and module.bias is None and not hasattr(module,'_native_linear_forward'):
            module._native_linear_forward=module.forward
            module.forward=MethodType(_forward,module)
            count+=1
    return count
