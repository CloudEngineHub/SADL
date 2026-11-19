"""Architecture-independent MLP replay using PyTorch GEMMs and exact erf GELU."""
from types import MethodType

import torch
from torch.nn import functional as F


class RecomputeMLP(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, w1, b1, w2, b2):
        ctx.shape, ctx.input_dtype = x.shape, x.dtype
        ctx.parameter_dtypes = tuple(value.dtype for value in (w1, b1, w2, b2))
        inputs = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
        w1, b1, w2, b2 = (value.to(torch.bfloat16) for value in (w1, b1, w2, b2))
        hidden = F.gelu(F.linear(inputs, w1, b1), approximate='none')
        output = F.linear(hidden, w2, b2)
        ctx.save_for_backward(inputs, w1, b1, w2)
        return output.reshape(*ctx.shape[:-1], w2.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        inputs, w1, b1, w2 = ctx.saved_tensors
        gradient = gradient.reshape(-1, w2.shape[0]).to(torch.bfloat16)
        preactivation = F.linear(inputs, w1, b1)
        hidden = F.gelu(preactivation, approximate='none')
        delta = torch.ops.aten.gelu_backward.default(gradient @ w2, preactivation)
        dx = (delta @ w1).to(ctx.input_dtype).reshape(ctx.shape)
        parameters = (delta.T @ inputs, delta.sum(0), gradient.T @ hidden, gradient.sum(0))
        return dx, *(value.to(dtype) for value, dtype in zip(parameters, ctx.parameter_dtypes))


def configure_portable_mlp(model, *, recompute=True):
    schema = [(name, id(value), tuple(value.shape)) for name, value in model.named_parameters()]
    selected = []
    for index, layer in enumerate(model.encoder.layers[:-1]):
        mlp = layer.mlp
        if mlp.config.hidden_act != 'gelu' or mlp.fc1.bias is None or mlp.fc2.bias is None:
            raise ValueError('Portable MLP requires the existing biased erf-GELU MLP')
        selected.append(index)
        if not recompute:
            continue
        original = mlp.forward

        def forward(self, x, _original=original):
            if (not self.training or not x.is_cuda or not torch.is_autocast_enabled('cuda')
                    or torch.get_autocast_dtype('cuda') != torch.bfloat16):
                return _original(x)
            return RecomputeMLP.apply(x, self.fc1.weight, self.fc1.bias,
                                      self.fc2.weight, self.fc2.bias)

        mlp.forward = MethodType(forward, mlp)
    if schema != [(name, id(value), tuple(value.shape)) for name, value in model.named_parameters()]:
        raise RuntimeError('Portable MLP changed parameter ownership')
    model._exec_fused_mlp = {'backend': 'torch_inductor_cublas', 'layers': selected,
                            'activation': 'erf_gelu', 'preactivation_recompute': recompute,
                            'optional_cuda_packages': False}
    return selected
