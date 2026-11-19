"""Lossless packing of native GELU BF16 values and FP32 derivatives."""
import torch
import triton
import triton.language as tl


def pack(gelu, derivative):
    if (gelu.shape != (65536,) or derivative.shape != gelu.shape
            or gelu.dtype != torch.bfloat16 or derivative.dtype != torch.float32
            or not gelu.is_cuda or derivative.device != gelu.device):
        raise ValueError('Expected the complete native CUDA GELU lookup tables')
    values = gelu.contiguous().view(torch.int16).to(torch.int64) & 0xffff
    gradients = derivative.contiguous().view(torch.int32).to(torch.int64) & 0xffffffff
    return (gradients << 16) | values


@triton.jit
def unpack(table, indices):
    bits = tl.load(table + indices)
    value = (bits & 0xffff).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    derivative = ((bits >> 16) & 0xffffffff).to(tl.uint32).to(tl.float32, bitcast=True)
    return value, derivative
