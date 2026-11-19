"""Compile adapter for flash-attn-4==4.0.0b35."""
import importlib.metadata

import torch
from torch.utils.checkpoint import CheckpointPolicy

if importlib.metadata.version('flash-attn-4') != '4.0.0b35':
    raise RuntimeError('This adapter requires flash-attn-4==4.0.0b35')

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


@torch.library.custom_op('sadl_fa4::forward', mutates_args=())
def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _, _ = _flash_attn_fwd(q, k, v, softmax_scale=scale,
                                        causal=False, return_lse=True)
    return out.contiguous(), lse.contiguous()


@attention_forward.register_fake
def _forward_fake(q, k, v, scale):
    return (q.new_empty((*q.shape[:-1], v.shape[-1])),
            torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device))


@torch.library.custom_op('sadl_fa4::backward', mutates_args=())
def attention_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       out: torch.Tensor, gradient: torch.Tensor, lse: torch.Tensor,
                       scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(q, k, v, out, gradient, lse,
                                softmax_scale=scale, causal=False, softcap=0.0)
    return dq.contiguous(), dk.contiguous(), dv.contiguous()


@attention_backward.register_fake
def _backward_fake(q, k, v, out, gradient, lse, scale):
    return tuple(x.new_empty(x.shape) for x in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v, ctx.scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.mark_non_differentiable(lse)


def _backward(ctx, gradient, unused_lse_gradient):
    q, k, v, out, lse = ctx.saved_tensors
    return (*attention_backward(q, k, v, out, gradient, lse, ctx.scale), None)


attention_forward.register_autograd(_backward, setup_context=_setup_context)


def save_attention(ctx, op, *args, **kwargs):
    if op == torch.ops.sadl_fa4.forward.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def attention_cudnn_tail(q, k, v, scale):
    if q.shape[1] != 1:
        return attention_forward(q,k,v,scale)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale=scale)
    return output.transpose(1,2),None


def layer_forward(layer, hidden, *, cls_only=False, attention_impl=None, packed_qkv=False,
                  parameter_cast_key=-1):
    def linear(projection, value):
        if parameter_cast_key < 0:
            return projection(value)
        from sadl_parameter_cast import cast
        weight = cast(projection.weight, parameter_cast_key)
        bias = cast(projection.bias, parameter_cast_key) if projection.bias is not None else None
        return torch.nn.functional.linear(value, weight, bias)

    attention = layer.attention
    normed = layer.layernorm_before(hidden)
    batch, length = normed.shape[:2]
    heads, width = attention.num_attention_heads, attention.head_dim
    query_input = normed[:, :1] if cls_only else normed
    query_length = 1 if cls_only else length
    qkv_backward = getattr(layer, '_exec_qkv_backward', None)
    batched_qkv = getattr(layer, '_exec_batched_qkv', None)
    fused_dgrad = getattr(layer, '_exec_qkv_dgrad', None)
    if fused_dgrad and not cls_only:
        if packed_qkv or qkv_backward or batched_qkv or parameter_cast_key >= 0:
            raise ValueError('Fused QKV derivatives cannot combine with other projection rewrites')
        from sadl_qkv_dgrad import project
        q, k, v = (value.view(batch,length,heads,width) for value in
                   project(normed,attention,fused_dgrad))
    elif batched_qkv and not cls_only:
        if packed_qkv or qkv_backward or parameter_cast_key >= 0:
            raise ValueError('Batched QKV cannot be combined with other projection rewrites')
        from sadl_batched_qkv import project_batched_qkv
        q, k, v = (value.view(batch,length,heads,width) for value in
                   project_batched_qkv(normed,attention,batched_qkv=='both'))
    elif qkv_backward and not cls_only:
        if packed_qkv or parameter_cast_key >= 0:
            raise ValueError('QKV backward packing cannot be combined with forward packing/cast caching')
        from sadl_qkv import project_qkv
        q, k, v = (value.view(batch,length,heads,width) for value in
                   project_qkv(normed,attention,combine_input=qkv_backward=='combined'))
    elif packed_qkv:
        projections = (attention.k_proj, attention.v_proj) if cls_only else (
            attention.q_proj, attention.k_proj, attention.v_proj)
        weight = torch.cat([projection.weight for projection in projections])
        bias = torch.cat([projection.bias for projection in projections])
        values = torch.nn.functional.linear(normed, weight, bias).view(
            batch, length, len(projections), heads, width).unbind(2)
        if cls_only:
            q = attention.q_proj(query_input).view(batch, query_length, heads, width)
            k, v = values
        else:
            q, k, v = values
    else:
        q = linear(attention.q_proj, query_input).view(batch, query_length, heads, width)
        k = linear(attention.k_proj, normed).view(batch, length, heads, width)
        v = linear(attention.v_proj, normed).view(batch, length, heads, width)
    context, _ = (attention_impl or attention_forward)(q, k, v, attention.scaling)
    context = context.reshape(batch, query_length, heads*width)
    residual = hidden[:, :1] if cls_only else hidden
    if getattr(attention.o_proj,'_exec_residual_configuration',''):
        from sadl_residual_gemm import attention_output
        output = attention_output(attention.o_proj,context,residual)
    else:
        output = linear(attention.o_proj, context) + residual
    if getattr(layer, '_exec_norm_mlp_backward', None) is not None:
        if parameter_cast_key >= 0:
            raise ValueError('Fused norm/MLP backward does not use parameter cast caching')
        from sadl_norm_backward_gemm import norm_mlp
        return norm_mlp(layer, output) + output
    normed = layer.layernorm_after(output)
    if getattr(layer.mlp,'_exec_residual_configuration',''):
        return layer.mlp._exec_residual_forward(normed,output)
    if parameter_cast_key >= 0:
        mlp = linear(layer.mlp.fc2, layer.mlp.activation_fn(linear(layer.mlp.fc1, normed)))
    else:
        mlp = layer.mlp(normed)
    return mlp + output
