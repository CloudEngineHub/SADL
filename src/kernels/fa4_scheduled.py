"""Measured SM90 backward schedule for the fixed SADL encoder head dimensions."""
from dataclasses import replace
from types import FunctionType
import os

import torch
from torch.utils.checkpoint import CheckpointPolicy

from fa4_execution import _flash_attn_fwd, _flash_attn_bwd, _forward_fake, _backward_fake

_original_tile = _flash_attn_bwd.__globals__['_tile_size_bwd_sm90']
_deterministic = os.environ.get('SADL_DETERMINISTIC_ATTENTION') == '1'
_backward_configuration = None


def set_backward_configuration(configuration):
    """Select an explicitly measured SM90 schedule without changing the default."""
    global _backward_configuration
    if configuration is not None:
        base = _original_tile(64, 64, False, False)
        replace(base, **configuration)
        configuration = dict(configuration)
    previous = _backward_configuration
    _backward_configuration = configuration
    return previous


def _tile(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    config = _original_tile(head_dim, head_dim_v, causal, local, sparse_block_size_q)
    if head_dim == head_dim_v == 64 and not (causal or local or sparse_block_size_q):
        if _backward_configuration is not None:
            return replace(config, **_backward_configuration)
        return replace(config, m_block_size=64, n_block_size=96,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=True,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
            num_wg=2, dQ_single_wg=False)
    return config


# Isolate the upstream helper binding; ordinary FA4 calls retain their schedule.
_globals = dict(_flash_attn_bwd.__globals__, _tile_size_bwd_sm90=_tile)
_backward_impl = FunctionType(_flash_attn_bwd.__code__, _globals,
    _flash_attn_bwd.__name__, _flash_attn_bwd.__defaults__, _flash_attn_bwd.__closure__)
_backward_impl.__kwdefaults__ = _flash_attn_bwd.__kwdefaults__
_backward_impl.__dict__.update(_flash_attn_bwd.__dict__)
_globals['_flash_attn_bwd'] = _backward_impl


@torch.library.custom_op('sadl_fa4_scheduled::forward', mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _flash_attn_fwd(q, k, v, softmax_scale=scale,
        causal=False, return_lse=True)
    return out.contiguous(), lse.contiguous()


attention.register_fake(_forward_fake)


@torch.library.custom_op('sadl_fa4_scheduled::backward', mutates_args=())
def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             out: torch.Tensor, gradient: torch.Tensor, lse: torch.Tensor,
             scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _backward_impl(q, k, v, out, gradient, lse,
        softmax_scale=scale, causal=False, softcap=0.0, deterministic=_deterministic)
    return tuple(value.contiguous() for value in values)


backward.register_fake(_backward_fake)


def _setup_context(ctx, inputs, output):
    q, k, v, ctx.scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.mark_non_differentiable(lse)


def _autograd(ctx, gradient, unused_lse_gradient):
    q, k, v, out, lse = ctx.saved_tensors
    return (*backward(q, k, v, out, gradient, lse, ctx.scale), None)


attention.register_autograd(_autograd, setup_context=_setup_context)


def save_attention(ctx, op, *args, **kwargs):
    if op == torch.ops.sadl_fa4_scheduled.forward.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE
