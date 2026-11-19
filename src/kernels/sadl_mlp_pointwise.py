"""MLP replay with GEMM buffers reused by the lookup epilogue."""
from functools import lru_cache
import json

import torch
import triton
import triton.language as tl
from sadl_gelu_lookup import unpack


@triton.jit
def numerical_gelu(value, POLYNOMIAL: tl.constexpr):
    density = tl.exp(-0.5*value*value)
    if POLYNOMIAL:
        t = 1.0/(1.0+0.3275911*tl.abs(value)*0.7071067811865476)
        tail = (((((1.061405429*t-1.453152027)*t+1.421413741)*t
                   -0.284496736)*t+0.254829592)*t)*density*0.5
        cdf = tl.where(value >= 0, 1.0-tail, tail)
    else:
        cdf = 0.5*(1.0+tl.erf(value*0.7071067811865476))
    return value*cdf, cdf+value*density*0.3989422804014327


@triton.jit
def _epilogue(PREACT, INCOMING, GELU, DERIVATIVE, N: tl.constexpr,
              BLOCK: tl.constexpr, ANALYTIC: tl.constexpr, STREAM: tl.constexpr, PACKED: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    preact = tl.load(PREACT+offsets, offsets < N, 0, cache_modifier='.cg' if STREAM else '')
    incoming = tl.load(INCOMING+offsets, offsets < N, 0,
                       cache_modifier='.cg' if STREAM else '').to(tl.float32)
    if ANALYTIC:
        post, derivative = numerical_gelu(preact.to(tl.float32), ANALYTIC == 2)
    else:
        indices = preact.to(tl.uint16, bitcast=True).to(tl.int32)
        if PACKED:
            post, derivative = unpack(DERIVATIVE, indices)
        else:
            post = tl.load(GELU+indices)
            derivative = tl.load(DERIVATIVE+indices)
    tl.store(PREACT+offsets, post, offsets < N)
    tl.store(INCOMING+offsets, incoming*derivative, offsets < N)


@triton.jit
def _persistent_epilogue(PREACT, INCOMING, GELU, DERIVATIVE, N: tl.constexpr,
                         BLOCK: tl.constexpr, STREAM: tl.constexpr):
    for start in range(tl.program_id(0)*BLOCK, N, tl.num_programs(0)*BLOCK):
        offsets = start+tl.arange(0, BLOCK)
        preact = tl.load(PREACT+offsets, offsets < N, 0,
                         cache_modifier='.cg' if STREAM else '')
        incoming = tl.load(INCOMING+offsets, offsets < N, 0,
                           cache_modifier='.cg' if STREAM else '').to(tl.float32)
        indices = preact.to(tl.uint16, bitcast=True).to(tl.int32)
        post = tl.load(GELU+indices)
        derivative = tl.load(DERIVATIVE+indices)
        tl.store(PREACT+offsets, post, offsets < N)
        tl.store(INCOMING+offsets, incoming*derivative, offsets < N)


@lru_cache(maxsize=128)
def parse_schedule(value):
    config = json.loads(value)
    if set(config) != {'block', 'warps', 'ctas_per_sm', 'stream'}:
        raise ValueError('Invalid replay schedule fields')
    if (config['block'] not in (1024, 2048, 4096, 8192, 16384)
            or config['warps'] not in (4, 8)
            or config['ctas_per_sm'] not in (0, 1, 2, 4)
            or type(config['stream']) is not bool):
        raise ValueError('Unsupported replay schedule')
    return config


@torch.library.custom_op('sadl_mlp_pointwise::backward_replay', mutates_args=())
def backward_replay(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                    gradient: torch.Tensor, w2: torch.Tensor,
                    gelu: torch.Tensor, derivative: torch.Tensor,
                    replay_config: str, block: int,
                    incoming_config: str = '', epilogue: str = 'lookup',
                    packed: torch.Tensor | None = None, schedule: str = ''
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    if any(not value.is_cuda or value.dtype != torch.bfloat16 or not value.is_contiguous()
           for value in (x, w1, b1, gradient, w2, gelu)):
        raise ValueError('Pointwise replay requires contiguous CUDA BF16 inputs')
    if (x.ndim != 2 or gradient.shape != (x.shape[0], w2.shape[0])
            or w1.shape[1] != x.shape[1] or b1.shape != (w1.shape[0],)
            or w2.shape[1] != w1.shape[0] or gelu.shape != (65536,)
            or derivative.shape != (65536,) or derivative.dtype != torch.float32):
        raise ValueError('Mismatched MLP or lookup geometry')
    if block not in (256, 512, 1024, 2048):
        raise ValueError('Unsupported pointwise block')
    if epilogue not in ('lookup', 'analytic', 'polynomial', 'stream'):
        raise ValueError('Unknown epilogue')
    if packed is not None and (epilogue != 'lookup' or packed.shape != (65536,)
            or packed.dtype != torch.int64 or packed.device != x.device or not packed.is_contiguous()):
        raise ValueError('Invalid lossless packed lookup')
    from sadl_mlp import configured_gemm
    preact = (configured_gemm(x, w1.T, b1, replay_config) if replay_config else
              torch.nn.functional.linear(x, w1, b1))
    incoming = (configured_gemm(gradient, w2, None, incoming_config)
                if incoming_config else gradient @ w2)
    # Both buffers are newly allocated here, never tensors saved by the caller.
    if schedule:
        if packed is not None or epilogue != 'lookup':
            raise ValueError('Scheduled replay requires native unpacked lookup')
        config = parse_schedule(schedule)
        grid = triton.cdiv(preact.numel(), config['block'])
        if config['ctas_per_sm']:
            grid = min(grid, torch.cuda.get_device_properties(x.device).multi_processor_count
                       * config['ctas_per_sm'])
        _persistent_epilogue[(grid,)](preact, incoming, gelu, derivative, preact.numel(),
            config['block'], config['stream'], num_warps=config['warps'])
    else:
        _epilogue[(triton.cdiv(preact.numel(), block),)](
            preact, incoming, gelu, derivative if packed is None else packed, preact.numel(), block,
            2 if epilogue=='polynomial' else int(epilogue=='analytic'), epilogue=='stream', packed is not None,
            num_warps=4 if block <= 1024 else 8)
    return incoming, preact


@backward_replay.register_fake
def _fake(x, w1, b1, gradient, w2, gelu, derivative, replay_config, block,
          incoming_config='', epilogue='lookup', packed=None, schedule=''):
    shape = (x.shape[0], w1.shape[0])
    return x.new_empty(shape), x.new_empty(shape)
