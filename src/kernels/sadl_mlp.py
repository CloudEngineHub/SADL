"""GEMM epilogues for erf GELU and its derivative."""
from types import MethodType
from functools import partial
from dataclasses import replace
import json
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16, Uint16, Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir_helpers.arith import bitcast
from quack.rounding import RoundingMode

# PyTorch 2.7/Python 3.10 require numeric text for enum-valued custom-op defaults.
RoundingMode.__str__ = lambda self: str(int(self))

from quack.activation import act_fn_map, dact_fn_map
from quack.gemm_interface import (act_to_pytorch_fn_map, gemm_act, gemm_dact,
                                  gemm_act_tuned, gemm_dact_tuned, gemm_tuned, default_config)


@dsl_user_op
def _gelu(x, *, loc=None, ip=None):
    if const_expr(isinstance(x, tuple)):
        return (_gelu(x[0], loc=loc, ip=ip), _gelu(x[1], loc=loc, ip=ip))
    value = x.to(BFloat16, loc=loc, ip=ip).to(Float32, loc=loc, ip=ip)
    cdf = (1.0 + cute.math.erf(value * 0.7071067811865476, loc=loc, ip=ip)) * 0.5
    return value * cdf


@dsl_user_op
def _gelu_backward(x, gradient, *, loc=None, ip=None):
    if const_expr(isinstance(x, tuple)):
        a, b = _gelu_backward(x[0], gradient[0], loc=loc, ip=ip)
        c, d = _gelu_backward(x[1], gradient[1], loc=loc, ip=ip)
        return (a, c), (b, d)
    value = x.to(BFloat16, loc=loc, ip=ip).to(Float32, loc=loc, ip=ip)
    gradient = gradient.to(BFloat16, loc=loc, ip=ip).to(Float32, loc=loc, ip=ip)
    cdf = (1.0 + cute.math.erf(value * 0.7071067811865476, loc=loc, ip=ip)) * 0.5
    density = cute.math.exp(-0.5 * value * value, loc=loc, ip=ip) * 0.3989422804014327
    return gradient * (cdf + value * density), value * cdf


_ACTIVATION = 'sadl_erf_gelu_bf16'
act_fn_map[_ACTIVATION] = _gelu
dact_fn_map[_ACTIVATION] = _gelu_backward
act_to_pytorch_fn_map[_ACTIVATION] = torch.nn.functional.gelu


@dsl_user_op
def _lookup(x, address, dtype, *, loc=None, ip=None):
    rounded = x.to(BFloat16, loc=loc, ip=ip)
    index = Uint16(bitcast(rounded.ir_value(loc=loc, ip=ip), T.i16(), loc=loc, ip=ip)).to(Int32)
    pointer = cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16, loc=loc, ip=ip)
    table = cute.make_tensor(pointer, cute.make_layout((65536,)), loc=loc, ip=ip)
    return table[index].to(Float32, loc=loc, ip=ip)


@dsl_user_op
def _gelu_lookup(x, address, *, loc=None, ip=None):
    if const_expr(isinstance(x, tuple)):
        return (_gelu_lookup(x[0], address, loc=loc, ip=ip),
                _gelu_lookup(x[1], address, loc=loc, ip=ip))
    return _lookup(x, address, BFloat16, loc=loc, ip=ip)


@dsl_user_op
def _gelu_lookup_backward(x, gradient, address, derivative_address, *, loc=None, ip=None):
    if const_expr(isinstance(x, tuple)):
        a, b = _gelu_lookup_backward(x[0], gradient[0], address, derivative_address, loc=loc, ip=ip)
        c, d = _gelu_lookup_backward(x[1], gradient[1], address, derivative_address, loc=loc, ip=ip)
        return (a, c), (b, d)
    gradient = gradient.to(BFloat16, loc=loc, ip=ip).to(Float32, loc=loc, ip=ip)
    return (gradient * _lookup(x, derivative_address, Float32, loc=loc, ip=ip),
            _gelu_lookup(x, address, loc=loc, ip=ip))


_lookup_tables = {}
_packed_lookup_tables = {}


def prepare_gelu_lookup(device):
    """Own immutable tables for the process lifetime and key kernels by their addresses."""
    device = torch.device(device)
    if device not in _lookup_tables:
        values = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        with torch.enable_grad(), torch.autocast('cuda', enabled=False):
            inputs = values.float().requires_grad_()
            output = torch.nn.functional.gelu(inputs)
            derivative, = torch.autograd.grad(output, inputs, torch.ones_like(output))
        forward = torch.nn.functional.gelu(values).detach()
        derivative = derivative.detach()
        # Absolute addresses are part of the compiled kernel key; stale pointers cannot be reused.
        key = f'sadl_gelu_lookup_{forward.data_ptr()}_{derivative.data_ptr()}'
        act_fn_map[key] = partial(_gelu_lookup, address=forward.data_ptr())
        dact_fn_map[key] = partial(_gelu_lookup_backward, address=forward.data_ptr(),
                                  derivative_address=derivative.data_ptr())
        act_to_pytorch_fn_map[key] = torch.nn.functional.gelu
        _lookup_tables[device] = (key, forward, derivative)
    return _lookup_tables[device][0]


@torch.library.custom_op('sadl_mlp::gemm_activation', mutates_args=())
def _gemm_activation(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor,
                     activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    preact = a.new_empty((a.shape[0], b.shape[1]))
    postact = torch.empty_like(preact)
    config = replace(default_config(a.device), tile_m=128, tile_n=128)
    gemm_act_tuned.fn(a, b, preact, postact, bias=bias, activation=activation, config=config)
    return preact, postact


@_gemm_activation.register_fake
def _gemm_activation_fake(a, b, bias, activation):
    return a.new_empty((a.shape[0], b.shape[1])), a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op('sadl_mlp::gemm_postactivation', mutates_args=())
def _gemm_postactivation(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor,
                         activation: str) -> torch.Tensor:
    postact = a.new_empty((a.shape[0], b.shape[1]))
    config = replace(default_config(a.device), tile_m=128, tile_n=128)
    gemm_act_tuned.fn(a, b, None, postact, bias=bias, activation=activation, config=config)
    return postact


@_gemm_postactivation.register_fake
def _gemm_postactivation_fake(a, b, bias, activation):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op('sadl_mlp::gemm_derivative', mutates_args=())
def _gemm_derivative(a: torch.Tensor, b: torch.Tensor, preact: torch.Tensor,
                     activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    delta, postact = torch.empty_like(preact), torch.empty_like(preact)
    config = replace(default_config(a.device), tile_m=128, tile_n=128)
    gemm_dact_tuned.fn(a, b, preact, delta, postact, activation=activation, config=config)
    return delta, postact


@_gemm_derivative.register_fake
def _gemm_derivative_fake(a, b, preact, activation):
    return torch.empty_like(preact), torch.empty_like(preact)


class _FusedMLP(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, w1, b1, w2, b2, activation=_ACTIVATION, tiled=False, recompute=False,
                dense_backend=False, te_wgrad=False, fused_bias=False, plain_fc2='', plain_dx='',
                plain_replay='', native_derivative=False, dual_replay=False,
                pointwise_replay=False, parallel_gradients=0, wgrad_transpose=0, packed_lookup=False,
                pointwise_schedule='', gelu_schedule=(), residual=None, residual_configuration=''):
        shape = x.shape
        ctx.input_dtype, ctx.parameter_dtypes = x.dtype, (w1.dtype, b1.dtype, w2.dtype, b2.dtype)
        x = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        w1, b1, w2, b2 = [value.to(torch.bfloat16) for value in (w1, b1, w2, b2)]
        if gelu_schedule and gelu_schedule[0] == 'forward':
            from sadl_gelu_polynomial import postactivation, ACTIVATION
            postact = postactivation(x, w1.T, b1, gelu_schedule[2],
                activation if gelu_schedule[1]=='lookup' else ACTIVATION)
        elif recompute:
            postact = _gemm_postactivation(x, w1.T, b1, activation)
        else:
            preact, postact = (_gemm_activation(x, w1.T, b1, activation) if tiled else
                              gemm_act(x, w1.T, bias=b1, activation=activation, tuned=False))
        ctx.residual_dtype = residual.dtype if residual is not None else None
        if residual is not None:
            from sadl_residual_gemm import forward as residual_gemm
            output = residual_gemm(postact,w2,b2,residual.reshape(-1,w2.shape[0]),residual_configuration)
        else:
            output = (configured_gemm(postact, w2.T, b2, plain_fc2) if plain_fc2 else
                  dense_gemm(postact, w2.T, b2) if dense_backend else
                  torch.nn.functional.linear(postact, w2, b2))
        # Recreate only the linear preactivation, avoiding a second GELU output write.
        saved = (x, w1, w2, b1 if recompute else preact)
        ctx.save_for_backward(*saved, *((b2,) if te_wgrad else ()))
        ctx.input_shape = shape
        ctx.activation = activation
        ctx.tiled = tiled
        ctx.recompute = recompute
        ctx.dense_backend = dense_backend
        ctx.te_wgrad = te_wgrad
        ctx.fused_bias = fused_bias
        ctx.plain_dx = plain_dx
        ctx.plain_replay = plain_replay
        ctx.native_derivative = native_derivative
        ctx.dual_replay = dual_replay
        ctx.pointwise_replay = pointwise_replay
        ctx.parallel_gradients = parallel_gradients
        ctx.wgrad_transpose = wgrad_transpose
        ctx.packed_lookup = packed_lookup
        ctx.pointwise_schedule = pointwise_schedule
        ctx.gelu_schedule = gelu_schedule
        return output.reshape(*shape[:-1], w2.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        x, w1, w2, saved, *extra = ctx.saved_tensors
        gradient = gradient.reshape(-1, w2.shape[0]).contiguous()
        packed = _packed_lookup_tables[x.device] if ctx.packed_lookup else None
        gelu_schedule = ctx.gelu_schedule
        if gelu_schedule and gelu_schedule[0] == 'dual_polynomial':
            from sadl_mlp_dual import backward_replay
            _, gelu, derivative = _lookup_tables[x.device]
            delta, postact = backward_replay(x.contiguous(),w1,saved,gradient,w2,
                gelu,derivative,*gelu_schedule[4],None,'polynomial')
        elif gelu_schedule and gelu_schedule[0] == 'backward_owned':
            from sadl_gelu_polynomial import owned_replay, ACTIVATION
            delta, postact = owned_replay(x,w1,saved,gradient,w2,ctx.plain_replay,
                gelu_schedule[2], ctx.activation if gelu_schedule[1]=='lookup' else ACTIVATION)
        elif gelu_schedule and gelu_schedule[0] == 'backward':
            from sadl_gelu_polynomial import derivative, ACTIVATION
            preact = configured_gemm(x, w1.T, saved, ctx.plain_replay)
            delta, postact = derivative(gradient, w2, preact,
                gelu_schedule[2], ctx.activation if gelu_schedule[1]=='lookup' else ACTIVATION)
        elif getattr(ctx, 'pointwise_replay', False):
            from sadl_mlp_pointwise import backward_replay
            _, gelu, derivative = _lookup_tables[x.device]
            polynomial = bool(gelu_schedule and gelu_schedule[0]=='pointwise_polynomial')
            delta, postact = backward_replay(x.contiguous(), w1, saved, gradient, w2,
                gelu, derivative, ctx.plain_replay, gelu_schedule[3] if polynomial else 1024,
                ctx.plain_replay, 'polynomial' if polynomial else 'lookup', packed,
                ctx.pointwise_schedule)
        elif ctx.dual_replay:
            from sadl_mlp_dual import backward_replay
            _, gelu, derivative = _lookup_tables[x.device]
            delta, postact = backward_replay(x.contiguous(), w1, saved, gradient, w2,
                                             gelu, derivative, 128, 64, 32, 4, 3, 1, packed)
        else:
            preact = ((configured_gemm(x, w1.T, saved, ctx.plain_replay) if ctx.plain_replay else
                       dense_gemm(x, w1.T, saved) if ctx.dense_backend else
                       torch.nn.functional.linear(x, w1, saved)) if ctx.recompute else saved)
            if ctx.native_derivative:
                projected = gradient @ w2
                delta = torch.ops.aten.gelu_backward.default(projected, preact)
                postact = torch.nn.functional.gelu(preact)
            elif ctx.fused_bias:
                from sadl_mlp_bias import derivative
                delta, postact, db1 = derivative(gradient, w2, preact, ctx.activation)
            else:
                delta, postact = (_gemm_derivative(gradient, w2, preact, ctx.activation) if ctx.tiled else
                                 gemm_dact(gradient, w2, preact, activation=ctx.activation, tuned=False))
        if ctx.parallel_gradients:
            from sadl_mlp_parallel import gradients
            dx, dw1, db1, dw2, db2 = gradients(delta.contiguous(), x.contiguous(),
                postact.contiguous(), gradient, w1, ctx.plain_dx, ctx.parallel_gradients)
        else:
            dx = (configured_gemm(delta, w1, None, ctx.plain_dx) if ctx.plain_dx else
                  dense_gemm(delta, w1, None) if ctx.dense_backend else delta @ w1)
        dx = dx.reshape(ctx.input_shape).to(ctx.input_dtype)
        if ctx.parallel_gradients:
            pass
        elif ctx.te_wgrad:
            from sadl_te_linear import weight_bias_backward
            dw1, db1 = weight_bias_backward(x, w1, saved, delta)
            dw2, db2 = weight_bias_backward(postact, w2, extra[0], gradient)
        else:
            dw1 = (x.T @ delta).T.contiguous() if ctx.wgrad_transpose & 1 else delta.T @ x
            if not ctx.fused_bias:
                db1 = delta.sum(0)
            dw2 = (postact.T @ gradient).T.contiguous() if ctx.wgrad_transpose & 2 else gradient.T @ postact
            db2 = gradient.sum(0)
        result = (dx, *(value.to(dtype) for value, dtype in
                        zip((dw1, db1, dw2, db2), ctx.parameter_dtypes)))
        result = result + (None,) * (len(ctx.needs_input_grad) - 5)
        if ctx.residual_dtype is not None:
            result = result[:-2] + (gradient.reshape(ctx.input_shape).to(ctx.residual_dtype),None)
        return result


def configure_fused_mlp(model, *, lookup=False, tiled=False, recompute=False, dense_backend=False,
                        te_wgrad=False, fused_bias=False, plain_gemm_configs=None,
                        native_derivative=False, dual_replay=False, pointwise_replay=False,
                        parallel_gradients=0, wgrad_transpose=0, packed_lookup=False,
                        pointwise_schedule='', gelu_schedule=''):
    """Use the same erf GELU and FP32 parameters with fused BF16 intermediate ops."""
    selected = []
    gelu_runtime = ()
    if gelu_schedule:
        schedule = json.loads(gelu_schedule)
        if (not lookup or not recompute or not plain_gemm_configs or fused_bias or
                schedule.get('phase') not in ('forward','backward','backward_owned','pointwise_polynomial','dual_polynomial') or
                schedule.get('kind') not in ('lookup','polynomial')):
            raise ValueError('GELU scheduling requires the lookup/recompute/plain GEMM path')
        if schedule['phase']=='pointwise_polynomial' and (
                not pointwise_replay or packed_lookup or pointwise_schedule or
                schedule['kind']!='polynomial' or schedule['configuration'].get('block') not in (256,512,1024,2048)):
            raise ValueError('Polynomial pointwise requires an unpacked, unscheduled replay path')
        dual_config = tuple(schedule['configuration'].get('dual_config',()))
        if schedule['phase']=='dual_polynomial' and (
                schedule['kind']!='polynomial' or native_derivative or pointwise_schedule or
                len(dual_config)!=6 or any(type(value) is not int for value in dual_config) or
                dual_config[0] not in (32,64,128) or dual_config[1] not in (64,128) or
                dual_config[2] not in (32,64) or dual_config[3]!=4 or
                dual_config[4] not in (1,2,3,4) or dual_config[5] not in (1,4,16)):
            raise ValueError('Unsupported numerical dual replay configuration')
        gelu_runtime = (schedule['phase'], schedule['kind'],
                        json.dumps(schedule['configuration'],sort_keys=True),
                        schedule['configuration'].get('block',0),dual_config)
    plain_gemm_configs = plain_gemm_configs or {}
    if wgrad_transpose not in (0, 1, 2, 3) or (wgrad_transpose and (parallel_gradients or te_wgrad)):
        raise ValueError('Weight-gradient layout requires native serial GEMMs and mask 0-3')
    if parallel_gradients not in (0, 1, 2):
        raise ValueError('Expected zero, one or two gradient branches')
    if parallel_gradients:
        if te_wgrad or fused_bias or dense_backend:
            raise ValueError('Parallel gradients require the native weight/bias gradient route')
        from sadl_mlp_parallel import prepare
        prepare(next(model.encoder.parameters()).device)
    if native_derivative and fused_bias:
        raise ValueError('Native derivative and fused bias cannot be combined')
    if dual_replay and (not lookup or not recompute or native_derivative or fused_bias):
        raise ValueError('Dual replay requires lookup/recompute and native bias gradients')
    if pointwise_replay and (not lookup or not recompute or native_derivative or fused_bias or dual_replay):
        raise ValueError('Pointwise replay requires lookup/recompute and its own backward route')
    if packed_lookup and not (pointwise_replay or dual_replay):
        raise ValueError('Packed lookup requires a measured lookup replay implementation')
    if pointwise_schedule:
        if not pointwise_replay or packed_lookup:
            raise ValueError('Replay schedule requires unpacked pointwise replay')
        from sadl_mlp_pointwise import parse_schedule
        parse_schedule(pointwise_schedule)
    if plain_gemm_configs and (dense_backend or te_wgrad):
        raise ValueError('Plain GEMM tuning requires native parameter gradients and replay')
    required = {'fc2', 'input_gradient'}
    if plain_gemm_configs and (not required <= set(plain_gemm_configs)
                              or set(plain_gemm_configs) - required - {'preactivation_replay'}):
        raise ValueError('Both measured plain GEMM configurations are required')
    plain_fc2 = json.dumps(plain_gemm_configs['fc2'], sort_keys=True) if plain_gemm_configs else ''
    plain_dx = json.dumps(plain_gemm_configs['input_gradient'], sort_keys=True) if plain_gemm_configs else ''
    plain_replay = (json.dumps(plain_gemm_configs['preactivation_replay'], sort_keys=True)
                    if 'preactivation_replay' in plain_gemm_configs else '')
    if fused_bias and (not tiled or not recompute or te_wgrad):
        raise ValueError('Fused bias reduction requires tiled replay and native weight gradients')
    if te_wgrad:
        if not recompute:
            raise ValueError('TE weight/bias fusion requires the existing MLP replay path')
        from transformer_engine.pytorch.module.base import get_workspace
        get_workspace()
    activation = prepare_gelu_lookup(torch.device('cuda', torch.cuda.current_device())) if lookup else _ACTIVATION
    if packed_lookup:
        from sadl_gelu_lookup import pack
        device = torch.device('cuda', torch.cuda.current_device())
        if device not in _packed_lookup_tables:
            _packed_lookup_tables[device] = pack(*_lookup_tables[device][1:])
    # The last CLS-only layer is too small to benefit from these GEMM epilogues.
    for index, layer in enumerate(model.encoder.layers[:-1]):
        mlp = layer.mlp
        if mlp.config.hidden_act != 'gelu' or mlp.fc1.bias is None or mlp.fc2.bias is None:
            raise ValueError('Fused MLP requires the existing biased erf-GELU ViT MLP')
        original = mlp.forward

        def forward(self, x, residual=None, _original=original):
            if (not x.is_cuda or not torch.is_autocast_enabled('cuda')
                    or torch.get_autocast_dtype('cuda') != torch.bfloat16):
                result = _original(x)
                return result if residual is None else result+residual
            return _FusedMLP.apply(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
                                   activation, tiled, recompute, dense_backend, te_wgrad, fused_bias,
                                   plain_fc2, plain_dx, plain_replay, native_derivative, dual_replay,
                                   pointwise_replay, parallel_gradients, wgrad_transpose, packed_lookup,
                                   pointwise_schedule,gelu_runtime,residual,getattr(self,'_exec_residual_configuration',''))

        mlp.forward = MethodType(forward, mlp)
        mlp._exec_residual_forward = mlp.forward
        selected.append(index)
    model._exec_fused_mlp = dict(layers=selected, lookup=lookup, tiled=tiled, recompute=recompute,
                                dense_backend='quack' if dense_backend else 'torch',te_wgrad=te_wgrad,
                                fused_bias=fused_bias, plain_gemm_configs=plain_gemm_configs,
                                native_derivative=native_derivative, dual_replay=dual_replay,
                                pointwise_replay=pointwise_replay,
                                parallel_gradients=parallel_gradients, wgrad_transpose=wgrad_transpose,
                                packed_lookup=packed_lookup, pointwise_schedule=pointwise_schedule,
                                gelu_schedule=gelu_schedule)
    return selected


def load_plain_gemm_configs(path, replay_path=None):
    configs = {}
    reports = [(path, ('fc2', 'input_gradient'))]
    if replay_path:
        reports.append((replay_path, ('preactivation_replay',)))
    for report_path, phases in reports:
        tuning = json.loads(Path(report_path).read_text())
        if not tuning['complete'] or not tuning['finished'] or tuning['gpu'] != torch.cuda.get_device_name():
            raise ValueError('A completed tuning report from this GPU model is required')
        for phase in phases:
            row = tuning['best'][phase]
            if not row or not row['passed']:
                raise ValueError(f'No passing configuration for {phase}')
            configs[phase] = row['config']
    return configs


@torch.library.custom_op('sadl_mlp::dense_gemm', mutates_args=())
def dense_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    output = a.new_empty((a.shape[0], b.shape[1]))
    gemm_tuned.fn(a, b, output, bias=bias, config=default_config(a.device))
    return output


@dense_gemm.register_fake
def _dense_fake(a, b, bias):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op('sadl_mlp::configured_gemm', mutates_args=())
def configured_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None,
                    configuration: str) -> torch.Tensor:
    base = default_config(a.device)
    values = json.loads(configuration)
    if values['device_capacity'] != base.device_capacity:
        raise ValueError('Plain GEMM configuration belongs to a different GPU architecture')
    output = a.new_empty((a.shape[0], b.shape[1]))
    gemm_tuned.fn(a, b, output, bias=bias, config=replace(base, **values))
    return output


@configured_gemm.register_fake
def _configured_fake(a, b, bias, configuration):
    return a.new_empty((a.shape[0], b.shape[1]))


def _dense_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.has_bias = inputs[2] is not None


def _dense_backward(ctx, gradient):
    a, b = ctx.saved_tensors
    return gradient @ b.T, a.T @ gradient, gradient.sum(0) if ctx.has_bias else None


dense_gemm.register_autograd(_dense_backward, setup_context=_dense_setup)
