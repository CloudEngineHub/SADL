"""Encoder and SIGReg execution paths; objective and parameters are unchanged."""
from types import MethodType

import torch


def _moments(projected, knots, phi, weights):
    phases = projected.unsqueeze(-1) * knots
    cosine = phases.cos().mean(-3)
    sine = phases.sin().mean(-3)
    loss = (((cosine - phi).square() + sine.square()) @ weights)
    return (loss * projected.size(-2)).mean(), cosine, sine


def _moment_backward(projected, knots, phi, weights, cosine, sine, gradient):
    phases = projected.unsqueeze(-1) * knots
    derivative = (-(cosine - phi).unsqueeze(-3) * phases.sin()
                  + sine.unsqueeze(-3) * phases.cos())
    scale = 2.0 / cosine[..., 0].numel()
    return ((derivative * (knots * weights)).sum(-1) * (gradient * scale)).to(projected.dtype)


class _StreamingStatistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, knots, phi, weights, forward, backward):
        loss, cosine, sine = forward(projected, knots, phi, weights)
        ctx.save_for_backward(projected, knots, phi, weights, cosine, sine)
        ctx.backward_kernel = backward
        return loss

    @staticmethod
    def backward(ctx, gradient):
        projected, knots, phi, weights, cosine, sine = ctx.saved_tensors
        result = ctx.backward_kernel(projected, knots, phi, weights, cosine, sine, gradient)
        return result, None, None, None, None, None


def configure_streaming_sigreg(regularizer, *, compile_kernels=True):
    """Save projected samples and moments, recomputing trigonometry in backward."""
    from module import SIGReg
    if type(regularizer) is not SIGReg:
        raise ValueError('Streaming SIGReg requires the native SIGReg implementation')
    if any(buffer.requires_grad for buffer in (regularizer.t, regularizer.phi, regularizer.weights)):
        raise ValueError('Quadrature buffers must be fixed')
    options = {'emulate_precision_casts': True, 'fallback_random': True}
    forward = torch.compile(_moments, fullgraph=True, dynamic=False, options=options) if compile_kernels else _moments
    backward = torch.compile(_moment_backward, fullgraph=True, dynamic=False, options=options) if compile_kernels else _moment_backward

    def run(self, proj):
        # Keep sampling, normalization and the autocast projection identical to SIGReg.
        values = proj.float()
        directions = torch.randn(values.size(-1), self.num_proj, device=proj.device, dtype=torch.float32)
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-8)
        projected = values @ directions
        return _StreamingStatistic.apply(projected, self.t, self.phi, self.weights,
                                         forward, backward).to(proj.dtype)

    regularizer.forward = MethodType(run, regularizer)
    return regularizer


def predict_all_windows(model, embeddings, action_embeddings, history_size):
    """One predictor batch, with the original ordered projection/BatchNorm calls."""
    from collapse_study.sdpa import sdpa_context
    transitions = embeddings.size(1) - 1
    if transitions < 1:
        raise ValueError('A training window must contain at least two frames')
    prefix = min(history_size, transitions)
    starts = [0, *range(1, transitions - prefix + 1)]
    contexts = torch.cat([embeddings[:, start:start + prefix] for start in starts])
    actions = torch.cat([action_embeddings[:, start:start + prefix] for start in starts])
    with sdpa_context(model.predictor_sdpa_backend):
        raw = model.predictor(contexts, actions)
    grouped_projection = getattr(model, '_exec_window_projection', None)
    projected_windows = (grouped_projection(raw.reshape(len(starts), -1, raw.size(-1)))
                         .reshape(len(starts), embeddings.size(0), prefix, -1)
                         if grouped_projection is not None else None)
    predictions = []
    for index, (value, context) in enumerate(zip(raw.split(embeddings.size(0)), contexts.split(embeddings.size(0)))):
        projected = (projected_windows[index] if projected_windows is not None else
                     model.pred_proj(value.reshape(-1, value.size(-1))).reshape_as(value))
        if model.predict_residual:
            projected = context + projected
        predictions.append(projected if index == 0 else projected[:, -1:])
    return torch.cat(predictions, dim=1)


def configure_async_checks(train):
    """Retain action validation without waiting for two GPU scalar reads per step."""
    import os
    original = train.intact_forward
    os.environ['INTACT_VALIDATE_BATCH'] = '0'

    def forward(self, batch, *args, **kwargs):
        torch._assert_async(torch.isfinite(batch['action']).all(), 'Target actions must be finite')
        torch._assert_async(torch.isfinite(batch['previous_action']).all(), 'Previous actions must be finite')
        return original(self, batch, *args, **kwargs)

    train.intact_forward = forward


def configure_capacity_async_checks():
    """Keep the positive-determinant check without copying its scalar to CPU."""
    from collapse_study import regularizers

    def check(sign, message):
        torch._assert_async((sign > 0).all(), message)

    regularizers._check_positive_slogdet = check


def configure_graph_checkpoint_encoder(model):
    """Capture block replay with backward so AOT can eliminate redundant intermediates."""
    from native_execution import _flash_layer_forward
    from infra_execution_shared import _cls_tail_forward
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint
    router = model._encoder_sdpa_router
    if set(router.backends) != {'flash'} or model.frame_readout != 'cls' or model._capture_block_outputs:
        raise ValueError('Graph replay requires the all-Flash CLS-only encoder')
    if any(layer.dropout.p or layer.attention.attention_dropout for layer in router.layers):
        raise ValueError('RNG-free replay requires dropout-free encoder blocks')
    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    # Each checkpoint closure owns a distinct block; train/eval specialize separately.
    limit_name = ('recompile_limit' if hasattr(torch._dynamo.config, 'recompile_limit')
                  else 'cache_size_limit')
    setattr(torch._dynamo.config, limit_name,
            max(getattr(torch._dynamo.config, limit_name), 2 * len(router.layers) + 4))
    router.close()
    for index, layer in enumerate(model.encoder.layers):
        layer.gradient_checkpointing = False
        if index == len(model.encoder.layers) - 1:
            def cls(self, *args, **kwargs):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return _cls_tail_forward(self, *args, **kwargs)
            forward = MethodType(cls, layer)
        else:
            forward = MethodType(_flash_layer_forward, layer)

        def replay(*args, _forward=forward, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False,
                                  preserve_rng_state=False, **kwargs)
            return _forward(*args, **kwargs)

        layer.forward = torch.compile(replay, dynamic=False,
            options={'emulate_precision_casts': True, 'fallback_random': True})
        layer._exec_graph_checkpoint = True
    if schema != [(name, tuple(value.shape)) for name, value in model.state_dict().items()]:
        raise RuntimeError('Graph checkpointing changed the parameter/buffer schema')


def configure_whole_encoder(model, *, replay='none', cuda_graphs=True, autotune=False,
                            bf16_residual=False, attention='native', precision_casts=True,
                            retained_layers=(), checkpoint_group_size=1, save_norm=False,
                            packed_qkv=False, frame_group_size=0, retention_policy='none',
                            retain_norm_statistics=False, compile_group_size=0):
    """Compile the dropout-free transformer as one graph with explicit replay policy."""
    from functools import partial
    from infra_execution_shared import _cls_tail_forward, _save_attention, _save_compute
    from native_execution import _flash_layer_forward
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    encoder = model.encoder
    if frame_group_size < 0 or frame_group_size and (replay != 'none' or retained_layers):
        raise ValueError('Frame groups require nonnegative size and no nested block replay')
    if frame_group_size and (attention != 'fa4' or checkpoint_group_size != 1 or packed_qkv):
        raise ValueError('Frame groups require ordinary unpacked FA4 blocks')
    if frame_group_size and any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                                for module in encoder.modules()):
        raise ValueError('Frame grouping cannot split encoder BatchNorm statistics')
    router = model._encoder_sdpa_router
    layers = tuple(router.layers)
    if compile_group_size < 0 or compile_group_size > len(layers):
        raise ValueError('Compile group size must fit the encoder')
    if compile_group_size and (frame_group_size or checkpoint_group_size != 1):
        raise ValueError('Compile groups require the existing per-layer replay boundaries')
    if packed_qkv and (not attention.startswith('fa4') or any(
            projection.bias is None for layer in layers for projection in
            (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj))):
        raise ValueError('Packed FA4 requires biased Q/K/V projections')
    if not 1 <= checkpoint_group_size <= len(layers):
        raise ValueError('Checkpoint group size must fit the encoder')
    if checkpoint_group_size != 1 and retained_layers:
        raise ValueError('Grouped replay does not accept separately retained layers')
    retained_layers = tuple(retained_layers)
    if retention_policy not in ('none', 'all', 'compute'):
        raise ValueError(retention_policy)
    if retention_policy != 'none' and (not retained_layers or replay == 'none'):
        raise ValueError('Selective retention requires retained layers and checkpoint replay')
    if len(set(retained_layers)) != len(retained_layers) or any(
            index < 0 or index >= len(layers) for index in retained_layers):
        raise ValueError('Retained block indices must be unique encoder layer indices')
    if replay not in ('none', 'attention', 'compute', 'mixed'):
        raise ValueError(replay)
    if attention not in ('native', 'portable', 'fa4', 'fa4-partition', 'fa4-partition-manual', 'fa4-tiled', 'fa4-scheduled', 'fa4-scheduled-forward', 'fa4-cudnn-tail'):
        raise ValueError(attention)
    if attention.startswith('fa4') and replay == 'compute':
        raise ValueError('FA4 execution supports none, attention or mixed replay')
    if save_norm and attention != 'fa4':
        raise ValueError('Norm retention requires ordinary FA4 attention')
    if set(router.backends) != {'flash'} or model.frame_readout != 'cls' or encoder.pooler is not None:
        raise ValueError('Whole-encoder execution requires all-Flash CLS output without a pooler')
    if model._capture_block_outputs or encoder.embeddings.dropout.p or any(
            layer.dropout.p or layer.attention.attention_dropout for layer in layers):
        raise ValueError('Whole-encoder execution requires dropout-free blocks without diagnostic hooks')
    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    router.close()
    attention_impl = None
    if attention.startswith('fa4'):
        from fa4_execution import layer_forward, save_attention
        if attention == 'fa4-partition':
            from fa4_partition import attention as attention_impl, save_attention
        elif attention == 'fa4-partition-manual':
            from fa4_partition import attention_manual as attention_impl, save_attention
        elif attention == 'fa4-tiled':
            from fa4_tiled import attention as attention_impl, save_attention
        elif attention == 'fa4-scheduled':
            from fa4_scheduled import attention as attention_impl, save_attention
        elif attention == 'fa4-scheduled-forward':
            from fa4_scheduled_forward import attention as attention_impl, save_attention
        elif attention == 'fa4-cudnn-tail':
            from fa4_execution import attention_cudnn_tail as attention_impl
        if save_norm:
            from sadl_norm import save_attention_and_norm as save_attention
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = False
        if attention.startswith('fa4'):
            layer.forward = partial(layer_forward, layer, cls_only=index == len(layers)-1,
                                    attention_impl=attention_impl, packed_qkv=packed_qkv)
        elif attention == 'portable':
            forward = _cls_tail_forward if index == len(layers)-1 else type(layer).forward
            layer.forward = MethodType(forward, layer)
        else:
            forward = _cls_tail_forward if index == len(layers)-1 else _flash_layer_forward
            layer.forward = MethodType(forward, layer)
    base_policy = (save_attention if attention.startswith('fa4') else
                   _save_compute if replay == 'compute' else _save_attention)
    residual_policy = getattr(model,'_exec_residual_policy','prefer')
    if residual_policy != 'prefer':
        if residual_policy not in ('recompute','save','save_mlp','save_attention'):
            raise ValueError('Unexpected residual checkpoint policy')
        original_policy = base_policy
        def base_policy(ctx,op,*args,**kwargs):
            if op == torch.ops.sadl_residual_gemm.forward.default:
                retain=(residual_policy=='save' or
                        residual_policy=='save_mlp' and args[0].shape[1]==768 or
                        residual_policy=='save_attention' and args[0].shape[1]==192)
                return CheckpointPolicy.MUST_SAVE if retain else CheckpointPolicy.MUST_RECOMPUTE
            return original_policy(ctx,op,*args,**kwargs)

    def statistic_policy(ctx, op, *args, **kwargs):
        if op in (torch.ops.sadl_norm_statistics.rows.default,
                  torch.ops.sadl_norm_statistics.native_rows.default):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def augmented_policy(ctx, op, *args, **kwargs):
        if statistic_policy(ctx, op, *args, **kwargs) == CheckpointPolicy.MUST_SAVE:
            return CheckpointPolicy.MUST_SAVE
        return base_policy(ctx, op, *args, **kwargs)

    if retain_norm_statistics:
        from sadl_norm_statistics import row_statistics
        if replay == 'none' or frame_group_size:
            raise ValueError('Row statistic retention requires block replay')
    context = partial(create_selective_checkpoint_contexts,
                      augmented_policy if retain_norm_statistics else base_policy)
    statistic_context = partial(create_selective_checkpoint_contexts, statistic_policy)

    def retain_compute(ctx, op, *args, **kwargs):
        if retention_policy == 'all' or op in {
                torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                torch.ops.aten.bmm.default}:
            return CheckpointPolicy.MUST_SAVE
        return (save_attention if attention.startswith('fa4') else _save_attention)(
            ctx, op, *args, **kwargs)

    retained_context = partial(create_selective_checkpoint_contexts, retain_compute)
    groups = []
    for offset in range(0, len(layers), checkpoint_group_size):
        members = layers[offset:offset + checkpoint_group_size]

        def group(hidden, members=members):
            for layer in members:
                hidden = layer.forward(hidden)
            return hidden

        groups.append(members[0].forward if checkpoint_group_size == 1 else group)

    def block_stack(hidden, parameter_cast_key=-1, start=0, stop=None):
        stop = len(groups) if stop is None else stop
        backends = SDPBackend.FLASH_ATTENTION
        with sdpa_kernel(backends):
            for index in range(start, stop):
                group = groups[index]
                if replay != 'none' and torch.is_grad_enabled() and (
                        index not in retained_layers or retention_policy != 'none'):
                    checkpoint_options = ({'context_fn': retained_context}
                        if index in retained_layers else
                        ({'context_fn': statistic_context} if retain_norm_statistics else {})
                        if replay == 'mixed' and index % 2 else {'context_fn': context})
                    hidden = checkpoint(group, hidden, use_reentrant=False,
                                        preserve_rng_state=False, **checkpoint_options)
                else:
                    hidden = (group(hidden, parameter_cast_key=parameter_cast_key)
                              if parameter_cast_key >= 0 else group(hidden))
        return encoder.layernorm(hidden) if stop == len(groups) else hidden

    def blocks(hidden):
        if not frame_group_size:
            return block_stack(hidden)
        output = []
        for index, chunk in enumerate(hidden.split(frame_group_size)):
            if torch.is_grad_enabled():
                value = checkpoint(block_stack, chunk, index, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = block_stack(chunk)
            output.append(value)
        return torch.cat(output, dim=0)

    compile_options = {
        'emulate_precision_casts': precision_casts, 'fallback_random': True,
        'triton.cudagraphs': cuda_graphs, 'max_autotune': autotune}
    if compile_group_size:
        segments = [torch.compile(partial(block_stack, start=start,
                    stop=min(start + compile_group_size, len(groups))),
                    dynamic=False, fullgraph=True, options=compile_options)
                    for start in range(0, len(groups), compile_group_size)]

        def compiled(hidden):
            for segment in segments:
                hidden = segment(hidden)
                if cuda_graphs:
                    hidden = hidden.clone()
            return hidden
    else:
        compiled = torch.compile(blocks, dynamic=False, fullgraph=True, options=compile_options)

    def forward(pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None,
                attention_mask=None, **kwargs):
        if model._capture_block_outputs or bool_masked_pos is not None or attention_mask is not None or kwargs:
            raise ValueError('Whole encoder supports the unmasked CLS training contract only')
        pixels = (pixel_values if getattr(model, '_exec_uint8_patch', False) and pixel_values.dtype == torch.uint8
                  else pixel_values.to(encoder.embeddings.patch_embeddings.projection.weight.dtype))
        hidden = encoder.embeddings(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        if bf16_residual and torch.is_autocast_enabled('cuda'):
            hidden = hidden.to(torch.get_autocast_dtype('cuda'))
        hidden = compiled(hidden)
        if cuda_graphs:
            hidden = hidden.clone()
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=None)

    encoder.forward = forward
    model._exec_whole_encoder = dict(replay=replay, cuda_graphs=cuda_graphs,
                                     autotune=autotune, bf16_residual=bf16_residual,
                                     attention=attention, precision_casts=precision_casts,
                                     retained_layers=retained_layers,
                                     retention_policy=retention_policy,
                                     checkpoint_group_size=checkpoint_group_size, save_norm=save_norm,
                                     packed_qkv=packed_qkv, frame_group_size=frame_group_size,
                                     frame_group_fp32_grad=bool(frame_group_size))
    model._exec_whole_encoder['retain_norm_statistics'] = retain_norm_statistics
    model._exec_whole_encoder['compile_group_size'] = compile_group_size
    if schema != [(name, tuple(value.shape)) for name, value in model.state_dict().items()]:
        raise RuntimeError('Whole encoder changed the parameter/buffer schema')


def configure_frame_replay(model, frames):
    """Checkpoint independent image groups; projector and loss keep the full batch."""
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    if frames < 1:
        raise ValueError('The frame chunk must be positive')
    encoder = model.encoder
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in encoder.modules()):
        raise ValueError('Frame replay cannot split encoder BatchNorm statistics')
    if encoder.embeddings.dropout.p or any(
            layer.dropout.p or layer.attention.attention_dropout for layer in encoder.layers):
        raise ValueError('Frame replay requires a dropout-free encoder')
    original = encoder.forward

    def forward(pixel_values=None, **kwargs):
        def encode(pixels):
            return original(pixel_values=pixels, **kwargs).last_hidden_state
        chunks = []
        for pixels in pixel_values.split(frames):
            if torch.is_grad_enabled():
                value = checkpoint(encode, pixels, use_reentrant=False, preserve_rng_state=False)
            else:
                value = encode(pixels)
            chunks.append(value)
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(chunks), pooler_output=None)

    encoder.forward = forward
    model._exec_frame_replay = frames
