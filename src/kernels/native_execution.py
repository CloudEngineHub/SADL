"""Shared execution kernels for INTACT / SADL training."""
import json
import os
from pathlib import Path
from types import MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def predict_adjacent_serial(model, embeddings, action_embeddings, history_size):
    """Keep per-window predictor and BatchNorm calls."""
    transitions = embeddings.size(1) - 1
    prefix = min(history_size, transitions)
    predictions = [model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])]
    for index in range(prefix, transitions):
        start = max(0, index - history_size + 1)
        predictions.append(model.predict(embeddings[:, start:index + 1],
                                         action_embeddings[:, start:index + 1])[:, -1:])
    return torch.cat(predictions, dim=1)


def predict_adjacent_window_bn(model, embeddings, action_embeddings, history_size):
    """Batch Predictor calls while preserving each projection BatchNorm group."""
    transitions = embeddings.size(1) - 1
    prefix = min(history_size, transitions)
    predictions = [model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])]
    if transitions == prefix:
        return predictions[0]
    indices = range(prefix, transitions)
    contexts = torch.cat([embeddings[:, i - history_size + 1:i + 1] for i in indices])
    actions = torch.cat([action_embeddings[:, i - history_size + 1:i + 1] for i in indices])
    from collapse_study.sdpa import sdpa_context
    with sdpa_context(model.predictor_sdpa_backend):
        raw = model.predictor(contexts, actions)
    batch = embeddings.size(0)
    for prediction, context in zip(raw.split(batch), contexts.split(batch)):
        projected = model.pred_proj(prediction.reshape(-1, prediction.size(-1)))
        projected = projected.reshape_as(prediction)
        if model.predict_residual:
            projected = context + projected
        predictions.append(projected[:, -1:])
    return torch.cat(predictions, dim=1)


def _flash_layer_forward(self, *args, **kwargs):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return type(self).forward(self, *args, **kwargs)


def _flash_checkpoint_layer_forward(self, *args, **kwargs):
    from infra_execution_shared import _checkpoint_mlp_forward
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return _checkpoint_mlp_forward(self, *args, **kwargs)


def configure_encoder_flash(model, *, checkpoint_mlp=False, shared_forward=False):
    before = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    for layer in model.encoder.layers:
        assert layer.dropout.p == 0 and layer.attention.attention_dropout == 0
        if shared_forward:
            implementation = _flash_checkpoint_layer_forward if checkpoint_mlp else _flash_layer_forward
            layer.forward = MethodType(implementation, layer)
            continue
        original = layer.forward
        if checkpoint_mlp:
            from infra_execution_shared import _checkpoint_mlp_forward
            original = MethodType(_checkpoint_mlp_forward, layer)
        def forward(*args, _original=original, **kwargs):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return _original(*args, **kwargs)
        layer.forward = forward
    assert before == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def configure_encoder_flash_fp16(model):
    """Keep model dtypes unchanged; use FP16 inside the encoder SDPA kernel."""
    return _configure_encoder_attention_precision(model, SDPBackend.FLASH_ATTENTION, torch.float16, 'infra_flash_fp16')


def configure_encoder_efficient_fp32(model):
    """Match Math's FP32 attention intermediates without materializing attention."""
    return _configure_encoder_attention_precision(model, SDPBackend.EFFICIENT_ATTENTION, torch.float32, 'infra_efficient_fp32')


def configure_encoder_execution(model, *, checkpoint_mlp=False, compile_layers=False,
                                compile_whole=False, cls_tail=False, flash=False):
    """Optimize block execution while retaining the configured attention interface."""
    before = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    if compile_layers and compile_whole:
        raise ValueError('Choose block compilation or whole encoder compilation')
    if cls_tail and getattr(model, 'frame_readout', 'cls') != 'cls':
        raise ValueError('CLS tail requires the fixed CLS frame readout')
    for index, layer in enumerate(model.encoder.layers):
        assert layer.dropout.p == 0 and layer.attention.attention_dropout == 0
        if cls_tail and index == len(model.encoder.layers) - 1:
            from infra_execution_shared import _cls_tail_forward

            def cls_forward(self, *args, **kwargs):
                if flash:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        return _cls_tail_forward(self, *args, **kwargs)
                return _cls_tail_forward(self, *args, **kwargs)

            layer.forward = MethodType(cls_forward, layer)
            layer._exec_cls_tail = True
        elif checkpoint_mlp:
            from infra_execution_shared import _checkpoint_mlp_forward
            if flash:
                def checkpoint_forward(self, *args, **kwargs):
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        return _checkpoint_mlp_forward(self, *args, **kwargs)
                layer.forward = MethodType(checkpoint_forward, layer)
            else:
                layer.forward = MethodType(_checkpoint_mlp_forward, layer)
        if compile_layers:
            layer.forward = torch.compile(layer.forward, dynamic=False,
                options={'emulate_precision_casts': True, 'fallback_random': True})
    if compile_whole:
        # Dynamo materializes the all-valid mask that eager SDPA normally omits.
        # Keep the original mask builder eager so Flash receives the same None.
        import transformers.models.vit.modeling_vit as vit
        if not getattr(vit.create_bidirectional_mask, '_exec_eager_mask', False):
            mask_builder = torch.compiler.disable(vit.create_bidirectional_mask)
            mask_builder._exec_eager_mask = True
            vit.create_bidirectional_mask = mask_builder
        model.encoder.forward = torch.compile(model.encoder.forward, dynamic=False,
            options={'emulate_precision_casts': True, 'fallback_random': True})
    assert before == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def configure_encoder_native_graph(model, *, cuda_graphs=True):
    """Capture the dropout-free native encoder with its original norm kernels."""
    from flash_native_ops import configure_native_encoder_ops
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    encoder = model.encoder
    router = model._encoder_sdpa_router
    layers = tuple(router.layers)
    if set(router.backends) != {'flash'} or encoder.pooler is not None:
        raise ValueError('Native encoder graphs require all-Flash attention without a pooler')
    if model._capture_block_outputs or encoder.embeddings.dropout.p or any(
        layer.dropout.p or layer.attention.attention_dropout for layer in layers
    ):
        raise ValueError('Native encoder graphs require dropout-free blocks without diagnostics')
    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    configure_native_encoder_ops(model)
    router.close()

    def encode(pixel_values, interpolate_pos_encoding):
        pixels = pixel_values.to(encoder.embeddings.patch_embeddings.projection.weight.dtype)
        hidden = encoder.embeddings(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for layer in layers:
                hidden = layer(hidden)
        return encoder.layernorm(hidden)

    compiled = torch.compile(encode, dynamic=False, options={
        'emulate_precision_casts': True, 'fallback_random': True,
        'triton.cudagraphs': cuda_graphs})

    def forward(pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None,
                attention_mask=None, **kwargs):
        if model._capture_block_outputs or bool_masked_pos is not None or attention_mask is not None or kwargs:
            raise ValueError('Native encoder graphs support the unmasked training contract only')
        with torch.random.fork_rng(devices=[]):
            hidden = compiled(pixel_values, interpolate_pos_encoding)
        if cuda_graphs:
            hidden = hidden.clone()
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=None)

    encoder.forward = forward
    assert schema == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def configure_encoder_cls_tail(model):
    """Skip final-block patch outputs for the fixed CLS-only SADL objective."""
    from infra_execution_shared import _cls_tail_forward

    assert model.frame_readout == 'cls' and not model._capture_block_outputs
    assert os.environ.get('INTACT_SKIP_UNUSED_PATCH_MEAN') == '1'
    layer = model.encoder.layers[-1]
    assert layer.dropout.p == 0 and layer.attention.attention_dropout == 0
    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    layer.forward = MethodType(_cls_tail_forward, layer)
    layer._exec_cls_tail = True
    assert schema == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def configure_encoder_mixed_checkpoint(model, mlp_layers):
    """Retain attention activations in selected layers and replay only their MLP."""
    from infra_execution_shared import _checkpoint_mlp_forward

    layers = model.encoder.layers
    eligible = [index for index, layer in enumerate(layers)
                if not getattr(layer, '_exec_cls_tail', False)]
    if not 0 <= mlp_layers <= len(eligible):
        raise ValueError('MLP replay layer count exceeds eligible encoder layers')
    selected = eligible[-mlp_layers:] if mlp_layers else []
    for index in selected:
        layer = layers[index]
        if not layer.gradient_checkpointing or layer.dropout.p or layer.attention.attention_dropout:
            raise ValueError('Mixed replay requires checkpointed, dropout-free encoder layers')
    for index in selected:
        layer = layers[index]
        layer.gradient_checkpointing = False
        layer.forward = MethodType(_checkpoint_mlp_forward, layer)
    return selected


def configure_encoder_attention_checkpoint(model):
    """Cache the Flash attention result while replaying the surrounding block."""
    from functools import partial
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
    from infra_execution_shared import _save_attention

    if set(model._encoder_sdpa_router.backends) != {'flash'}:
        raise ValueError('Attention caching requires the all-Flash encoder contract')
    layers = [layer for layer in model.encoder.layers if not getattr(layer, '_exec_cls_tail', False)]
    if any(not layer.gradient_checkpointing or layer.dropout.p or layer.attention.attention_dropout
           for layer in layers):
        raise ValueError('Attention caching requires checkpointed, dropout-free encoder layers')
    for layer in layers:
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False,
            preserve_rng_state=False,
            context_fn=partial(create_selective_checkpoint_contexts, _save_attention))
    return len(layers)


def configure_checkpoint_rng(model):
    """Omit RNG snapshots only for the deterministic ViT checkpoint path."""
    encoder = model.encoder
    if not encoder.is_gradient_checkpointing:
        raise ValueError('Encoder checkpointing must already be enabled')
    if encoder.embeddings.dropout.p or any(layer.dropout.p or layer.attention.attention_dropout
                                          for layer in encoder.layers):
        raise ValueError('RNG-free replay requires zero encoder dropout')
    settings = [dict(layer._gradient_checkpointing_func.keywords) for layer in encoder.layers]
    if not all(item == settings[0] for item in settings):
        raise ValueError('Checkpoint policies differ between encoder blocks')
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        **settings[0], 'preserve_rng_state': False})
    return model


def configure_encoder_segments(model, group_size=3, *, compile_segments=True):
    """Compile dropout-free encoder segments with unchanged parameter ownership."""
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    encoder = model.encoder
    router = model._encoder_sdpa_router
    layers = tuple(router.layers)
    if group_size < 1 or group_size > len(layers):
        raise ValueError('Invalid encoder segment size')
    if set(router.backends) != {'flash'} or encoder.pooler is not None:
        raise ValueError('Segmented encoder requires all-Flash attention without a pooler')
    if model._capture_block_outputs or encoder.embeddings.dropout.p or any(
        layer.dropout.p or layer.attention.attention_dropout for layer in layers
    ):
        raise ValueError('Segmented encoder requires dropout-free blocks without diagnostics')
    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    router.close()
    segments = []
    for offset in range(0, len(layers), group_size):
        blocks = layers[offset:offset + group_size]

        def segment(hidden, blocks=blocks):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for layer in blocks:
                    hidden = layer.forward(hidden)
            return hidden

        segments.append(torch.compile(segment, dynamic=False,
            options={'emulate_precision_casts': True, 'fallback_random': True}) if compile_segments else segment)

    def forward(pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None,
                attention_mask=None, **kwargs):
        if model._capture_block_outputs or bool_masked_pos is not None or attention_mask is not None or kwargs:
            raise ValueError('Segmented encoder supports the unmasked training contract only')
        pixels = pixel_values.to(encoder.embeddings.patch_embeddings.projection.weight.dtype)
        hidden = encoder.embeddings(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        for segment in segments:
            if encoder.training and torch.is_grad_enabled():
                # Direct forward calls bypass per-layer checkpoint wrappers;
                # each segment has one replay boundary and no stochastic ops.
                hidden = checkpoint(segment, hidden, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = segment(hidden)
        return BaseModelOutputWithPooling(last_hidden_state=encoder.layernorm(hidden), pooler_output=None)

    encoder.forward = forward
    assert schema == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


class _PackedQKVSeparateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pixels, qw, kw, vw, qb, kb, vb):
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else pixels.dtype
        ctx.input_dtype = pixels.dtype
        ctx.parameter_dtype = qw.dtype
        ctx.has_bias = qb is not None
        values = [value.to(dtype) for value in (pixels, qw, kw, vw)]
        ctx.save_for_backward(*values)
        weight = torch.cat(values[1:], dim=0)
        bias = torch.cat([value.to(dtype) for value in (qb, kb, vb)]) if ctx.has_bias else None
        return torch.nn.functional.linear(values[0], weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        pixels, *weights = ctx.saved_tensors
        flat = pixels.reshape(-1, pixels.shape[-1])
        gradients = gradient.reshape(-1, gradient.shape[-1]).chunk(3, dim=-1)
        # Separate Linear nodes round each input gradient before accumulating
        # in the original input dtype. A packed GEMM otherwise changes that order.
        with torch.autocast('cuda', enabled=False):
            inputs = [(part @ weight).to(ctx.input_dtype) for part, weight in zip(gradients, weights)]
            grad_input = ((inputs[2] + inputs[1]) + inputs[0]).reshape(pixels.shape)
            grad_weights = [(part.transpose(0, 1) @ flat).to(ctx.parameter_dtype) for part in gradients]
            grad_bias = [part.sum(dim=0).to(ctx.parameter_dtype) for part in gradients] if ctx.has_bias else [None] * 3
        return grad_input, *grad_weights, *grad_bias


def _packed_attention_separate_backward(attention, hidden_states, attention_mask=None, **kwargs):
    if attention_mask is not None or kwargs:
        raise ValueError('Packed QKV supports the unmasked ViT only')
    q, k, v = attention.q_proj, attention.k_proj, attention.v_proj
    batch, length = hidden_states.shape[:2]
    heads, width = attention.num_attention_heads, attention.head_dim
    qkv = _PackedQKVSeparateBackward.apply(hidden_states, q.weight, k.weight, v.weight,
                                          q.bias, k.bias, v.bias).view(batch, length, 3, heads, width)
    query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(2))
    output = torch.nn.functional.scaled_dot_product_attention(query, key, value,
        dropout_p=0.0, scale=attention.scaling)
    output = output.transpose(1, 2).reshape(batch, length, heads * width).contiguous()
    return attention.o_proj(output), None


def configure_encoder_packed_qkv(model, *, separate_backward=False):
    """Use one projection GEMM while keeping Q/K/V parameters independent."""
    from infra_execution_shared import _packed_attention_forward

    schema = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    for layer in model.encoder.layers:
        assert layer.dropout.p == 0 and layer.attention.attention_dropout == 0
        implementation = _packed_attention_separate_backward if separate_backward else _packed_attention_forward
        layer.attention.forward = MethodType(implementation, layer.attention)
    assert schema == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def _configure_encoder_attention_precision(model, backend, compute_dtype, interface_name):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, **kwargs):
        dtype = query.dtype
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(compute_dtype)
        with torch.autocast('cuda', enabled=False), sdpa_kernel(backend):
            result, weights = sdpa_attention_forward(
                module, query.to(compute_dtype), key.to(compute_dtype), value.to(compute_dtype),
                attention_mask, **kwargs)
        return result.to(dtype), weights

    before = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    ALL_ATTENTION_FUNCTIONS.register(interface_name, attention)
    for layer in model.encoder.layers:
        assert layer.dropout.p == 0 and layer.attention.attention_dropout == 0
        assert layer.attention.config._attn_implementation in ('sdpa', interface_name)
        layer.attention.config._attn_implementation = interface_name
    assert before == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    return model


def configure_auxiliary(model, *, cuda_graphs=False, compile_projection=True):
    before = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    options = {'fallback_random': True, 'emulate_precision_casts': True}
    if cuda_graphs:
        options['triton.cudagraphs'] = True
    names = ['predictor', 'intent_actor', 'action_encoder']
    if compile_projection:
        names.append('pred_proj')
    for name in names:
        module = getattr(model, name)
        if module is None:
            continue
        module.forward = torch.compile(module.forward, dynamic=False, options=options)
    assert before == [(name, tuple(value.shape)) for name, value in model.state_dict().items()]


def _native_sigreg_statistic(proj32, directions, knots, phi, weights):
    phases = (proj32 @ directions).unsqueeze(-1) * knots
    error = (phases.cos().mean(-3) - phi).square() + phases.sin().mean(-3).square()
    return ((error @ weights) * proj32.size(-2)).mean()


def _native_sigreg_forward(self, proj):
    proj32 = proj.float()
    directions = torch.randn(proj32.size(-1), self.num_proj, device=proj.device, dtype=torch.float32)
    directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-8)
    return self._exec_statistic(proj32, directions, self.t, self.phi, self.weights).to(proj.dtype)


def configure_sigreg_execution(regularizer):
    """Compile only the deterministic native statistic; retain eager RNG order."""
    from module import SIGReg

    if type(regularizer) is not SIGReg:
        raise ValueError('Compiled native statistic requires the frozen SIGReg implementation')
    if hasattr(regularizer, '_exec_statistic'):
        return regularizer
    schema = [(name, tuple(value.shape)) for name, value in regularizer.state_dict().items()]
    regularizer._exec_statistic = torch.compile(_native_sigreg_statistic, dynamic=False, fullgraph=True,
        options={'emulate_precision_casts': True, 'fallback_random': True})
    regularizer.forward = MethodType(_native_sigreg_forward, regularizer)
    assert schema == [(name, tuple(value.shape)) for name, value in regularizer.state_dict().items()]
    return regularizer


def install_uint8_dataset(trainer, implementation, receipt):
    from infra_pixels import KeepUint8, normalize_pixels
    original_preprocessor = implementation.get_img_preprocessor
    implementation.get_img_preprocessor = KeepUint8
    original_dataset = trainer.build_dataset

    def checked_dataset(cfg):
        dataset = original_dataset(cfg)
        transform = original_preprocessor('pixels', 'pixels', img_size=cfg.img_size)
        checks = []
        for index in (0, 501, len(dataset) - 1):
            raw = dataset[index]['pixels']
            expected = transform({'pixels': raw.clone()})['pixels'].as_subclass(torch.Tensor)
            actual = normalize_pixels(raw.cuda()).cpu()
            checks.append(dict(index=index, shape=list(raw.shape), equal=torch.equal(actual, expected),
                               max_abs=(actual - expected).abs().max().item()))
            assert checks[-1]['equal'], checks[-1]
        Path(receipt).write_text(json.dumps(checks, indent=2))
        return dataset

    trainer.build_dataset = checked_dataset


def prepare_batch(batch):
    if batch['pixels'].dtype == torch.uint8:
        from infra_pixels import normalize_pixels
        return dict(batch, pixels=normalize_pixels(batch['pixels']))
    return batch


def configure_encoder_cpu_offload(model):
    """Offload saved encoder tensors without changing dtype or batch geometry."""
    original = model.encoder.forward

    def forward(*args, **kwargs):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return original(*args, **kwargs)

    model.encoder.forward = forward


def configure_encoder_image_chunks(model, max_images):
    """Bound ViT workspace while keeping the projector's complete batch."""
    assert max_images > 0
    encoder = model.encoder
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                   for module in encoder.modules())
    assert not any(isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p
                   for module in encoder.modules())
    original = encoder.forward

    def forward(pixel_values=None, *args, **kwargs):
        # Diagnostic hooks require one complete call and the original hidden-state layout.
        if (pixel_values is None or pixel_values.shape[0] <= max_images or args
                or getattr(model, '_capture_block_outputs', False)
                or kwargs.get('output_hidden_states', encoder.config.output_hidden_states)
                or kwargs.get('output_attentions', encoder.config.output_attentions)
                or kwargs.get('return_dict', encoder.config.use_return_dict) is False):
            return original(pixel_values, *args, **kwargs)
        outputs = [original(part, **kwargs) for part in pixel_values.split(max_images)]
        keys = set(outputs[0].keys())
        assert keys <= {'last_hidden_state', 'pooler_output'}
        assert all(set(output.keys()) == keys for output in outputs)
        return type(outputs[0])(**{key: torch.cat([output[key] for output in outputs], dim=0)
                                  for key in keys})

    encoder.forward = forward


def configure_encoder_pixel_lifetime(model, *, channels_last=False, replay_conv_input=False,
                                     direct_pixel_restore=False):
    """Release normalized FP32 images before loss and encoder recomputation."""
    original = model.encode
    assert not direct_pixel_restore or replay_conv_input
    from contextvars import ContextVar
    raw_input = ContextVar('intact_conv_uint8_input', default=None)
    if channels_last:
        model.encoder.to(memory_format=torch.channels_last)
    if replay_conv_input:
        projection = model.encoder.embeddings.patch_embeddings.projection
        original_projection = projection.forward
        model._exec_pixel_replay_stats = dict(calls=0)

        def projection_forward(pixels):
            raw = raw_input.get()
            if raw is None or not torch.is_grad_enabled():
                return original_projection(pixels)
            shape = tuple(pixels.shape)
            assert len(shape) == 4 and raw.numel() == pixels.numel()

            def pack(value):
                if tuple(value.shape) != shape:
                    return value.detach()
                model._exec_pixel_replay_stats.update(
                    calls=model._exec_pixel_replay_stats['calls'] + 1,
                    saved_input_bytes=value.numel() * value.element_size(),
                    raw_input_bytes=raw.numel(), dtype=str(value.dtype))
                return raw, value.dtype, tuple(value.stride())

            def unpack(value):
                if isinstance(value, torch.Tensor):
                    return value
                from infra_pixels import normalize_pixels
                frames, dtype, stride = value
                if direct_pixel_restore:
                    from native_pixel_replay import restore_conv_pixels
                    restored = restore_conv_pixels(frames, shape, dtype, channels_last)
                else:
                    restored = normalize_pixels(frames, channels_last=channels_last).reshape(shape).to(dtype)
                assert tuple(restored.stride()) == stride
                return restored

            # Only the patch convolution's saved image is replaced. Its native
            # backward and saved weights keep their original implementation.
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                return original_projection(pixels)

        projection.forward = projection_forward

    def encode(info):
        pixels = info['pixels']
        if pixels.dtype != torch.uint8:
            return original(info)
        from infra_pixels import normalize_pixels
        info['pixels'] = normalize_pixels(pixels, channels_last=channels_last)
        token = raw_input.set(pixels) if replay_conv_input else None
        try:
            return original(info)
        finally:
            if token is not None:
                raw_input.reset(token)
            # Autograd keeps the tensors needed by convolution; this batch no
            # longer needs a separate normalized FP32 image reference.
            info['pixels'] = pixels

    model.encode = encode
