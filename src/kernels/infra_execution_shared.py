"""Optional execution changes for the fixed 4 x 64 RoboTwin recipe."""
import atexit
import os
from functools import partial
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts,
)


def _save_attention(ctx, op, *args, **kwargs):
    if op in {torch.ops.aten._scaled_dot_product_flash_attention.default,
              torch.ops.aten._scaled_dot_product_cudnn_attention.default}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _save_compute(ctx, op, *args, **kwargs):
    if op in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
              torch.ops.aten.bmm.default}:
        return CheckpointPolicy.MUST_SAVE
    return _save_attention(ctx, op, *args, **kwargs)


def _checkpoint_forward(layer, selective=False, save_compute=False):
    forward = layer.forward
    policy = _save_compute if save_compute else _save_attention
    context = partial(create_selective_checkpoint_contexts, policy)

    def run(*args, **kwargs):
        if not layer.training or not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        if selective or save_compute:
            return checkpoint(forward, *args, use_reentrant=False,
                              context_fn=context, **kwargs)
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)

    return run


def _cls_tail_forward(layer, hidden_states, attention_mask=None, **kwargs):
    if attention_mask is not None or kwargs:
        raise ValueError('CLS tail supports the unmasked RoboTwin ViT only')
    attention = layer.attention
    normed = layer.layernorm_before(hidden_states)
    batch, length = normed.shape[:2]
    heads, width = attention.num_attention_heads, attention.head_dim
    query = attention.q_proj(normed[:, :1]).view(batch, 1, heads, width).transpose(1, 2)
    key = attention.k_proj(normed).view(batch, length, heads, width).transpose(1, 2)
    value = attention.v_proj(normed).view(batch, length, heads, width).transpose(1, 2)
    if getattr(attention, '_exec_cudnn_attention', False):
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    context = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=attention.scaling)
    context = context.transpose(1, 2).reshape(batch, 1, heads * width).contiguous()
    output = attention.o_proj(context) + hidden_states[:, :1]
    return layer.mlp(layer.layernorm_after(output)) + output


def _packed_attention_forward(attention, hidden_states, attention_mask=None, **kwargs):
    if attention_mask is not None or kwargs:
        raise ValueError('Packed QKV supports the unmasked RoboTwin ViT only')
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    weight = torch.cat([projection.weight for projection in projections], dim=0)
    bias = (torch.cat([projection.bias for projection in projections], dim=0)
            if projections[0].bias is not None else None)
    batch, length = hidden_states.shape[:2]
    heads, width = attention.num_attention_heads, attention.head_dim
    # The original parameters remain separate; cat backward routes each slice
    # to its original weight, optimizer state, and DDP bucket.
    qkv = F.linear(hidden_states, weight, bias).view(batch, length, 3, heads, width)
    query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(2))
    if getattr(attention, '_exec_cudnn_attention', False):
        # PyTorch 2.7's cuDNN meta kernel assumes a contiguous BHSD output.
        # Contiguous Q/K/V also make the actual cuDNN output follow that layout.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                           scale=attention.scaling)
    output = output.transpose(1, 2).reshape(batch, length, heads * width).contiguous()
    return attention.o_proj(output), None


def _checkpoint_mlp_forward(layer, hidden_states, attention_mask=None, **kwargs):
    attention, _ = layer.attention(layer.layernorm_before(hidden_states), attention_mask, **kwargs)
    residual = hidden_states + attention

    def feed_forward(value):
        return layer.mlp(layer.layernorm_after(value))

    if layer.training and torch.is_grad_enabled():
        output = checkpoint(feed_forward, residual, use_reentrant=False, preserve_rng_state=False)
    else:
        output = feed_forward(residual)
    return residual + output


def _affine_flow_sample(actor, z, z_next, context_emb, noise, steps=None):
    steps = actor.sample_steps if steps is None else int(steps)
    if steps < 1:
        raise ValueError('flow sampling steps must be positive')
    context = actor.context_trunk(actor.context_features(z, z_next, context_emb))
    first = actor.velocity_net[0]
    width = context.shape[-1]
    constant = F.linear(context, first.weight[:, :width], first.bias)
    action = noise
    dt = 1.0 / float(steps)
    for index in range(steps):
        time = action.new_full((*action.shape[:-1], 1), index * dt)
        dynamic = torch.cat([action, actor.time_embedding(time)], dim=-1)
        velocity = constant + F.linear(dynamic, first.weight[:, width:])
        for layer in actor.velocity_net[1:]:
            velocity = layer(velocity)
        action = action + dt * velocity
    return action


def _compile_flow_sample(actor, affine=False):
    original = actor.sample
    forward = partial(_affine_flow_sample, actor) if affine else original
    compiled = torch.compile(forward, dynamic=False)

    def sample(z, z_next, context_emb, noise=None, steps=None):
        if actor.training or torch.is_grad_enabled():
            return original(z, z_next, context_emb, noise=noise, steps=steps)
        # Keep the original RNG operation outside the deterministic ODE graph.
        if noise is None:
            noise = torch.randn(*z.shape[:-1], actor.action_dim,
                                device=z.device, dtype=z.dtype) * actor.noise_scale
        return compiled(z, z_next, context_emb, noise=noise, steps=steps)

    actor.sample = sample


def _encoder_segments(model, group_size, autotune=False):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    encoder = model.encoder
    router = model._encoder_sdpa_router
    if any(backend != 'flash' for backend in router.backends) or encoder.pooler is not None:
        raise ValueError('Segmented encoder requires all-Flash ViT without a pooler')
    if encoder.embeddings.dropout.p or any(
        layer.dropout.p or layer.attention.attention_dropout for layer in router.layers
    ):
        raise ValueError('Segmented encoder requires dropout-free blocks')
    router.close()
    segments = []
    for offset in range(0, len(router.layers), group_size):
        layers = tuple(router.layers[offset:offset + group_size])

        def segment(hidden, layers=layers):
            for layer in layers:
                hidden = layer.forward(hidden)
            return hidden

        mode = 'max-autotune-no-cudagraphs' if autotune else 'default'
        segments.append(torch.compile(segment, dynamic=False, mode=mode))

    def forward(pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None,
                attention_mask=None, **kwargs):
        if bool_masked_pos is not None or attention_mask is not None or kwargs:
            raise ValueError('Segmented ViT supports the unmasked training contract only')
        pixels = pixel_values.to(encoder.embeddings.patch_embeddings.projection.weight.dtype)
        hidden = encoder.embeddings(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        for segment in segments:
            if encoder.training and torch.is_grad_enabled():
                hidden = checkpoint(segment, hidden, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = segment(hidden)
        return BaseModelOutputWithPooling(last_hidden_state=encoder.layernorm(hidden), pooler_output=None)

    encoder.forward = forward


def _encoder_tiles(model, tile_size):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    encoder = model.encoder
    layers = tuple(model._encoder_sdpa_router.layers)
    if encoder.pooler is not None or encoder.embeddings.dropout.p or any(
        layer.dropout.p or layer.attention.attention_dropout for layer in layers
    ):
        raise ValueError('Tiled encoder requires the dropout-free ViT without a pooler')

    def encode_tokens(hidden):
        for layer in layers:
            hidden = layer.forward(hidden)
        return encoder.layernorm(hidden)

    def forward(pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None,
                attention_mask=None, **kwargs):
        if bool_masked_pos is not None or attention_mask is not None or kwargs:
            raise ValueError('Tiled ViT supports the unmasked training contract only')
        pixels = pixel_values.to(encoder.embeddings.patch_embeddings.projection.weight.dtype)
        outputs = []
        for tile in pixels.split(tile_size, dim=0):
            # Embeddings stay outside recomputation, avoiding a second patch
            # convolution. The projector/BatchNorm still receives the full batch.
            hidden = encoder.embeddings(tile, interpolate_pos_encoding=interpolate_pos_encoding)
            if encoder.training and torch.is_grad_enabled():
                hidden = checkpoint(encode_tokens, hidden, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = encode_tokens(hidden)
            outputs.append(hidden)
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(outputs, dim=0), pooler_output=None)

    encoder.forward = forward


def configure(model, variant):
    prior = getattr(model, '_exec_configured_variant', None)
    if prior is not None:
        if prior != variant:
            raise ValueError('Cannot change infra variant on an initialized model')
        return
    options = set(variant.split('+')) - {'baseline'}
    unknown = options - {'compile', 'checkpoint', 'checkpoint_half', 'gpu_pixels', 'fused', 'channels_last', 'reuse', 'cls_tail', 'packed_metrics', 'strict_casts', 'checkpoint_graph', 'checkpoint_attention', 'packed_qkv', 'compile_actor', 'async_metrics', 'checkpoint_mlp', 'no_rng_checkpoint', 'compile_flow', 'segments3', 'segments6', 'flow_affine', 'cudnn', 'autotune', 'compile_aux', 'checkpoint_compute', 'sadl_cache', 'sigreg_compile', 'sadl_batched', 'encoder_tiles128', 'encoder_tiles256', 'encoder_tiles512', 'pixels_nhwc'}
    if unknown:
        raise ValueError(f'Unknown infra options: {unknown}')
    if 'pixels_nhwc' in options and not {'gpu_pixels', 'channels_last'} <= options:
        raise ValueError('NHWC pixels require uint8 GPU preprocessing and channels-last convolution')
    if options & {'cudnn', 'autotune'} and 'compile' not in options:
        raise ValueError('Kernel selection options require the compiled encoder')
    if 'autotune' in options and 'strict_casts' in options:
        raise ValueError('Autotuning and strict cast emulation are separate policies')
    if 'cudnn' in options:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise ValueError('cuDNN attention screening requires a Hopper-or-newer GPU')
        if 'packed_qkv' not in options:
            raise ValueError('cuDNN encoder requires the explicit packed QKV layout')
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Predictor/fusion retain their explicit Flash contexts. Only the
        # compiled encoder inherits this priority; unsupported shapes use Flash.
        context = sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION],
                              set_priority=True)
        context.__enter__()
        model._exec_sdpa_context = context
        atexit.register(context.__exit__, None, None, None)
        for layer in model._encoder_sdpa_router.layers:
            layer.attention._exec_cudnn_attention = True
    if 'async_metrics' in options and 'packed_metrics' not in options:
        raise ValueError('Asynchronous metrics require packed_metrics')
    if 'flow_affine' in options and 'compile_flow' not in options:
        raise ValueError('Factored flow sampling requires compile_flow')
    if 'strict_casts' in options and 'compile' not in options:
        raise ValueError('strict_casts requires the compiled encoder')
    tile_modes = options & {'encoder_tiles128', 'encoder_tiles256', 'encoder_tiles512'}
    checkpoint_modes = options & {'checkpoint', 'checkpoint_half', 'checkpoint_graph', 'checkpoint_attention', 'checkpoint_compute', 'checkpoint_mlp', 'segments3', 'segments6'} | tile_modes
    if len(checkpoint_modes) > 1:
        raise ValueError('Choose exactly one activation checkpoint policy')
    if tile_modes and not {'compile', 'cls_tail'} <= options:
        raise ValueError('Tiled encoder requires compiled blocks and the CLS-only tail')
    if checkpoint_modes & {'checkpoint_graph', 'checkpoint_attention', 'checkpoint_compute'} and 'compile' not in options:
        raise ValueError('Graph checkpointing requires the compiled encoder')
    before = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    if 'compile_aux' in options:
        # Preserve predictor dropout's eager RNG and each original BatchNorm
        # call/sample group while fusing the surrounding small kernels.
        for name in ('predictor', 'projector', 'pred_proj', 'action_encoder',
                     'qpos_encoder', 'view_fusion'):
            module = getattr(model, name, None)
            if module is not None:
                module.forward = torch.compile(module.forward, dynamic=False,
                                               options={'fallback_random': True})
    if options & {'no_rng_checkpoint', 'checkpoint_mlp'}:
        for layer in model._encoder_sdpa_router.layers:
            if layer.dropout.p != 0 or layer.attention.attention_dropout != 0:
                raise ValueError('Deterministic checkpointing requires dropout-free ViT blocks')
    if 'checkpoint_mlp' in options:
        for layer in model._encoder_sdpa_router.layers:
            layer.forward = MethodType(_checkpoint_mlp_forward, layer)
    if 'packed_qkv' in options:
        if 'compile' not in options:
            raise ValueError('Packed QKV requires the compiled encoder')
        for layer in model._encoder_sdpa_router.layers:
            attention = layer.attention
            biases = [projection.bias is not None for projection in
                      (attention.q_proj, attention.k_proj, attention.v_proj)]
            if attention.attention_dropout != 0 or len(set(biases)) != 1:
                raise ValueError('Packed QKV requires dropout-free, matching projections')
            attention.forward = MethodType(_packed_attention_forward, attention)
    if 'cls_tail' in options:
        layers = model._encoder_sdpa_router.layers
        if not layers or model.encoder.pooler is not None:
            raise ValueError('CLS tail requires the ViT without a pooler')
        layer = layers[-1]
        if layer.dropout.p != 0 or layer.attention.attention_dropout != 0:
            raise ValueError('CLS tail requires dropout-free encoder blocks')
        # JEPA consumes only CLS. At the final block, non-CLS queries and MLP
        # rows cannot contribute to CLS; keys/values still use every token.
        layer.forward = MethodType(_cls_tail_forward, layer)
    if 'channels_last' in options:
        for module in model.encoder.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.to(memory_format=torch.channels_last)
    if 'compile_actor' in options:
        actor = model.inverse_actor
        if not getattr(actor, 'is_flow_matching', False):
            raise ValueError('Compiled actor requires the FlowMatching actor')
        if any(isinstance(module, torch.nn.Dropout) and module.p != 0
               for module in actor.modules()):
            raise ValueError('Compiled actor requires dropout-free velocity evaluation')
        # Noise and interpolation time are still sampled by the original
        # flow_matching_loss, outside this deterministic compiled region.
        actor.velocity = torch.compile(actor.velocity, dynamic=False)
    if 'compile_flow' in options:
        actor = model.inverse_actor
        if not getattr(actor, 'is_flow_matching', False) or 'reuse' not in options:
            raise ValueError('Compiled sampling requires FlowMatching and context reuse')
        if any(isinstance(module, torch.nn.Dropout) and module.p != 0
               for module in actor.modules()):
            raise ValueError('Compiled sampling requires a dropout-free actor')
        _compile_flow_sample(actor, 'flow_affine' in options)
    if 'checkpoint' in options or 'checkpoint_half' in options:
        checkpoint_kwargs = {'use_reentrant': False}
        if 'no_rng_checkpoint' in options:
            checkpoint_kwargs['preserve_rng_state'] = False
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        if 'checkpoint_half' in options:
            for index, layer in enumerate(model._encoder_sdpa_router.layers):
                layer.gradient_checkpointing = index % 2 == 0
    if options & {'segments3', 'segments6'}:
        if 'compile' not in options or 'strict_casts' in options:
            raise ValueError('Segmented encoder requires the default compiler policy')
        _encoder_segments(model, 3 if 'segments3' in options else 6, 'autotune' in options)
    elif 'compile' in options:
        # All encoder layers have the same Flash backend in this recipe.
        # Closing the per-layer context hooks avoids graph breaks; the global
        # Flash-only backend and the capture-disabled JEPA path are preserved.
        router = model._encoder_sdpa_router
        if any(backend != 'flash' for backend in router.backends):
            raise ValueError('Compiled encoder requires the all-Flash recipe')
        router.close()
        for layer in router.layers:
            compile_options = {'emulate_precision_casts': True} if 'strict_casts' in options else None
            forward = layer.forward
            if checkpoint_modes & {'checkpoint_graph', 'checkpoint_attention', 'checkpoint_compute'}:
                # Capture checkpoint and block together so AOTAutograd can fuse
                # recomputation with backward instead of replaying an opaque graph.
                forward = _checkpoint_forward(layer, 'checkpoint_attention' in options,
                                              'checkpoint_compute' in options)
            mode = 'max-autotune-no-cudagraphs' if 'autotune' in options else None
            layer.forward = torch.compile(forward, dynamic=False, options=compile_options, mode=mode)
    if tile_modes:
        _encoder_tiles(model, int(next(iter(tile_modes)).removeprefix('encoder_tiles')))
    after = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    if before != after:
        raise AssertionError('Infrastructure variant changed the checkpoint schema')
    model._exec_configured_variant = variant


def enabled(name):
    return name in os.environ.get('INTACT_INFRA_VARIANT', '').split('+')
