"""uint8-to-patch GEMM with image replay in the weight backward."""
import torch
import triton
import triton.language as tl


@triton.jit
def _patches(Raw, Table, Patches, N: tl.constexpr, HEIGHT: tl.constexpr,
             WIDTH: tl.constexpr, PATCH: tl.constexpr, BLOCK: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = offset % (3 * PATCH * PATCH)
    patch = offset // (3 * PATCH * PATCH)
    channel = inner // (PATCH * PATCH)
    row = patch // (WIDTH // PATCH) % (HEIGHT // PATCH) * PATCH + inner // PATCH % PATCH
    col = patch % (WIDTH // PATCH) * PATCH + inner % PATCH
    image = patch // ((HEIGHT // PATCH) * (WIDTH // PATCH))
    source = image*S0 + channel*S1 + row*S2 + col*S3
    pixel = tl.load(Raw + source, offset < N, other=0).to(tl.int32)
    normalized = tl.load(Table + channel * 256 + pixel)
    tl.store(Patches + offset, normalized, offset < N)


def normalized_patches(raw, table, patch, dtype):
    height, width = raw.shape[-2:]
    if (raw.ndim != 4 or raw.dtype != torch.uint8 or raw.shape[-3] != 3
            or not (raw.is_contiguous() or raw.is_contiguous(memory_format=torch.channels_last))):
        raise ValueError('Patch projection requires dense NCHW or channels-last uint8 RGB images')
    if height % patch or width % patch:
        raise ValueError('Patch projection requires complete non-overlapping patches')
    columns = 3 * patch * patch
    output = torch.empty((raw.numel() // columns, columns), device=raw.device, dtype=dtype)
    _patches[(triton.cdiv(raw.numel(), 1024),)](
        raw, table, output, raw.numel(), height, width, patch, 1024,*raw.stride())
    return output


class _PatchProjection(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, raw, table, weight, bias):
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else weight.dtype
        patch = weight.shape[-1]
        pixels = normalized_patches(raw, table, patch, dtype)
        # cuDNN convolution rounds the product before its separate bias addition.
        output = torch.nn.functional.linear(pixels, weight.flatten(1).to(dtype))
        output = output + bias.to(dtype)
        ctx.save_for_backward(raw, table)
        ctx.weight_shape, ctx.weight_dtype, ctx.bias_dtype = weight.shape, weight.dtype, bias.dtype
        ctx.patch, ctx.dtype = patch, dtype
        height, width = raw.shape[-2:]
        return output.view(raw.shape[0], height // patch, width // patch, weight.shape[0]).permute(0, 3, 1, 2)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        raw, table = ctx.saved_tensors
        pixels = normalized_patches(raw, table, ctx.patch, ctx.dtype)
        gradient = gradient.permute(0, 2, 3, 1).reshape(-1, ctx.weight_shape[0])
        weight_gradient = gradient.T @ pixels
        bias_gradient = gradient.float().sum(0).to(ctx.dtype)
        return (None, None, weight_gradient.view(ctx.weight_shape).to(ctx.weight_dtype),
                bias_gradient.to(ctx.bias_dtype))


def normalized_images(raw, table, dtype):
    from infra_pixels import _lookup_pixels_nhwc
    output = torch.empty(raw.shape, dtype=dtype, device=raw.device, memory_format=torch.channels_last)
    _lookup_pixels_nhwc[(triton.cdiv(raw.numel(), 1024),)](
        raw, table, output, raw.numel(), raw.shape[-1] * raw.shape[-2], 1024)
    return output


class _RawConvolution(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, raw, table, weight, bias):
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else weight.dtype
        pixels = normalized_images(raw, table, dtype)
        typed_weight = weight.to(dtype)
        output = torch.nn.functional.conv2d(pixels, typed_weight, bias.to(dtype), stride=weight.shape[-1])
        ctx.save_for_backward(raw, table, typed_weight)
        ctx.weight_dtype, ctx.bias_dtype = weight.dtype, bias.dtype
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        raw, table, weight = ctx.saved_tensors
        pixels = normalized_images(raw, table, weight.dtype)
        _, dw, db = torch.ops.aten.convolution_backward.default(
            gradient, pixels, weight, [weight.shape[0]], list(weight.shape[-2:]),
            [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
        return None, None, dw.to(ctx.weight_dtype), db.to(ctx.bias_dtype)


def configure_uint8_patch(model, *, implementation='convolution', image_layout='nchw'):
    """Keep the native embedding, projector and parameter schema; bypass image casts."""
    projection = model.encoder.embeddings.patch_embeddings.projection
    patch = projection.kernel_size
    if (patch[0] != patch[1] or projection.stride != patch or projection.padding != (0, 0)
            or projection.dilation != (1, 1) or projection.groups != 1 or projection.bias is None
            or projection.in_channels != 3 or model.frame_readout != 'cls'):
        raise ValueError('Uint8 patch projection requires non-overlapping RGB CLS ViT')
    original_projection, original_encode = projection.forward, model.encode
    if implementation not in ('convolution', 'gemm'):
        raise ValueError(implementation)
    if image_layout not in ('nchw','channels_last') or image_layout=='channels_last' and implementation!='gemm':
        raise ValueError('Channels-last uint8 transport requires the patch GEMM')
    operator = _RawConvolution if implementation == 'convolution' else _PatchProjection

    def project(raw):
        if raw.dtype != torch.uint8:
            return original_projection(raw)
        from infra_pixels import _tables, normalize_pixels
        if raw.device not in _tables:
            normalize_pixels(raw[:1, :, :1, :1].contiguous())
        return operator.apply(raw, _tables[raw.device], projection.weight, projection.bias)

    def encode(info):
        raw = info['pixels']
        if raw.dtype != torch.uint8:
            return original_encode(info)
        if model._capture_block_outputs:
            raise ValueError('Uint8 patch candidate does not support diagnostic token capture')
        batch, frames = raw.shape[:2]
        images = raw.reshape(-1, *raw.shape[2:])
        if image_layout == 'nchw':
            images = images.contiguous()
        elif not images.is_contiguous(memory_format=torch.channels_last):
            images = images.contiguous(memory_format=torch.channels_last)
        tokens = model.encoder(images,
                               interpolate_pos_encoding=True).last_hidden_state
        cls = tokens[:, 0]
        embedding = model.projector(cls)
        info['h_cls'] = info['h_repr'] = cls.reshape(batch, frames, -1)
        info['emb'] = embedding.reshape(batch, frames, -1)
        if 'action' in info:
            info['act_emb'] = model.action_encoder(info['action'])
        return info

    projection.forward, model.encode = project, encode
    model._exec_uint8_patch = implementation
