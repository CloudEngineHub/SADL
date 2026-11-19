"""Transport uint8 images and reproduce the CPU transform bit for bit on GPU."""
import torch
import triton
import triton.language as tl


class KeepUint8:
    def __init__(self, source, target, img_size):
        self.source, self.target, self.img_size = source, target, img_size

    def __call__(self, sample):
        value = sample[self.source]
        if value.dtype != torch.uint8 or value.shape[-3:] != (3, self.img_size, self.img_size):
            raise ValueError('uint8 transport requires native-size three-channel images')
        sample[self.target] = value
        return sample


@triton.jit
def _lookup_pixels(X, Table, Y, N: tl.constexpr, PLANE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < N, other=0).to(tl.int32)
    channel = offset // PLANE % 3
    normalized = tl.load(Table + channel * 256 + value)
    tl.store(Y + offset, normalized, offset < N)


@triton.jit
def _lookup_pixels_nhwc(X, Table, Y, N: tl.constexpr, PLANE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = offset % 3
    image = offset // (3 * PLANE)
    spatial = offset // 3 % PLANE
    source_offset = image * (3 * PLANE) + channel * PLANE + spatial
    value = tl.load(X + source_offset, offset < N, other=0).to(tl.int32)
    normalized = tl.load(Table + channel * 256 + value)
    tl.store(Y + offset, normalized, offset < N)


_tables = {}


def normalize_pixels(pixels, channels_last=False):
    if pixels.dtype != torch.uint8 or not pixels.is_cuda:
        raise ValueError('GPU pixel normalization requires CUDA uint8 input')
    device = pixels.device
    if device not in _tables:
        from stable_pretraining import data as dt
        transform = dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source='pixels', target='pixels')
        values = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1).contiguous()
        table = transform({'pixels': values})['pixels'].reshape(3, 256).as_subclass(torch.Tensor)
        if not torch.isfinite(table).all():
            raise ValueError('Invalid normalization lookup table')
        _tables[device] = table.to(device)
    pixels = pixels.contiguous()
    if channels_last:
        height, width = pixels.shape[-2:]
        flat = torch.empty((pixels.numel() // (3 * height * width), 3, height, width),
                           dtype=torch.float32, device=device, memory_format=torch.channels_last)
        result = flat.view(pixels.shape)
        kernel = _lookup_pixels_nhwc
    else:
        result = torch.empty(pixels.shape, dtype=torch.float32, device=device)
        kernel = _lookup_pixels
    kernel[(triton.cdiv(pixels.numel(), 1024),)](
        pixels, _tables[device], result, pixels.numel(), pixels.shape[-1] * pixels.shape[-2], 1024,
    )
    return result
