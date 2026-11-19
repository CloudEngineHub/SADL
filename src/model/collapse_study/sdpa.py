"""Layer-resolved SDPA routing without changing checkpoint keys."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None
    sdpa_kernel = None


VALID_BACKENDS = {"math", "flash", "inherit"}


def configure_training_sdpa(global_backend: str) -> None:
    """Configure kernels outside explicitly routed ViT layers."""

    if global_backend not in {"math", "auto"}:
        raise ValueError("global_backend must be 'math' or 'auto'")
    math_only = global_backend == "math"
    torch.backends.cuda.enable_flash_sdp(not math_only)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def sdpa_context(name: str):
    if name == "inherit":
        return nullcontext()
    if sdpa_kernel is not None:
        if name == "math":
            return sdpa_kernel(SDPBackend.MATH)
        # "flash": prefer Flash (BF16); FP32 inputs fall back to the efficient / math kernels.
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    return torch.backends.cuda.sdp_kernel(
        enable_flash=name == "flash",
        enable_math=name == "math",
        enable_mem_efficient=False,
    )


def _vit_layers(encoder: torch.nn.Module) -> list[torch.nn.Module]:
    """Return transformer blocks, or an empty list for encoder test doubles.

    The production encoder is a ViT and must expose one of the two supported
    block containers below.  Small unit-test encoders intentionally do not
    have attention blocks; treating those models as an unrouted no-op keeps
    the router orthogonal to the model/objective tests.
    """
    if hasattr(encoder, "layers"):
        return list(encoder.layers)
    nested = getattr(encoder, "encoder", None)
    if nested is not None and hasattr(nested, "layer"):
        return list(nested.layer)
    return []


def resolve_layer_backends(spec: str | Iterable[str], depth: int) -> list[str]:
    """Resolve aliases or a comma-separated backend list for any ViT depth."""

    if depth < 1:
        raise ValueError("ViT depth must be positive")
    if not isinstance(spec, str):
        values = [str(value).strip().lower() for value in spec]
    else:
        alias = spec.strip().lower()
        midpoint = depth // 2
        aliases = {
            "all_math": ["math"] * depth,
            "all_flash": ["flash"] * depth,
            "first_half_math": ["math"] * midpoint + ["flash"] * (depth - midpoint),
            "last_half_math": ["flash"] * midpoint + ["math"] * (depth - midpoint),
        }
        values = aliases.get(alias, [part.strip() for part in alias.split(",")])
    if len(values) != depth:
        raise ValueError(f"Expected {depth} layer backends, got {len(values)}: {values}")
    invalid = sorted(set(values) - VALID_BACKENDS)
    if invalid:
        raise ValueError(f"Unknown layer backends: {invalid}")
    return values


class LayerwiseSDPARouter:
    """Enter a requested SDPA context around each ViT block forward."""

    def __init__(self, encoder: torch.nn.Module, backends: str | Iterable[str]) -> None:
        self.layers = _vit_layers(encoder)
        self.backends = (
            resolve_layer_backends(backends, len(self.layers))
            if self.layers
            else []
        )
        self.capture_enabled = False
        self.captures: list[torch.Tensor] = []
        self._context_stacks: list[list] = [[] for _ in self.layers]
        self._handles = []
        for index, (layer, backend) in enumerate(zip(self.layers, self.backends)):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_pre_hook(index, backend))
            )
            self._handles.append(
                layer.register_forward_hook(
                    self._make_post_hook(index), always_call=True
                )
            )

    def _make_pre_hook(self, index: int, backend: str):
        def hook(_module, _args):
            context = sdpa_context(backend)
            context.__enter__()
            self._context_stacks[index].append(context)

        return hook

    def _make_post_hook(self, index: int):
        def hook(_module, _args, output):
            stack = self._context_stacks[index]
            if stack:
                stack.pop().__exit__(None, None, None)
            if self.capture_enabled and output is not None:
                hidden = output[0] if isinstance(output, tuple) else output
                self.captures.append(hidden[:, 0])
            return output

        return hook

    def begin_capture(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)
        self.captures.clear()

    def consume_captures(self) -> list[torch.Tensor]:
        captures = self.captures
        self.captures = []
        self.capture_enabled = False
        return captures

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def state_dict(self) -> dict:
        return {
            "depth": len(self.layers),
            "backends": list(self.backends),
            "math_layers": [i for i, mode in enumerate(self.backends) if mode == "math"],
            "flash_layers": [i for i, mode in enumerate(self.backends) if mode == "flash"],
        }
