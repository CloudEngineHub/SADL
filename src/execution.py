"""Select the execution path for the shared shared-BN model model."""
from pathlib import Path
import sys

INFRA = Path(__file__).resolve().parent / 'kernels'
sys.path.insert(0, str(INFRA))


def configure(model):
    from sadl_execution import configure_whole_encoder, configure_capacity_async_checks
    from sadl_portable import configure_portable_mlp
    from sadl_patch import configure_uint8_patch
    configure_whole_encoder(model, replay='attention', bf16_residual=True,
                            cuda_graphs=False, attention='portable')
    configure_portable_mlp(model, recompute=True)
    configure_uint8_patch(model, implementation='gemm')
    configure_capacity_async_checks()
    return dict(path='portable',
        encoder=model._exec_whole_encoder, mlp=model._exec_fused_mlp,
        attention='Flash-only (portable fallback explicitly disabled)',
        predictor='batched causal windows', actor='paired local/goal forward',
        sigreg='published streaming trigonometric recomputation',
        unchanged='parameter ownership of encoder and visual projector; losses and BN population')


def configure_fast(model):
    """FA4 encoder attention and fused encoder MLP on SM90 / SM120 GPUs; portable operators elsewhere."""
    import importlib.metadata, json as _json, site
    import torch
    capability = torch.cuda.get_device_capability()[0]
    try:
        site.addsitedir(str(importlib.metadata.distribution('flash-attn-4').locate_file('')))
        fa4 = True
    except importlib.metadata.PackageNotFoundError:
        fa4 = False
    if capability not in (9, 12) or not fa4:
        receipt = configure(model)
        receipt.update(fast_path=False, reason=f'sm{capability}0, flash-attn-4 {"found" if fa4 else "missing"}')
        return receipt
    from sadl_execution import configure_whole_encoder, configure_capacity_async_checks
    from sadl_mlp import configure_fused_mlp
    from sadl_patch import configure_uint8_patch
    gemm = _json.loads((INFRA / 'gemm_configs.json').read_text())[str(capability)]
    configure_whole_encoder(model, replay='attention', bf16_residual=True,
                            cuda_graphs=False, attention='fa4-scheduled')
    configure_fused_mlp(model, lookup=True, tiled=True, recompute=True,
                        pointwise_replay=True, plain_gemm_configs=gemm)
    configure_uint8_patch(model, implementation='gemm')
    configure_capacity_async_checks()
    return dict(path='fast', fast_path=True,
        cuda_capability=capability, encoder=model._exec_whole_encoder, mlp=model._exec_fused_mlp,
        attention='FA4 scheduled (encoder); Flash SDPA (shared core)',
        sigreg='published streaming trigonometric recomputation',
        unchanged='parameters, losses, BN populations and sampling')
