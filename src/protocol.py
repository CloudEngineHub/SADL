"""one shared model, explicit statistics and matched sample budgets."""
import json
import math
from pathlib import Path

FOUR = ['pusht', 'cube', 'reacher', 'tworoom']
COMMON = dict(seed=3072, epochs=1, lr=5e-4, warmup_fraction=0.01,
    scheduler='cosine', weight_decay=1e-3, grad_clip=1.0, workers=5, prefetch_factor=4,
    train_fraction=0.9, precision='bf16', deterministic=False,
    sigreg_weight=0.015, sadl_weight=0.01, bn_momentum=0.1,
    pred_proj_norm='batch', objective='intact', image_size=224, frames=8, frameskip=5,
    history_size=3, latent_dim=192, encoder_sdpa='all_flash', predictor_sdpa='flash',
    checkpoint_encoder=False, checkpoint_predictor=False, execution='sadl_fast',
    checkpoint_every=1000, log_every=50, tasks=FOUR, world_size=4, micro_batch=64,
    accumulation=1, global_batch=256, epoch='matched_global_pass_budget', core_depth=2, core_heads=4,
    actor_hidden=576, actor_experts=4, prediction_norm='none')


def cells(seeds=(3072, 3073, 3074)):
    """Training cells: INTACT and LeWM objectives, four-task joint and single-task, one epoch each.

    Every cell uses one global mixed-task SyncBN population (independent of GPU and task ID), SADL 0.01,
    SIGReg 0.015, pred_proj BN, cosine LR 5e-4 with 1% warmup, BF16 + Flash attention."""
    result = {}
    for objective in ('intact', 'lewm'):
        tag = objective.upper()
        for seed in seeds:
            name = f'{tag}_4T_S{seed}'
            result[name] = dict(COMMON, id=name, seed=seed, objective=objective, sampling='balanced',
                                downstream='transformer_mlp',
                                epoch='largest_task_once_with_smaller_task_repeats')
            for task in FOUR:
                name = f'{tag}_{task.upper()}_S{seed}'
                result[name] = dict(COMMON, id=name, seed=seed, objective=objective, tasks=[task],
                                    sampling='balanced', downstream='transformer_mlp',
                                    epoch='largest_task_once_with_smaller_task_repeats')
    return result


def validate(c):
    if c['global_batch'] != c['world_size'] * c['micro_batch'] * c['accumulation']:
        raise ValueError('Optimizer batch must equal world * micro * accumulation')
    if c['accumulation'] != 1:
        raise ValueError('Each optimizer step uses one forward over the global batch')
    if len(c['tasks']) == 1 and c['scheduler'] != 'cosine':
        raise ValueError('Single-task training requires cosine')
    if c['scheduler'] == 'constant' and c['warmup_fraction']:
        raise ValueError('Constant schedule has no warmup')
    if c['encoder_sdpa'] != 'all_flash' or c['predictor_sdpa'] != 'flash':
        raise ValueError('All attention uses Flash; no MATH fallback')
    if c['pred_proj_norm'] not in ('batch', 'none'):
        raise ValueError('pred_proj_norm must be batch (release default) or none')
    if c.get('objective', 'intact') not in ('intact', 'lewm'):
        raise ValueError('objective must be intact or lewm')
    if int(c.get('epochs', 1)) < 1:
        raise ValueError('epochs must be >= 1')


def learning_rate(c, step, steps):
    if c['scheduler'] == 'constant':
        return c['lr']
    warm = max(1, int(steps * c['warmup_fraction'])) if c['warmup_fraction'] else 0
    if step < warm:
        return c['lr'] * step / warm
    return c['lr'] * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))


def read_cell(path, name):
    result = json.loads(Path(path).read_text())[name]
    validate(result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    values = cells()
    for c in values.values():
        validate(c)
    text = json.dumps(values, indent=2) + '\n'
    if args.write:
        args.write.write_text(text)
    else:
        print(text)
