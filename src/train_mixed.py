"""Distributed training of one cell (launched by train.py through torch.distributed.run)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from protocol import FOUR, learning_rate, read_cell
from sampling import GlobalBatches, IndexedSubset, MixedDataset, collate_mixed, epoch_order, save_split
from engine import RUNTIME, MixedModel, OverlappedGradientSum, backward_step, sum_gradients
from normalization import STEP_LAYOUT


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def atomic_torch(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def configure(c, paths, out, rank):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    import train
    from train import build_dataset
    if c.get('execution') in ('sadl_portable', 'sadl_fast'):
        from execution import INFRA
        from infra_pixels import KeepUint8
        train.get_img_preprocessor = KeepUint8
    cfgs, datasets, identities = [], [], []
    for task in c['tasks']:
        path = Path(paths[task]['train']).resolve(strict=True)
        with initialize_config_dir(version_base=None, config_dir=str(RUNTIME / 'config/train')):
            cfg = compose(config_name='intact_goal', overrides=[f'data=cotrain_{task}'])
        with open_dict(cfg):
            cfg.data.dataset.name = str(path)
            cfg.data.dataset.keys_to_load = ['pixels', 'action']
            cfg.data.dataset.keys_to_cache = ['action']
            cfg.seed = c['seed']
            cfg.num_frames, cfg.img_size = c['frames'], c['image_size']
            cfg.history_size, cfg.embed_dim = c['history_size'], c['latent_dim']
            cfg.data.dataset.frameskip = c['frameskip']
            cfg.model.encoder_sdpa_layers = c['encoder_sdpa']
            cfg.model.predictor_sdpa_backend = c['predictor_sdpa']
            cfg.model.pred_proj.norm_fn = ({'_target_': 'torch.nn.BatchNorm1d', '_partial_': True}
                                           if c['pred_proj_norm'] == 'batch' else None)
            cfg.loss.sigreg.weight = c['sigreg_weight']
            if c['objective'] == 'lewm':
                # LeWM objective: latent prediction only (SIGReg/SADL unchanged); no intent/actor loss.
                cfg.loss.intent.local_weight = 0.0
                cfg.loss.intent.goal_weight = 0.0
            cfg.optimizer.lr, cfg.optimizer.weight_decay = c['lr'], c['weight_decay']
            cfg.loader.num_workers = c['workers']
            cfg.loader.batch_size = c['micro_batch']
        dataset = build_dataset(cfg)
        split_seed = c['seed'] + 100 * FOUR.index(task)
        if rank == 0:
            save_split(out / 'splits', task, len(dataset), c['train_fraction'], split_seed)
        barrier()
        train_file = out / 'splits' / f'{task}_train.npy'
        subset = IndexedSubset(dataset, train_file)
        identities.append(dict(task=task, path=str(path), size=path.stat().st_size,
                               mtime_ns=path.stat().st_mtime_ns, total_windows=len(dataset),
                               train_windows=len(subset), split_seed=split_seed))
        if rank == 0:
            atomic_json(out / 'action_stats' / f'{task}.json', dict(
                mean=dataset.action_mean.flatten().tolist(),
                std=dataset.action_std.flatten().tolist(), ddof=1))
            OmegaConf.save(cfg, out / f'{task}_resolved.yaml', resolve=True)
        cfgs.append(cfg)
        datasets.append(subset)
    # All cells and ranks start from the same seeded model initialization.
    torch.manual_seed(c['seed'])
    maximum = max(int(cfg.model.intent_actor.action_dim) for cfg in cfgs)
    for cfg in cfgs:
        with open_dict(cfg):
            task_dim = int(cfg.model.intent_actor.action_dim)
            cfg.model._target_ = 'shared_model.SharedJEPA'
            cfg.model.action_encoder.input_dim = maximum
            cfg.model.intent_actor.action_dim = maximum
            cfg.model.task_action_dim = task_dim
            for key in ('downstream', 'core_depth', 'core_heads', 'actor_hidden', 'actor_experts'):
                cfg.model[key] = c[key]
    import copy
    shared_cfg = copy.deepcopy(cfgs[0].model)
    shared_cfg.task_action_dim = 0
    models = [hydra.utils.instantiate(shared_cfg)]
    if rank == 0:
        for task, cfg in zip(c['tasks'], cfgs):
            OmegaConf.save(cfg, out / f'{task}_resolved.yaml', resolve=True)
    return cfgs, datasets, identities, models


class StreamPrefetcher:
    """Copy the next batch to the GPU on a side stream while the current step runs."""

    def __init__(self, iterator, c, device, depth=1):
        self.iterator, self.c, self.device = iterator, c, device
        self.stream = torch.cuda.Stream(device=device)
        self.queue = []
        for _ in range(depth):
            self._load()

    def _load(self):
        try:
            host = next(self.iterator)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            batch = to_device(host, self.c, self.device)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.queue.append((batch, event))

    def __next__(self):
        if not self.queue:
            raise StopIteration
        batch, event = self.queue.pop(0)
        current = torch.cuda.current_stream(self.device)
        current.wait_event(event)
        for value in batch.values():
            value.record_stream(current)
        self._load()
        return batch


def to_device(batch, c, device):
    if batch is not None:
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return dict(pixels=torch.empty(0, c['frames'], 3, c['image_size'], c['image_size'], device=device),
                action=torch.empty(0, c['frames'], 0, device=device),
                previous_action=torch.empty(0, c['frames'], 0, device=device),
                task_id=torch.empty(0, dtype=torch.long, device=device),
                action_dim=torch.empty(0, dtype=torch.long, device=device),
                position=torch.empty(0, dtype=torch.long, device=device),
                sample_id=torch.empty(0, dtype=torch.long, device=device))


def save_state(out, model, optimizer, c, identities, next_step, elapsed, rank, world, counts):
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if rank == 0:
        atomic_torch(out / 'last.ckpt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            contract=c, datasets=identities, next_step=next_step, rng_by_rank=states,
            elapsed_seconds=elapsed, samples_seen_by_task=counts.cpu()))
    barrier()


def export_models(out, model, cfgs, c):
    from omegaconf import OmegaConf
    shared = out / 'shared_weights_epoch_1.pt'
    atomic_torch(shared, model.models[0].state_dict())
    for task, cfg in zip(c['tasks'], cfgs):
        target = out / 'exports' / task
        target.mkdir(parents=True, exist_ok=True)
        # Deployment uses ordinary visual BN and the same normalization-free
        # prediction MLP, instantiated from this exported model configuration.
        link = target / 'weights_epoch_1.pt'
        if link.exists():
            link.unlink()
        link.symlink_to(shared)
        atomic_json(target / 'config.json', OmegaConf.to_container(cfg.model, resolve=True))
        OmegaConf.save(cfg.model, target / 'config.yaml', resolve=True)
        atomic_json(target / 'action_stats.json', json.loads((out / 'action_stats' / f'{task}.json').read_text()))


def run(args):
    c = read_cell(args.matrix, args.cell)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank, local_rank = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    if world != c['world_size']:
        raise ValueError(f"{args.cell} needs {c['world_size']} ranks; got {world}")
    if c['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(c['deterministic'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    out = args.output.resolve()
    if (out / 'COMPLETE.json').exists():
        raise RuntimeError('This run is already complete')
    if (out / 'contract.json').exists() and not args.resume:
        if not args.restart_unstarted or (out / 'progress.json').exists() or (out / 'last.ckpt').exists():
            raise RuntimeError('Existing valid training requires --resume')
        if json.loads((out / 'contract.json').read_text()) != c:
            raise ValueError('Cannot restart an unstarted run with a changed configuration')
    out.mkdir(parents=True, exist_ok=True)
    paths = json.loads(args.datasets.read_text())
    if any(task not in paths for task in c['tasks']):
        raise ValueError('Dataset manifest must include every selected task')
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.set_num_threads(4)
    os.environ['INTACT_REUSE_ACTION_EMBEDDING'] = '1'
    cfgs, datasets, identities, task_models = configure(c, paths, out, rank)
    model = MixedModel(task_models, cfgs, c).to(device).train()
    if rank == 0:
        atomic_json(out / 'effective_model.json', dict(
            parameters=sum(p.numel() for p in model.parameters()),
            shared_models=len(model.models), task_specific_parameters=False,
            downstream=c['downstream'], execution=model.execution_receipt,
            actual_sigreg_weight=model.regularizer.output_weight,
            actual_sadl_weight=model.regularizer.bridge_weight,
            normalization='one SyncBN population over the global batch, grouped by window position before sharding',
            history=c['history_size'], frames=c['frames'], frameskip=c['frameskip']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'],
                                 weight_decay=c['weight_decay'], fused=True)
    start, prior_elapsed, saved_rng = 0, 0.0, None
    counts = torch.zeros(len(datasets), device=device, dtype=torch.long)
    if args.resume:
        state = torch.load(out / 'last.ckpt', map_location='cpu', weights_only=False)
        if state['contract'] != c or state['datasets'] != identities:
            raise ValueError('Resume contract/datasets differ from saved training state')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, prior_elapsed = state['next_step'], state['elapsed_seconds']
        saved_rng = state['rng_by_rank'][rank]
        counts.copy_(state['samples_seen_by_task'].to(device))
    if rank == 0:
        atomic_json(out / 'contract.json', c)
        atomic_json(out / 'datasets.json', identities)
        lengths = [len(d) for d in datasets]
        np.save(out / 'global_order.npy', np.concatenate([epoch_order(lengths, c['sampling'], c['seed'] + e) for e in range(c['epochs'])]))
    barrier()
    order = np.load(out / 'global_order.npy', mmap_mode='r')
    lengths = [len(d) for d in datasets]
    expected_counts = (lengths if c['sampling'] == 'proportional' else
                       [max(lengths)] * len(datasets))
    expected_counts = [n * c['epochs'] for n in expected_counts]
    sampler = GlobalBatches(order, world, rank, c['micro_batch'], c['accumulation'], start)
    options = dict(num_workers=c['workers'], pin_memory=True,
                   generator=torch.Generator().manual_seed(c['seed'] + rank + 50_000))
    if c['workers']:
        options.update(persistent_workers=True, prefetch_factor=c['prefetch_factor'], multiprocessing_context='spawn')
    loader = DataLoader(MixedDataset(datasets), batch_sampler=sampler,
                        collate_fn=collate_mixed, **options)
    iterator = iter(loader)
    prefetcher = StreamPrefetcher(iterator, c, device) if c['accumulation'] == 1 else None
    if saved_rng is not None:
        restore_rng(saved_rng)
    if rank == 0:
        atomic_json(out / 'budget.json', dict(total_windows=len(order), optimizer_steps=sampler.steps,
            warmup_steps=(max(1, int(sampler.steps * c['warmup_fraction']))
                          if c['warmup_fraction'] else 0),
            first_update_lr=learning_rate(c, 0, sampler.steps),
            last_update_lr=learning_rate(c, sampler.steps - 1, sampler.steps),
            terminal_schedule_lr=learning_rate(c, sampler.steps, sampler.steps),
            unique_train_windows=sum(len(d) for d in datasets),
            epoch_definition=c['epoch'], expected_task_windows=expected_counts))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    reducer = OverlappedGradientSum(model.parameters()) if world > 1 else None
    bn_bounds = np.cumsum([0] + list([c['global_batch']]))
    reg_bounds = np.cumsum([0] + list([c['global_batch']]))
    for step in range(start, sampler.steps):
        optimizer.zero_grad(set_to_none=True)
        if c['accumulation'] == 1:
            # Host-side step layout from the deterministic sampler (no device synchronisation).
            first = step * sampler.global_batch
            valid = min(sampler.global_batch, len(order) - first)
            STEP_LAYOUT.update(valid_windows=valid,
                rank_windows=[len(range(first + r, first + valid, world)) for r in range(world)],
                group_windows=[int(max(0, min(valid, b) - a)) for a, b in zip(bn_bounds[:-1], bn_bounds[1:])],
                regularizer_windows=[int(max(0, min(valid, b) - a)) for a, b in zip(reg_bounds[:-1], reg_bounds[1:])])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(c, step, sampler.steps)
        batches = ([next(prefetcher)] if prefetcher is not None else
                   [to_device(next(iterator), c, device) for _ in range(c['accumulation'])])
        if step == start:
            audit = [{'sample_ids': b['sample_id'].tolist(), 'task_ids': b['task_id'].tolist(),
                      'global_positions': b['position'].tolist()}
                     for b in batches]
            atomic_json(out / f'first_batch_rank{rank}.json', audit)
        metrics = backward_step(model, batches, step)
        if reducer is not None:
            reducer.finish()
        else:
            sum_gradients(model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        STEP_LAYOUT['nonfinite'] = STEP_LAYOUT['nonfinite'] | ~torch.isfinite(grad_norm) \
            if 'nonfinite' in STEP_LAYOUT else ~torch.isfinite(grad_norm)
        optimizer.step()
        counts += metrics['task_counts']
        if (step + 1) % c['log_every'] == 0 or step == start or step + 1 == sampler.steps:
            torch.cuda.synchronize()
            if 'nonfinite' in STEP_LAYOUT and bool(STEP_LAYOUT.pop('nonfinite').item()):
                raise FloatingPointError(f'Non-finite objective or gradient before step {step + 1}')
            elapsed = prior_elapsed + time.perf_counter() - started
            peak = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                                device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                row = dict(step=step + 1, steps=sampler.steps, lr=optimizer.param_groups[0]['lr'],
                    elapsed_seconds=elapsed, grad_norm=float(grad_norm), task_windows=counts.tolist(),
                    samples_per_second=min((step + 1) * c['global_batch'], len(order)) / max(elapsed, 1e-9),
                    peak_allocated_gib=peak[0].item() / 2**30, peak_reserved_gib=peak[1].item() / 2**30,
                    **{k: float(v) for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1})
                row['task_metrics'] = {c['tasks'][i]: (values / metrics['task_counts'][i].clamp_min(1)).tolist()
                                       for i, values in metrics['task_metrics'].items()}
                with (out / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                atomic_json(out / 'progress.json', row)
                print(json.dumps(row), flush=True)
        if (step + 1) % c['checkpoint_every'] == 0 or step + 1 == sampler.steps:
            save_state(out, model, optimizer, c, identities, step + 1,
                       prior_elapsed + time.perf_counter() - started, rank, world, counts)
    expected = torch.tensor(expected_counts, device=device)
    if not torch.equal(counts, expected):
        raise RuntimeError(f'Actual task coverage differs: {counts.tolist()} != {expected.tolist()}')
    if rank == 0:
        export_models(out, model, cfgs, c)
        atomic_json(out / 'COMPLETE.json', dict(cell=c['id'], optimizer_steps=sampler.steps,
            total_windows=len(order), elapsed_seconds=prior_elapsed + time.perf_counter() - started,
            export_dir=str(out / 'exports'), pbn=False, parameter_ema=False))
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', type=Path, default=Path(__file__).with_name('matrix.json'))
    p.add_argument('--cell', required=True)
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--restart-unstarted', action='store_true')
    run(p.parse_args())
