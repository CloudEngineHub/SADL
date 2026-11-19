"""Regularizers and auditable artifacts for the independent Lightning trainer."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from collapse_study import CanonicalCapacityMatching, GaussianQQRegularizer, VISRegularizer
from module import SIGReg

TASKS = {'pusht': 'pusht_expert_train.lance', 'cube': 'ogbench/cube_single_expert.h5',
         'reacher': 'reacher.h5', 'tworoom': 'tworoom.h5'}
WEIGHTS = {'sig': 0.03, 'vis': 1.5, 'qq': 3.5}
BRIDGES = {'sig': 0.01, 'vis': 0.003, 'qq': 0.003}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def enabled(cfg):
    return bool(OmegaConf.select(cfg, 'baseline.enabled', default=False))


def configure(cfg):
    baseline = cfg.baseline
    if baseline.task not in TASKS or str(cfg.data.dataset.name) != TASKS[baseline.task]:
        raise ValueError('Single-task dataset does not match the explicit task')
    if baseline.endpoint not in WEIGHTS:
        raise ValueError('Unsupported endpoint')
    if int(cfg.trainer.devices) != 1:
        raise ValueError('Independent single-task training requires one device')
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise ValueError('Single-task train.py cannot participate in a multi-rank launcher')
    if cfg.init_weights_path:
        raise ValueError('Fresh single-task baselines cannot warm-start from invalid multi weights')
    if float(baseline.endpoint_weight) != WEIGHTS[baseline.endpoint]:
        raise ValueError('Endpoint lambda must match the formal baseline')
    expected_bridge = BRIDGES[baseline.endpoint] if baseline.sadl else 0.0
    if float(baseline.bridge_weight) != expected_bridge:
        raise ValueError('SADL bridge weight does not match the formal baseline')
    with open_dict(cfg):
        cfg.loss.sigreg.weight = 0.0 if baseline.sadl else baseline.endpoint_weight
        cfg.model.encoder_sdpa_layers = baseline.encoder_sdpa
        cfg.model.predictor_sdpa_backend = 'math'
        cfg.loader.persistent_workers = cfg.loader.num_workers > 0
        if cfg.loader.num_workers:
            cfg.loader.multiprocessing_context = 'spawn'
            cfg.loader.prefetch_factor = int(baseline.prefetch_factor)
        else:
            cfg.loader.pop('multiprocessing_context', None)
            cfg.loader.pop('prefetch_factor', None)


def endpoint_regularizer(cfg):
    if not enabled(cfg):
        return SIGReg(**cfg.loss.sigreg.kwargs)
    projections = int(cfg.baseline.num_proj)
    if cfg.baseline.endpoint == 'sig':
        kwargs = OmegaConf.to_container(cfg.loss.sigreg.kwargs, resolve=True)
        kwargs['num_proj'] = projections
        return SIGReg(**kwargs)
    if cfg.baseline.endpoint == 'vis':
        return VISRegularizer(num_proj=projections, quantile_rule='official_n_over_n_plus_one')
    return GaussianQQRegularizer(num_proj=projections)


def joint_regularizer(cfg):
    if not enabled(cfg) or not cfg.baseline.sadl:
        return None
    return CanonicalCapacityMatching(endpoint_regularizer(cfg),
        output_weight=float(cfg.baseline.endpoint_weight), bridge_weight=float(cfg.baseline.bridge_weight),
        ridge=1e-3, snr=9.0, mode='reference_excitation_free_energy', demand_alpha=0.5)


def build_loaders(cfg, train_set, validation_set, sampling_generator):
    # Persistent-worker initialization must not consume the shuffle or model RNG on resume.
    kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    sampler = torch.utils.data.RandomSampler(train_set, generator=sampling_generator)
    train_loader = torch.utils.data.DataLoader(train_set, **kwargs, sampler=sampler, drop_last=True,
        generator=torch.Generator().manual_seed(int(cfg.seed)))
    validation_loader = torch.utils.data.DataLoader(validation_set, **kwargs, shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(int(cfg.seed) + 1))
    return train_loader, validation_loader


def contract(cfg):
    return dict(scope='single-task', entrypoint='train.py', task=str(cfg.baseline.task),
        tasks=[str(cfg.baseline.task)], world_size=1, variant=str(cfg.variant), seed=int(cfg.seed),
        endpoint=str(cfg.baseline.endpoint), endpoint_weight=float(cfg.baseline.endpoint_weight),
        sadl=bool(cfg.baseline.sadl), bridge_weight=float(cfg.baseline.bridge_weight),
        joint_kind='sadl_' + str(cfg.baseline.endpoint) if cfg.baseline.sadl else 'none',
        num_proj=int(cfg.baseline.num_proj), num_workers=int(cfg.loader.num_workers),
        batch_size=int(cfg.loader.batch_size), prefetch_factor=int(cfg.baseline.prefetch_factor),
        history_size=int(cfg.history_size), num_frames=int(cfg.num_frames), train_split=float(cfg.train_split),
        optimizer=OmegaConf.to_container(cfg.optimizer, resolve=True),
        encoder_sdpa=str(cfg.baseline.encoder_sdpa), predictor_sdpa='math',
        precision=str(cfg.trainer.precision), target_epoch=int(cfg.trainer.max_epochs),
        full_epoch=not bool(cfg.baseline.smoke), gradient_checkpointing=bool(cfg.baseline.gradient_checkpointing),
        previous_action_contract='boundary_aware_raw_zero_left_pad_v1',
        sampler_rng='dedicated_checkpointed_generator', worker_rng='separate_deterministic_preprocessing',
        scheduler='LinearWarmupCosineAnnealingLR', scheduler_interval='epoch')


class SingleBaselineCheckpoint(ModelCheckpoint):
    def __init__(self, cfg, generator, dataset_size, train_size, validation_size):
        self.cfg = cfg
        self.generator = generator
        self.run_dir = Path(cfg.baseline.run_dir)
        self.output_cache = Path(cfg.baseline.output_cache)
        self.run_name = str(cfg.output_model_name)
        self.identity = contract(cfg)
        self.pending_rng = None
        self.started = time.time()
        super().__init__(dirpath=str(self.run_dir / 'epochs'), filename='epoch_{epoch:02d}',
                         save_top_k=-1, save_last=False, every_n_epochs=1, save_on_train_epoch_end=True)
        atomic_json(self.run_dir / 'run_identity.json', dict(self.identity,
            protocol='single_lightning', dataset=str(cfg.data.dataset.name), dataset_clips=dataset_size,
            train_clips=train_size, validation_clips=validation_size,
            source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(__file__).with_name('train.py'))}))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['single_baseline_contract'] = self.identity
        numpy_state = np.random.get_state()
        numpy_safe = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
        checkpoint['single_baseline_rng'] = dict(python=random.getstate(), numpy=numpy_safe,
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            loader_generator=self.generator.get_state())

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint.get('single_baseline_contract') != self.identity:
            raise ValueError('Lightning single-task resume contract mismatch')
        self.pending_rng = checkpoint['single_baseline_rng']
        self.generator.set_state(self.pending_rng['loader_generator'].cpu())

    def on_train_start(self, trainer, pl_module):
        if self.pending_rng is not None:
            random.setstate(self.pending_rng['python'])
            numpy_state = self.pending_rng['numpy']
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            torch.set_rng_state(self.pending_rng['torch'].cpu())
            if self.pending_rng['cuda']:
                torch.cuda.set_rng_state_all([state.cpu() for state in self.pending_rng['cuda']])
            self.pending_rng = None

    def _save_checkpoint(self, trainer, filepath):
        epoch = trainer.current_epoch + 1
        epoch_dir = self.run_dir / 'epochs' / f'epoch_{epoch}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        state_path = epoch_dir / 'training_state.ckpt'
        temporary = epoch_dir / 'training_state.ckpt.tmp'
        super()._save_checkpoint(trainer, str(temporary))
        temporary.replace(state_path)
        self._last_checkpoint_saved = str(state_path)
        checkpoint_dir = self.output_cache / 'checkpoints' / self.run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg.model, checkpoint_dir / 'config.yaml', resolve=True)
        metrics = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()
                   if torch.is_tensor(value) and value.numel() == 1}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('Cannot archive nonfinite training metrics')
        weight = checkpoint_dir / f'weights_epoch_{epoch}.pt'
        atomic_torch(weight, trainer.lightning_module.model.state_dict())
        metadata = dict(self.identity, epoch=epoch, method='LeWM' if self.cfg.variant == 'lewm' else 'INTACT',
                        metrics=metrics, global_step=trainer.global_step, saved_at=time.time())
        atomic_json(checkpoint_dir / f'training_metadata_epoch_{epoch}.json', metadata)
        atomic_json(epoch_dir / 'metrics.json', metrics)
        atomic_json(epoch_dir / 'complete.json', dict(epoch=epoch, status='complete',
            entrypoint='train.py', scope='single-task', task=self.identity['task'],
            num_workers=self.identity['num_workers'], contract=self.identity, checkpoint='training_state.ckpt',
            weights=str(weight), elapsed_seconds=time.time() - self.started))
        print(json.dumps(dict(event='single_epoch_complete', epoch=epoch, task=self.identity['task'],
                              endpoint=self.identity['endpoint'], sadl=self.identity['sadl'])), flush=True)
