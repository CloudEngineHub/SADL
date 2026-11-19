"""Shared visual forward and matching logical groups for BN, SIGReg and SADL."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.utils.checkpoint import checkpoint

RUNTIME = Path(__file__).resolve().parent / 'model'
sys.path.insert(0, str(RUNTIME))
from normalization import STEP_LAYOUT, norms, replace_bn, world_size


def gather_rows(value):
    if world_size() == 1:
        return value
    sizes = STEP_LAYOUT.get('rank_windows')
    if sizes is None:
        count = torch.tensor([len(value)], device=value.device, dtype=torch.long)
        counts = [torch.zeros_like(count) for _ in range(world_size())]
        dist.all_gather(counts, count)
        sizes = [int(n.item()) for n in counts]
    maximum = max(sizes)
    if maximum == 0:
        return value
    padding = value.new_zeros((maximum - len(value),) + value.shape[1:])
    padded = torch.cat((value, padding))
    gathered = dist_nn.all_gather(padded)
    return torch.cat([v[:n] for v, n in zip(gathered, sizes)])


class MixedModel(nn.Module):
    def __init__(self, models, cfgs, contract):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.cfgs, self.contract = cfgs, contract
        if len(models) != 1:
            raise ValueError('shared-BN model owns exactly one model across every task')
        replace_bn(models[0].projector, 'global', contract['bn_momentum'])
        if contract['pred_proj_norm'] == 'batch':
            # Same global mixed-task logical groups as the visual projector.
            replace_bn(models[0].pred_proj, 'global', contract['bn_momentum'])
        for model in ([] if contract['pred_proj_norm'] == 'batch' else models):
            if any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm))
                   for m in model.pred_proj.modules()):
                raise ValueError('pred_proj must be constructed without normalization')
        if contract['checkpoint_encoder'] and hasattr(models[0].encoder, 'gradient_checkpointing_enable'):
            models[0].encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        from collapse_study import CanonicalCapacityMatching
        from module import SIGReg
        self.regularizer = CanonicalCapacityMatching(
            SIGReg(knots=17, num_proj=1024),
            output_weight=contract['sigreg_weight'], bridge_weight=contract['sadl_weight'],
            ridge=1e-3, snr=9.0, mode='reference_excitation_free_energy', demand_alpha=0.5)
        self.execution_receipt = {}
        if contract.get('execution') in ('sadl_portable', 'sadl_fast'):
            from execution import configure
            from sadl_execution import configure_streaming_sigreg
            if contract['execution'] == 'sadl_fast':
                from execution import configure_fast
                self.execution_receipt = configure_fast(models[0])
            else:
                self.execution_receipt = configure(models[0])
            configure_streaming_sigreg(self.regularizer.output_regularizer)

    def logical_groups(self, positions):
        if positions is None:
            raise ValueError('Statistics need global sample positions before rank sharding')
        boundaries = positions.new_tensor([self.contract['global_batch']]).cumsum(0)[:-1]
        return torch.bucketize(positions.contiguous(), boundaries, right=True)

    def set_normalization_groups(self, module, positions, tokens_per_window):
        if not self.training:
            return
        groups = self.logical_groups(positions)
        rows = groups.repeat_interleave(tokens_per_window)
        for normalizer in norms(module):
            normalizer.set_groups(rows, len([self.contract['global_batch']]), tokens_per_window)

    def visual(self, pixels, positions=None):
        model = self.models[0]
        batch, frames = pixels.shape[:2]
        if batch:
            images = pixels.flatten(0, 1)
            if not getattr(model, '_exec_uint8_patch', False):
                images = images.float()
            encoded = model.encoder(images,
                                    interpolate_pos_encoding=True).last_hidden_state
            h = encoded[:, 0]
        else:
            # All ranks must participate in SyncBN even with an empty tail shard.
            h = pixels.new_empty((0, self.contract['latent_dim']), requires_grad=True)
        self.set_normalization_groups(model.projector, positions, frames)
        z = model.projector(h)
        return h.reshape(batch, frames, -1 if batch else self.contract['latent_dim']), \
            z.reshape(batch, frames, -1 if batch else self.contract['latent_dim'])

    def predict_transitions(self, model, z, action):
        from collapse_study.sdpa import sdpa_context
        history = self.contract['history_size']
        transitions = z.shape[1] - 1
        if len(z):
            act = model.action_encoder(action)
            outputs = []
            prefix = min(history, transitions)
            windows = [(0, prefix, False)] + [
                (max(0, i - history + 1), i + 1, True) for i in range(prefix, transitions)]
            for start, end, last_only in windows:
                with sdpa_context(model.predictor_sdpa_backend):
                    args = (z[:, start:end], act[:, start:end])
                    if self.contract['checkpoint_predictor'] and self.training:
                        prediction = checkpoint(
                            model.predictor, *args, use_reentrant=False,
                            context_fn=lambda: (sdpa_context(model.predictor_sdpa_backend),
                                                sdpa_context(model.predictor_sdpa_backend)))
                    else:
                        prediction = model.predictor(*args)
                outputs.append(prediction[:, -1:] if last_only else prediction)
            prediction = torch.cat(outputs, dim=1)
        else:
            prediction = z[:, :transitions]
        projected = model.pred_proj(prediction.reshape(-1, z.shape[-1]))
        result = projected.reshape(len(z), transitions, z.shape[-1])
        return result + z[:, :-1] if model.predict_residual else result

    def task_loss(self, z, batch, global_count):
        from torch.nn import functional as F
        from train import construct_intents
        model, cfg = self.models[0], self.cfgs[0]
        if not len(z):
            empty = z.sum() * 0
            if self.contract['pred_proj_norm'] == 'batch' and self.training:
                # An empty tail shard must still join the pred_proj SyncBN collectives.
                prefix = min(self.contract['history_size'], z.shape[1] - 1)
                for normalizer in norms(model.pred_proj):
                    normalizer.set_groups(batch['position'].new_zeros(0), len([self.contract['global_batch']]),
                                          (z.shape[1] - prefix) * prefix)
                empty = empty + model.pred_proj(z.new_zeros((0, z.shape[-1]))).sum() * 0
            return empty, {i: z.new_zeros(5) for i in range(len(self.cfgs))}
        action = F.pad(batch['action'], (0, model.max_action_dim - batch['action'].shape[-1]))
        previous = F.pad(batch['previous_action'], (0, model.max_action_dim - batch['previous_action'].shape[-1]))
        encoded = model.action_encoder(torch.cat((action, previous), 1))
        act, prev = encoded.split(z.shape[1], 1)
        # Batch all sliding contexts. There is no task-local predictor or task-local BN.
        prefix = min(self.contract['history_size'], z.shape[1] - 1)
        starts = list(range(z.shape[1] - prefix))
        contexts = torch.cat([z[:, i:i+prefix] for i in starts])
        actions = torch.cat([act[:, i:i+prefix] for i in starts])
        raw = model.predictor(contexts, actions)
        if self.contract['pred_proj_norm'] == 'batch' and self.training:
            groups = self.logical_groups(batch['position'])
            rows = groups.repeat(len(starts)).repeat_interleave(raw.shape[1])
            for normalizer in norms(model.pred_proj):
                normalizer.set_groups(rows, len([self.contract['global_batch']]), len(starts) * raw.shape[1])
        projected = model.pred_proj(raw.flatten(0, 1)).reshape_as(raw)
        if model.predict_residual:
            projected = projected + contexts
        windows = projected.split(len(z), 0)
        prediction = torch.cat([windows[0]] + [p[:, -1:] for p in windows[1:]], 1)
        pred = (prediction.float() - z[:, 1:].float()).square().mean((1, 2))
        local, goal = construct_intents(z, model.intent_mode)
        current = torch.cat((z[:, :-1], z[:, :-1]), 0)
        mean, log_std = model.intent_actor(current, torch.cat((local, goal), 0),
                                          torch.cat((prev[:, :-1], prev[:, :-1]), 0))
        mean, log_std = mean.float(), log_std.float()
        target = torch.cat((action[:, :-1], action[:, :-1]), 0).float()
        dimensions = torch.cat((batch['action_dim'], batch['action_dim']))
        mask = torch.arange(model.max_action_dim, device=z.device)[None, :] < dimensions[:, None]
        nll = .5 * ((target - mean).square() * torch.exp(-2 * log_std) + 2 * log_std)
        nll = ((nll * mask[:, None]).sum(-1) / dimensions[:, None]).mean(1)
        mae = (((target - mean).abs() * mask[:, None]).sum(-1) / dimensions[:, None]).mean(1)
        local_nll, goal_nll = nll.chunk(2)
        local_mae, goal_mae = mae.chunk(2)
        loss = pred + cfg.loss.intent.local_weight * local_nll + cfg.loss.intent.goal_weight * goal_nll
        values = torch.stack((pred, local_nll, goal_nll, local_mae, goal_mae), 1).detach()
        metrics = {i: values[batch['task_id'] == i].sum(0) for i in range(len(self.cfgs))}
        return loss.sum() / global_count, metrics

    def _regularize_group(self, h, z, step):
        shape = (len(h), h.shape[1])
        if hasattr(self.regularizer, '_output_collapse_reference') and \
                getattr(self.regularizer, '_logical_batch_shape', None) != shape:
            # The collapse reference scales with this logical group's window
            # count, including unequal segments and the incomplete final group.
            self.regularizer._output_collapse_reference.fill_(float('nan'))
            self.regularizer._collapse_reference_valid = False
            self.regularizer._logical_batch_shape = shape
        # Use common random projections across logical groups and ranks at this
        # step. Dropout and data loading retain independent RNG streams.
        devices = [h.device.index] if h.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.contract['seed'] + 1_000_000 + step)
            with torch.autocast(device_type=h.device.type, enabled=False):
                result = self.regularizer(h.float(), z.float().transpose(0, 1))
        return result

    def regularization(self, h, z, step, positions):
        h, z, positions = gather_rows(h), gather_rows(z), gather_rows(positions)
        segments = [self.contract['global_batch']]
        host = STEP_LAYOUT.get('regularizer_windows')
        if host is not None and len(segments) == 1:
            groups, counts = None, host
        else:
            boundaries = positions.new_tensor(segments).cumsum(0)[:-1]
            groups = torch.bucketize(positions.contiguous(), boundaries, right=True)
            counts = torch.bincount(groups, minlength=len(segments)).tolist()
        result = {}
        for group, n in enumerate(counts):
            if n == 0:
                continue
            if groups is None:
                values = self._regularize_group(h, z, step)
            else:
                selected = groups == group
                values = self._regularize_group(h[selected], z[selected], step)
            weight = n / len(h)
            for key, value in values.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    result[key] = result.get(key, 0) + value * weight
        active = [n for n in counts if n > 0]
        result['regularizer_groups'] = h.new_tensor(len(active))
        result['regularizer_min_group_windows'] = h.new_tensor(min(active))
        result['regularizer_max_group_windows'] = h.new_tensor(max(active))
        return result, h, z


def global_count(batch, device):
    if 'valid_windows' in STEP_LAYOUT:
        return STEP_LAYOUT['valid_windows']
    count = torch.tensor(len(batch['pixels']), device=device, dtype=torch.long)
    if world_size() > 1:
        dist.all_reduce(count)
    return int(count.item())


def autocast_for(model):
    device = next(model.parameters()).device
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=model.contract['precision'] == 'bf16')


def backward_step(model, batches, step):
    for normalizer in norms(model):
        normalizer.begin_step()
    device = next(model.parameters()).device
    count = sum(global_count(batch, device) for batch in batches)
    metrics = {i: torch.zeros(5, device=device) for i in range(len(model.cfgs))}
    task_counts = torch.zeros(len(model.cfgs), device=device, dtype=torch.long)
    for batch in batches:
        task_counts += torch.bincount(batch['task_id'], minlength=len(model.cfgs))
    if len(batches) == 1:
        batch = batches[0]
        with autocast_for(model):
            h, z = model.visual(batch['pixels'], batch['position'])
            task_loss, values = model.task_loss(z, batch, count)
        result, all_h, all_z = model.regularization(h, z, step, batch['position'])
        # all_gather backward sums the replicated regularizer gradients. Scaling
        # by world size and SUM-reducing parameter gradients gives the exact
        # global objective, including uneven/empty tail shards.
        loss = task_loss + result['loss'] / world_size()
        # Deferred check: raised at the next logging step without a per-step device sync.
        STEP_LAYOUT['nonfinite'] = STEP_LAYOUT['nonfinite'] | ~torch.isfinite(loss.detach()) \
            if 'nonfinite' in STEP_LAYOUT else ~torch.isfinite(loss.detach())
        loss.backward()
        metrics = values
    else:
        if world_size() != 1:
            raise ValueError('Accumulation cells in this matrix use one GPU')
        cached_h, cached_z, random_states = [], [], []
        for batch in batches:
            state = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            random_states.append(state)
            with torch.no_grad(), autocast_for(model):
                h, z = model.visual(batch['pixels'], batch['position'])
            cached_h.append(h.detach())
            cached_z.append(z.detach())
        h = torch.cat(cached_h).requires_grad_()
        z = torch.cat(cached_z).requires_grad_()
        positions = torch.cat([batch['position'] for batch in batches])
        result, all_h, all_z = model.regularization(h, z, step, positions)
        dh, dz = torch.autograd.grad(result['loss'], (h, z))
        for normalizer in norms(model):
            normalizer.rewind_step()
        offset = 0
        for batch, state in zip(batches, random_states):
            devices = [device.index] if device.type == 'cuda' else []
            # Replay only the visual RNG. Restoring the entire step RNG before
            # each head forward would repeat predictor dropout across micros.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(state[0])
                if state[1] is not None:
                    torch.cuda.set_rng_state(state[1], device)
                with autocast_for(model):
                    replay_h, replay_z = model.visual(batch['pixels'], batch['position'])
            with autocast_for(model):
                task_loss, values = model.task_loss(replay_z, batch, count)
            n = len(replay_h)
            surrogate = (replay_h.float() * dh[offset:offset+n]).sum()
            surrogate = surrogate + (replay_z.float() * dz[offset:offset+n]).sum()
            loss = task_loss + surrogate
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite accumulated objective')
            loss.backward()
            offset += n
            for i in metrics:
                metrics[i] += values[i]
    for normalizer in norms(model):
        normalizer.commit_step()
    if world_size() > 1:
        dist.all_reduce(task_counts)
        for value in metrics.values():
            dist.all_reduce(value)
    diagnostics = {k: v.detach() for k, v in result.items() if torch.is_tensor(v) and v.numel() == 1}
    diagnostics['regularizer_loss'] = diagnostics['loss']
    data_loss = sum(values[0] + model.cfgs[i].loss.intent.local_weight * values[1]
                    + model.cfgs[i].loss.intent.goal_weight * values[2]
                    for i, values in metrics.items()) / count
    diagnostics['task_loss'] = data_loss
    diagnostics['loss'] = data_loss + diagnostics['regularizer_loss']
    diagnostics['global_count'] = count
    diagnostics['task_counts'] = task_counts
    diagnostics['task_metrics'] = metrics
    if step % 500 == 0:
        with torch.no_grad():
            for name, value in [('h', all_h), ('z', all_z)]:
                flat = value.detach().float().flatten(0, 1)
                flat = flat - flat.mean(0)
                eigen = torch.linalg.eigvalsh(flat.T @ flat / max(1, len(flat) - 1)).clamp_min(0)
                p = eigen / eigen.sum().clamp_min(1e-20)
                diagnostics[name + '_effective_rank'] = (-(p * p.clamp_min(1e-20).log()).sum()).exp()
                diagnostics[name + '_variance'] = eigen.mean()
    return diagnostics


def sum_gradients(parameters, max_elements=2_000_000):
    parameters = list(parameters)
    if world_size() == 1:
        return
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                        dtype=torch.int32)
    dist.all_reduce(used)
    bucket, size = [], 0

    def reduce_bucket(items):
        if not items:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in items])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in items:
            gradient = flat[offset:offset+p.numel()].view_as(p)
            if p.grad is None:
                p.grad = gradient.clone()
            else:
                p.grad.copy_(gradient)
            offset += p.numel()

    for p, active in zip(parameters, used.tolist()):
        if not active:
            continue
        if size + p.numel() > max_elements:
            reduce_bucket(bucket)
            bucket, size = [], 0
        bucket.append(p)
        size += p.numel()
    reduce_bucket(bucket)


class OverlappedGradientSum:
    """Sum gradients over ranks while backward runs. Buckets have a fixed order and are launched strictly in that
    order once all their gradients are ready, so every rank issues identical collectives. Parameters whose gradient
    is None on every rank at the first step are excluded and keep grad None (as in sum_gradients)."""

    def __init__(self, parameters, max_elements=2_000_000):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.max_elements = max_elements
        self.buckets = None

    def build(self):
        used = torch.tensor([p.grad is not None for p in self.parameters], device=self.parameters[0].device,
                            dtype=torch.int32)
        dist.all_reduce(used)
        active = [p for p, u in zip(self.parameters, used.tolist()) if u]
        self.buckets, bucket, size = [], [], 0
        for p in reversed(active):
            if bucket and size + p.numel() > self.max_elements:
                self.buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(p)
            size += p.numel()
        if bucket:
            self.buckets.append(bucket)
        self.index = {p: i for i, bucket in enumerate(self.buckets) for p in bucket}
        for p in active:
            p.register_post_accumulate_grad_hook(self._ready)
        self.reset()

    def reset(self):
        self.pending = [len(b) for b in self.buckets]
        self.next = 0
        self.works = []

    def _launch_ready(self):
        while self.next < len(self.buckets) and self.pending[self.next] == 0:
            bucket = self.buckets[self.next]
            flat = torch.cat([p.grad.reshape(-1) for p in bucket])
            self.works.append((bucket, flat, dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)))
            self.next += 1

    def _ready(self, p):
        if self.buckets is None:
            return
        self.pending[self.index[p]] -= 1
        self._launch_ready()

    def finish(self):
        if self.buckets is None:
            sum_gradients(self.parameters)
            self.build()
            return
        for i, bucket in enumerate(self.buckets):
            if self.pending[i]:
                for p in bucket:
                    if p.grad is None:  # e.g. an empty tail shard on this rank
                        p.grad = torch.zeros_like(p)
                self.pending[i] = 0
        self._launch_ready()
        for bucket, flat, work in self.works:
            work.wait()
            offset = 0
            for p in bucket:
                p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
                offset += p.numel()
        self.reset()
