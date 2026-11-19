"""Native BN/SyncBN kernels with explicit optimizer-step buffer updates."""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


# Host-side layout of the current optimizer step (set by train_mixed before each step).
STEP_LAYOUT = {}


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


class StepBatchNorm(nn.SyncBatchNorm):
    def __init__(self, source, scope, momentum):
        super().__init__(source.num_features, eps=source.eps, momentum=1.0,
                         affine=source.affine, track_running_stats=True)
        self.load_state_dict(source.state_dict(), strict=True)
        self.scope, self.step_momentum = scope, momentum
        self._start = None
        self._groups = None

    def set_groups(self, row_groups, num_groups, tokens_per_window=None):
        self._groups, self._num_groups = row_groups, num_groups
        self._tokens = tokens_per_window

    def begin_step(self):
        self._start = (self.running_mean.clone(), self.running_var.clone(),
                       self.num_batches_tracked.clone())
        self._reset_accumulator()

    def _reset_accumulator(self):
        self._sum = torch.zeros_like(self.running_mean, dtype=torch.float64)
        self._variance_sum = torch.zeros_like(self._sum)
        self._count = 0
        self._dof = 0
        self._seen_groups = set()

    def rewind_step(self):
        with torch.no_grad():
            for target, source in zip((self.running_mean, self.running_var,
                                       self.num_batches_tracked), self._start):
                target.copy_(source)
        self._reset_accumulator()

    def forward(self, x):
        if not self.training:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        if self._start is None:
            raise RuntimeError('begin_step must precede training BN forwards')
        if x.ndim != 2:
            raise ValueError('Only vector MLP BN is changed by this implementation')
        if self._groups is None or len(self._groups) != len(x):
            raise ValueError('Explicit logical BN group IDs are required for every row')
        global_scope = self.scope in ('global', 'sync', 'sync_global')
        local_scope = self.scope in ('local', 'rank_local')
        if not (global_scope or local_scope):
            raise ValueError(f'Unknown BN scope: {self.scope}')
        layout = STEP_LAYOUT.get('group_windows')
        if layout is not None and self._tokens is not None and global_scope:
            counts = [n * self._tokens for n in layout]  # host-known, no device sync
        else:
            counts = torch.bincount(self._groups, minlength=self._num_groups)
            if global_scope and world_size() > 1:
                dist.all_reduce(counts)
            counts = counts.tolist()
        if self._num_groups == 1 and counts[0] >= 2 and global_scope:
            # One logical group: identical to selecting every row.
            normalized = (super().forward(x) if world_size() > 1 else
                          F.batch_norm(x, self.running_mean, self.running_var,
                                       self.weight, self.bias, True, 1.0, self.eps))
            with torch.no_grad():
                n = counts[0]
                self._sum.add_(self.running_mean.double(), alpha=n)
                self._variance_sum.add_(self.running_var.double(), alpha=n - 1)
                self._count += n
                self._dof += n - 1
            return normalized
        out = x * 0
        for group, n in enumerate(counts):
            if n == 0:
                continue
            if n < 2:
                raise ValueError('BN requires at least two real visual/prediction tokens')
            if group in self._seen_groups:
                raise ValueError('A logical BN group was split across sequential forwards')
            self._seen_groups.add(group)
            indices = torch.where(self._groups == group)[0]
            selected = x.index_select(0, indices)
            if global_scope and world_size() > 1:
                normalized = super().forward(selected)
            else:
                normalized = F.batch_norm(selected, self.running_mean, self.running_var,
                                          self.weight, self.bias, True, 1.0, self.eps)
            # Keep empty local groups in autograd: their SyncBN backward must
            # still participate when other ranks own rows in this logical group.
            out = out.index_copy(0, indices, normalized)
            with torch.no_grad():
                self._sum.add_(self.running_mean.double(), alpha=n)
                self._variance_sum.add_(self.running_var.double(), alpha=n - 1)
                self._count += n
                self._dof += n - 1
        return out

    def commit_step(self):
        with torch.no_grad():
            # Every rank already has the same synchronized logical-group stats.
            self.running_mean.copy_(self._start[0])
            self.running_var.copy_(self._start[1])
            self.num_batches_tracked.copy_(self._start[2])
            if self._count > 0:
                self.running_mean.lerp_((self._sum / self._count).float(), self.step_momentum)
                self.running_var.lerp_((self._variance_sum / self._dof).float(), self.step_momentum)
                self.num_batches_tracked.add_(1)
        self._start = None
        self._groups = None


def replace_bn(module, scope, momentum):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            setattr(module, name, StepBatchNorm(child, scope, momentum))
        else:
            replace_bn(child, scope, momentum)


def norms(module):
    return [m for m in module.modules() if isinstance(m, StepBatchNorm)]
