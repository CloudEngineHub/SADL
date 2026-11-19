"""A global window stream is constructed before any GPU partitioning."""
from __future__ import annotations

import bisect
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def epoch_order(lengths, sampling, seed):
    if not lengths or min(lengths) < 1:
        raise ValueError('Each task needs at least one training window')
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0] + list(lengths))
    if sampling == 'proportional':
        return rng.permutation(int(offsets[-1]))
    if sampling != 'balanced':
        raise ValueError(sampling)
    # Cover every task: largest task exactly once, smaller tasks cycle.
    quotas = [max(lengths)] * len(lengths)
    chunks = []
    for offset, size, target in zip(offsets, lengths, quotas):
        cycles = [rng.permutation(size) + offset for _ in range(math.ceil(target / size))]
        chunks.append(np.concatenate(cycles)[:target])
    full = min(quotas)
    ordered = np.stack([chunk[:full] for chunk in chunks], axis=1)
    # Shuffle each four-sample tuple; task identity must not equal rank modulo four.
    keys = rng.random(ordered.shape)
    ordered = np.take_along_axis(ordered, keys.argsort(axis=1), axis=1).reshape(-1)
    tail = np.concatenate([chunk[full:] for chunk in chunks])
    return np.concatenate([ordered, tail])


class GlobalBatches(Sampler):
    def __init__(self, order, world, rank, micro, accumulation, start_step=0):
        self.order = order
        self.world, self.rank = world, rank
        self.micro, self.accumulation = micro, accumulation
        self.global_batch = world * micro * accumulation
        self.steps = math.ceil(len(order) / self.global_batch)
        self.start_step = start_step

    def __len__(self):
        return (self.steps - self.start_step) * self.accumulation

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            base = step * self.global_batch
            for part in range(self.accumulation):
                start = base + part * self.world * self.micro
                stop = min(start + self.world * self.micro, len(self.order))
                yield [(int(self.order[i]), i - base)
                       for i in range(start + self.rank, stop, self.world)]


class IndexedSubset(Dataset):
    def __init__(self, dataset, index_file):
        self.dataset, self.index_file = dataset, str(index_file)
        self.indices = np.load(self.index_file, mmap_mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('indices')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.indices = np.load(self.index_file, mmap_mode='r')


class MixedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets]).tolist()

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        index, position = index
        task = bisect.bisect_right(self.offsets, index) - 1
        row = dict(self.datasets[task][index - self.offsets[task]])
        return {'task_id': task, 'sample_id': index, 'position': position,
                **{k: row[k] for k in ('pixels', 'action', 'previous_action')}}


def collate_mixed(rows):
    if not rows:
        return None
    action_dim = max(torch.as_tensor(row['action']).shape[-1] for row in rows)
    out = {name: torch.tensor([row[name] for row in rows], dtype=torch.long)
           for name in ('task_id', 'sample_id', 'position')}
    out['action_dim'] = torch.tensor([row['action'].shape[-1] for row in rows], dtype=torch.long)
    first = torch.as_tensor(rows[0]['pixels'])
    pixels = torch.empty((len(rows),) + tuple(first.shape), dtype=first.dtype)
    if torch.utils.data.get_worker_info() is not None:
        pixels.share_memory_()  # filled in place; the main process receives it without a copy
    torch.stack([torch.as_tensor(row['pixels']) for row in rows], out=pixels)
    out['pixels'] = pixels
    for name in ('action', 'previous_action'):
        values = []
        for row in rows:
            value = torch.as_tensor(row[name])
            values.append(torch.nn.functional.pad(value, (0, action_dim - value.shape[-1])))
        out[name] = torch.stack(values)
    return out


def save_split(directory, task, count, fraction, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train_file = directory / f'{task}_train.npy'
    val_file = directory / f'{task}_val.npy'
    # Match torch random_split's rounded fractional sizes, with one seed per
    # task, independent of rank and of the other selected tasks.
    count_train = math.ceil(count * fraction)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).numpy()
    np.save(train_file, order[:count_train])
    np.save(val_file, order[count_train:])
    return train_file, val_file
