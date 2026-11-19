"""Dataset-local HDF5 pixel reads: contiguous window read, in-memory frame subsampling."""

from __future__ import annotations

import numpy as np
import torch
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


class StridedPixelHDF5Dataset(HDF5Dataset):
    """HDF5 reader that loads a window once and keeps every frameskip-th frame."""

    @classmethod
    def from_dataset(cls, dataset: HDF5Dataset) -> "StridedPixelHDF5Dataset":
        adapted = cls.__new__(cls)
        # Reuse the existing column cache, clip indices, transform, and SWMR
        # handle. The inherited pickle hook still clears the worker handle.
        adapted.__dict__ = dataset.__dict__.copy()
        return adapted

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            if col == "pixels":
                column = src[col]
                if col in self._cache or self.frameskip == 1:
                    data = column[g_start:g_end:self.frameskip]
                else:
                    # h5py's multidimensional strided selection is expensive
                    # for 100-frame image chunks. Contiguous frame reads reuse
                    # its chunk cache while copying only the sampled frames.
                    # One contiguous read decompresses each 100-frame chunk once; then
                    # keep every frameskip-th frame in memory (identical bytes).
                    data = column[int(g_start):int(g_end)][::self.frameskip]
            else:
                data = src[col][g_start:g_end]
                if col != "action":
                    data = data[::self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps


def adapt_hdf5_pixel_reads(dataset):
    """Apply only to the concrete built-in HDF5 reader, leaving other formats alone."""
    if type(dataset) is HDF5Dataset:
        return StridedPixelHDF5Dataset.from_dataset(dataset)
    return dataset
