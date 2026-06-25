"""Dataset, collation, and DataLoader helpers for waveform flash data.

A dataset on disk is a single ``.npy`` file holding a pickled ``dict`` with keys:
    ``waveforms``     - array ``(N, L)`` of waveforms,
    ``arrival_times`` - per-waveform list of flash start-time bin indices,
    ``num_photons``   - per-waveform list of photon counts (parallel to arrival_times).
See ``examples/00_simulate_dataset.ipynb`` for how these are generated.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class WaveformDataset(Dataset):
    """Waveforms with per-bin flash and photon-count targets.

    From the variable-length ``arrival_times``/``num_photons`` lists this builds, per
    waveform: a binary indicator array (1.0 at each flash start bin) and a photon-count
    array (photon count at each flash start bin). The raw per-flash lists are kept too,
    so losses can do hard-negative mining around true hit bins.

    ``__getitem__`` returns the 5-tuple
    ``(waveform, arrival_bin, hit_times, photon_bin, photon_list)``.
    """

    def __init__(self, data):
        # Pad the ragged per-waveform lists to a common width so they stack cleanly.
        max_hits = max(len(arr) for arr in data["arrival_times"])
        arrival_times = self._pad_sequences(data["arrival_times"], max_hits, pad_value=-1)
        nphotons = self._pad_sequences(data["num_photons"], max_hits, pad_value=0)

        waveforms = np.asarray(data["waveforms"])
        offset = 0  # global time offset applied when placing hits into bins

        if waveforms.ndim == 1:
            waveforms = waveforms[:, None]
        elif waveforms.ndim > 2:
            waveforms = waveforms.reshape(waveforms.shape[0], -1)

        N, L = waveforms.shape
        assert len(arrival_times) == N, "Mismatch between waveforms and arrival_times length"

        arrival_bin = np.zeros((N, L), dtype=np.float32)
        photon_bin = np.zeros((N, L), dtype=np.int32)
        hit_times_list = []
        photon_list = []

        for i, times in enumerate(arrival_times):
            if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
                hit_times_list.append([])
                photon_list.append([])
                continue

            if np.isscalar(times):
                times = [times]
                photons = [nphotons[i]]
            else:
                times = np.asarray(times).flatten()
                photons = np.asarray(nphotons[i]).flatten()

            hit_times_list.append(times)
            photon_list.append(photons)

            for j, t in enumerate(times):
                t_idx = int(np.clip(t + offset, 0, L - 1))  # clamp to valid index range
                arrival_bin[i, t_idx] = 1.0
                photon_bin[i, t_idx] = photons[j]

        self.waveforms = torch.from_numpy(waveforms).float()
        self.arrival_times = torch.from_numpy(arrival_bin).float()  # (N, L)
        self.photon_per_times = torch.from_numpy(photon_bin).int()
        self.hit_times_list = hit_times_list
        self.photon_list = photon_list

    def __len__(self):
        return self.waveforms.shape[0]

    def __getitem__(self, idx):
        return (
            self.waveforms[idx],
            self.arrival_times[idx],
            self.hit_times_list[idx],
            self.photon_per_times[idx],
            self.photon_list[idx],
        )

    @staticmethod
    def _pad_sequences(seq_list, max_len, pad_value=-1):
        """Pad each 1D sequence in ``seq_list`` to ``max_len`` with ``pad_value``."""
        padded = np.full((len(seq_list), max_len), pad_value, dtype=np.int64)
        for i, seq in enumerate(seq_list):
            seq = np.asarray(seq)
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        return padded


def custom_collate_fn(batch):
    """Collate a list of dataset items into batched, per-sample-normalized tensors.

    Returns ``(waveforms, arrival_times, hit_times, photon_bins, photon_list)`` where
    ``waveforms`` is ``[B, 1, L]`` (z-score normalized per waveform) and the bin targets
    carry a channel dim ``[B, 1, L]``. ``hit_times``/``photon_list`` are padded ``[B, max_hits]``.
    """
    waveforms, arrival_times, hit_times, photon_bins, photon_list = zip(*batch)

    waveforms = torch.stack(waveforms, dim=0)  # [B, L]
    waveforms = (waveforms - waveforms.mean(dim=1, keepdim=True)) / (
        waveforms.std(dim=1, keepdim=True) + 1e-8
    )
    waveforms = waveforms.unsqueeze(1)  # [B, 1, L]

    arrival_times = torch.stack(arrival_times, dim=0).unsqueeze(1)  # [B, 1, L]
    photon_bins = torch.stack(photon_bins, dim=0).unsqueeze(1)      # [B, 1, L]

    hit_times = torch.tensor([item[2] for item in batch])
    photon_list = torch.tensor([item[4] for item in batch])
    return waveforms, arrival_times, hit_times, photon_bins, photon_list


def split_dataset(dataset, val_ratio=0.1, test_ratio=0.0, seed=42):
    """Reproducibly split a dataset into (train, val, test) ``Subset``s."""
    g = torch.Generator().manual_seed(seed)
    total = len(dataset)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size], generator=g)


def make_dataloader(
    data_path,
    seed=42,
    batch_size=25,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    """Build a single ``DataLoader`` over the whole ``.npy`` dataset at ``data_path``."""
    load_wfs = np.load(data_path, allow_pickle=True).item()
    dataset = WaveformDataset(load_wfs)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def make_train_val_loaders(
    data_path,
    seed=42,
    batch_size=25,
    val_ratio=0.1,
    test_ratio=0.0,
    num_workers=0,
):
    """Load ``data_path``, split it, and return ``(train_loader, val_loader, test_loader)``.

    The train loader is shuffled; val/test are not. Any of the returned loaders may be
    empty if its ratio is 0. This is the entry point used by ``train.py``.
    """
    load_wfs = np.load(data_path, allow_pickle=True).item()
    dataset = WaveformDataset(load_wfs)
    train_ds, val_ds, test_ds = split_dataset(dataset, val_ratio, test_ratio, seed)
    g = torch.Generator().manual_seed(seed)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
