import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import numpy as np

class WaveformDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary with keys 'waveforms' and 'arrival_times'.
                - 'waveforms': list or np.ndarray of shape (N, L) where N is number of samples, L is waveform length
                - 'arrival_times': list or np.ndarray of shape (N,) or (N, 1) --> should become (N, L) where only bin corresponding to arrival time is 1, all others are 0
        """
        waveforms = np.asarray(data['waveforms'])
        arrival_times = np.asarray(data['arrival_times'])
        
        # Convert arrival_times to (N, L) binary array: 1 at arrival time index, 0 elsewhere
        N = waveforms.shape[0]
        L = waveforms.shape[1]
        arrival_bin = np.zeros((N, L), dtype=np.float32)
        # Ensure arrival_times is 1D array of length N
        arrival_times_flat = arrival_times.squeeze()
        if arrival_times_flat.ndim == 0:
            arrival_times_flat = np.array([arrival_times_flat])
        for i, t in enumerate(arrival_times_flat):
            # Clamp t to valid range
            t_idx = int(np.clip(t, 0, L-1))
            arrival_bin[i, t_idx] = 1.0
        arrival_times = arrival_bin

        # Ensure waveforms is 2D: (N, L)
        if waveforms.ndim == 1:
            waveforms = waveforms[:, None]
        elif waveforms.ndim > 2:
            waveforms = waveforms.reshape(waveforms.shape[0], -1)

        self.waveforms = torch.from_numpy(waveforms).float()
        self.arrival_times = torch.from_numpy(arrival_times).float()
        if self.arrival_times.ndim == 0:
            self.arrival_times = self.arrival_times.unsqueeze(0)

    def __len__(self):
        return self.waveforms.shape[0]

    def __getitem__(self, idx):
        return self.waveforms[idx], self.arrival_times[idx]