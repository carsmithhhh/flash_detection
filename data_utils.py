import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import numpy as np

class WaveformDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary with keys 'waveforms', 'arrival_times', and 'num_photons'.
                - 'waveforms': list or np.ndarray of shape (N, L)
                - 'arrival_times': list or np.ndarray where each entry can be:
                  * a single time (scalar) for one flash
                  * a list/array of times for multiple flashes
                  * None/empty for no flashes
                - 'num_photons': list or np.ndarray where each entry can be:
                  * a single time (scalar) for one hit
                  * a list/array of num_photons for multiple flashes
                  * None/empty for no flashes

        Supports multiple flashes per waveform by creating a binary indicator array
        where multiple time bins can be set to 1.0 for multiple flashes.
        """
        waveforms = np.asarray(data['waveforms'])
        arrival_times = np.asarray(data['arrival_times'])
        nphotons = np.asarray(data['num_photons'])
        offset = 0
    
        # Ensure waveforms is 2D: (N, L)
        if waveforms.ndim == 1:
            waveforms = waveforms[:, None]
        elif waveforms.ndim > 2:
            waveforms = waveforms.reshape(waveforms.shape[0], -1)
    
        N, L = waveforms.shape
        assert len(arrival_times) == N, "Mismatch between waveforms and arrival_times length"
    
        # Convert arrival_times to binary indicator array of shape (N, L)
        arrival_bin = np.zeros((N, L), dtype=np.float32)
        photon_bin = np.zeros((N, L), dtype=np.int32)
        hit_times_list = []
        photon_list = []

        for i, times in enumerate(arrival_times):
            # Handle different input formats
            if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
                # No flashes
                hit_times_list.append([])
                photon_list.append([])
                continue
                
            # Convert to list if it's a single time
            if np.isscalar(times):
                times = [times]
                photons = [nphotons[i]]
            else:
                times = np.asarray(times).flatten()
                photons = np.asarray(nphotons[i]).flatten()
            
            # Store hit times for this waveform
            hit_times_list.append(times)
            photon_list.append(photons)
            
            # Set binary indicators for all flashes in this waveform
            for j, t in enumerate(times):
                t_idx = int(np.clip(t + offset, 0, L - 1))  # Clamp to valid index range, INCLUDING OFFSET FROM WAVEFORM GEN
                arrival_bin[i, t_idx] = 1.0
                photon_bin[i, t_idx] = photons[j]
    
        # Convert to torch tensors
        self.waveforms = torch.from_numpy(waveforms).float()
        self.arrival_times = torch.from_numpy(arrival_bin).float()  # already 2D: (N, L)
        self.photon_per_times = torch.from_numpy(photon_bin).int()
        self.hit_times_list = hit_times_list
        self.photon_list = photon_list

    def __len__(self):
        return self.waveforms.shape[0]

    def __getitem__(self, idx):
        return self.waveforms[idx], self.arrival_times[idx], self.hit_times_list[idx], self.photon_per_times[idx], self.photon_list[idx]