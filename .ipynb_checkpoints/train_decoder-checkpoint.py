import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import pickle
import wandb

from decoding_model import *
from data_utils import *
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using CUDA

# Loading in data
load_wfs = np.load('/sdf/home/c/carsmith/sdf_data/flash_detection_data/flash_files/delay_200ks/2_8.npy', allow_pickle=True)
dataset = WaveformDataset(load_wfs.item())

g = torch.Generator()
g.manual_seed(seed)

# Splitting data
val_ratio = 0.75
total_size = len(dataset)
val_size = int(total_size * val_ratio)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

batch_size = 25

def decoding_collate_fn(batch):
    '''
    Each item in batch is a tuple: (waveform, arrival_times, hit_times, photon_bins, photon_list)
    batch_size = 25

    We need to construct: 
        Normalized waveforms for training - Normalized signal amplitude, maybe also width? not sure 

        Targets for evaluating loss:
        targets[b] = {
            "start_time": tensor of shape [num_gt_b],  # normalized 0..1
            "duration": tensor of shape [num_gt_b],    # normalized 0..1
            "photons": tensor of shape [num_gt_b]      # scalar
        }
    '''
    waveforms, arrival_times, hit_times, photon_bins, photon_list = zip(*batch) # all of these are tuples of length 25
    batch_size = len(waveforms)
    window_len = waveforms[0].shape[0]

    # Stack & normalize waveforms
    waveforms = torch.stack(waveforms, dim=0) # [B, L]
    waveforms = (waveforms - waveforms.mean(dim=1, keepdim=True)) / (waveforms.std(dim=1, keepdim=True) + 1e-8)
    waveforms = waveforms.unsqueeze(1)

    # Build targets
    targets = []
    for b in range(batch_size):
        hit_times_b = torch.tensor(hit_times[b], dtype=torch.float32) # start times, length 8, pads with 0's
        hit_times_b = hit_times_b[hit_times_b > 0]
        
        photon_counts = torch.tensor(photon_list[b], dtype=torch.float32) # scalar photons per pulse
        photon_counts = photon_counts[photon_counts > 0]
        # photon_counts = F.pad(photon_counts, (0, 8 - photon_counts.shape[0]))  # pad with zeros on the right

        num_real_pulses = hit_times_b.shape[0]
        
        durations = torch.full((num_real_pulses,), 800.0, dtype=torch.float32) # all pulses have same width... not really learning anything
        # durations = F.pad(durations, (0, 8 - durations.shape[0]))
        
        # Normalize to [0,1] relative to waveform length
        start_norm = hit_times_b / window_len
        duration_norm = durations / window_len

        target = {
            "start_time": start_norm,
            "duration": duration_norm,
            "photons": photon_counts
        }
        targets.append(target)
    
    return waveforms, targets

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    collate_fn=decoding_collate_fn,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=decoding_collate_fn,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

epochs = 30
batch_size = 25

logger = wandb.init(
    project="decoding_test",
    name="50k_nozeros",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 1e-4,
    }
)

device = 'cuda'

model = DecodingModel()
model.to(device)
model.train()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
wandb.watch(model, log="all", log_freq=100)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

train_progress = tqdm(train_loader, desc=f"Epoch {1}/{epochs}", leave=False, position=0, disable=True)
val_progress = tqdm(val_loader, desc='Validating', leave=False, position=0, disable=True)

for epoch in range(epochs):
    train_loss = 0.0
    signal_loss = 0.0
    reg_loss = 0.0
    conf_loss = 0.0
    
    for i, (waveforms, targets) in enumerate(train_progress):
        model.train()
        waveforms = waveforms.to(device)
    
        signal_output, reg_output, confidence_output = model(waveforms)
    
        loss, loss_signal, loss_photon, confidence_loss = set_based_loss(signal_output, reg_output, confidence_output, targets)
        train_loss += loss
        signal_loss += loss_signal
        conf_loss += confidence_loss
        reg_loss += loss_photon
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_progress.set_postfix({"train_loss": train_loss/(i+1)})
        
    logger.log({
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader),
        "signal_loss": signal_loss / len(train_loader),
        "reg_loss": reg_loss / len(train_loader),
        "conf_loss": conf_loss / len(train_loader),
        # "grad_norm": _compute_grad_norm(model.parameters(), norm_type=2)
    })

    # if epoch % 1 == 0:
    #     val_loss = 0.0
    #     val_signal_loss = 0.0
    #     val_reg_loss = 0.0
    #     val_conf_loss = 0.0
        
    #     model.eval()
    #     for i, (waveforms, targets) in enumerate(val_progress):
    #         waveforms = waveforms.to(device)
    #         signal_output, reg_output, _ = model(waveforms)
        
    #         loss, loss_signal, loss_photon, confidence_loss = set_based_loss(signal_output, reg_output, targets)
    #         val_loss += loss
    #         val_signal_loss += loss_signal
    #         val_reg_loss += loss_photon
    #         val_conf_loss += confidence_loss
            
    #     logger.log({
    #         "epoch": epoch,
    #         "val_loss": val_loss / len(val_loader),
    #         "val_signal_loss": val_signal_loss / len(val_loader),
    #         "val_reg_loss": val_reg_loss / len(val_loader),
    #         "val_conf_loss": val_conf_loss / len(val_loader),
    #     })

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, f"/sdf/home/c/carsmith/sdf_data/flash_detection_data/decoder_path.pth")

wandb.finish()