'''
Script that runs performance studies on trained models. Can toggle studies to run with flags in performance_analysis_config.yaml

Single Flash Performance Study ("Upper Bound"):
- merged accuracy vs. nphotons
- merged purity vs. nphotons
- merged window width vs. nphotons
- reconstructed photon fraction vs. nphotons


2-Flash Delta T Study: [-random, -10, -50, -100, -500, or -1000 photons]
- merged accuracy flash 1 vs. delta t
- merged accuracy flash 2 vs. delta t
- reconstructed photon fraction flash 1 vs. delta t
- reconstructed photon fraction flash 2 vs. delta t
- merged purity vs. delta t
- merged window width vs. delta t
'''

import sys
sys.path.append('..')
sys.path.append('../..')
from data_utils import *
from waveforms.make_waveform import BatchedLightSimulation

import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import importlib
import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import yaml

from transformer import *
from hybrid_loss import *
from model import *
from evaluation import *
from conformer import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using CUDA

# Load Analysis Config
with open("performance_analysis_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Models
device='cuda'

MODEL_CLASSES = {
    "TransformerModel": TransformerModel,
    "ConformerModel": ConformerModel,
    "ConformerModelv2": ConformerModelv2,
    "UNet1D": UNet1D
}

models = load_models("model_list.yaml", MODEL_CLASSES)
print(models.keys())

###################################### Single Flash ###########################################
if any(config["single_flash"].values()):
    
    merge_loader = make_dataloader("/sdf/home/c/carsmith/sdf_data/flash_detection_data/flash_files/1delayphot_reg.npy", seed=42, batch_size=25, shuffle=False)
    print("Single Flash Loader Length: ", len(merge_loader))
    bin_width = 100
    batches_per_photon = 4
    single_phot_results = {name: {"merge_acc": [], "merge_pure": [], "interval": [], "reco_frac": []} for name in models.keys()}
    epochs = 1
    
    for epoch in range(epochs):
        acc_progress = tqdm(merge_loader, desc=f"Scanning {epoch+1}/{epochs}", leave=False, position=0)
    
        # temporary accumulators
        interval_bins = {name: [] for name in models.keys()}
        reco_frac = {name: 0.0 for name in models.keys()}
        merged_acc = {name: 0.0 for name in models.keys()}
        merged_pure = {name: 0.0 for name in models.keys()}
        avg_interval = None
    
        with torch.no_grad():
            for i, (data, target, hit_times, photon_target, photon_list) in enumerate(acc_progress):
                data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)
    
                # loop through all models in dict
                for name, (model, reg_loss) in models.items():
                    predict_fn = torch.expm1 if reg_loss == 'mse' else torch.exp
                    
                    class_output, reg_output = model(data, mode="bce")
                    merged_mask = merge_bins(class_output, skip_tol=0)
    
                    # interval widths
                    for b in range(merged_mask.shape[0]):
                        mask_row = merged_mask[b, 0]
                        intervals = mask_to_intervals(mask_row)
                        widths = [(e - s + 1) for (s, e) in intervals]
                        interval_bins[name].extend(widths)
        
                    if config["single_flash"]["reco_frac"]:
                        interval_pred_sums = sum_photons_in_intervals(predict_fn(reg_output), merged_mask)
                        interval_true_sums = sum_photons_in_intervals(photon_target, merged_mask)
                        pred = torch.tensor([np.sum(x) for x in interval_pred_sums])
                        true = torch.tensor([np.sum(x) for x in interval_true_sums])
                        mask = true > 0
                        if mask.any():
                            reco_frac[name] += torch.mean(pred[mask] / true[mask]).item()
        
                    if config["single_flash"]["merged_acc"]:
                        merged_acc[name] += merged_class_acc(merged_mask, hit_times, device)
                        
                    if config["single_flash"]["merged_pure"]:
                        merged_pure[name] += merged_class_purity(merged_mask, hit_times, device)
                        
                    if config["single_flash"]["merged_window_width"]:
                        continue
                    
                if (i + 1) % batches_per_photon == 0:
                    for name in models.keys():
                        reco_frac[name] /= batches_per_photon
                        merged_acc[name] /= batches_per_photon
                        merged_pure[name] /= batches_per_photon
                        
                        if config["single_flash"]["merged_window_width"]: 
                            avg_interval = np.mean(interval_bins[name]) if interval_bins[name] else 0.0
    
                        single_phot_results[name]["interval"].append(avg_interval)
                        single_phot_results[name]["reco_frac"].append(reco_frac[name])
                        single_phot_results[name]["merge_acc"].append(merged_acc[name])
                        single_phot_results[name]["merge_pure"].append(merged_pure[name])
    
                        # reset accumulators
                        interval_bins[name] = []
                        reco_frac[name] = 0.0
                        merged_acc[name] = 0.0
                        merged_pure[name] = 0.0
    
    np.save("notebooks/performance_analysis/con_v5mse_v5+_singleflash_nomerge.npy", single_phot_results, allow_pickle=True)

###################################### Double Flash ###########################################
bool_values = [v for v in config["double_flash"].values() if isinstance(v, bool)]
if any(bool_values):

    delta_name = config["double_flash"]["fixed_photon"]
    delta_loader = make_dataloader(f"/sdf/home/c/carsmith/sdf_data/flash_detection_data/flash_files/delayed_delta_t/delta_t_{delta_name}phot.npy", seed=42, batch_size=25, shuffle=False)
    print("Delta Loader Length: ", len(delta_loader))
    
    delta_results = {
        name: {
            "bin_counts": torch.zeros(1501),
            "reco_frac_flash1": torch.zeros(1501),
            "reco_frac_flash2": torch.zeros(1501),
            "merge_acc_flash1": torch.zeros(1501),
            "merge_acc_flash2": torch.zeros(1501),
            "merge_pure": torch.zeros(1501)
        }
        for name in models.keys()
    }
    epochs = 1
    flash1_acc_ls = np.zeros(delta_loader.batch_size)
    flash2_acc_ls = np.zeros(delta_loader.batch_size)
    purity_ls = np.zeros(delta_loader.batch_size)
    
    for epoch in range(epochs):
        acc_progress = tqdm(delta_loader, desc=f"Scanning {epoch+1}/{epochs}", leave=False, position=0)
    
        with torch.no_grad():
            for i, (data, target, hit_times, photon_target, photon_list) in enumerate(acc_progress):
                data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)
    
                indices = torch.tensor([int(t[1] - t[0]) for t in hit_times], dtype=torch.long)
    
                for name, (model, reg_loss) in models.items():
                    predict_fn = torch.expm1 if reg_loss == 'mse' else torch.exp
                    
                    class_output, reg_output = model(data, mode="bce")
                    merged_mask = merge_bins(class_output, skip_tol=5)
    
                    if config["double_flash"]["reco_frac"]:
                        interval_pred_sums = sum_photons_in_intervals(predict_fn(reg_output), merged_mask)
                        interval_true_sums = sum_photons_in_intervals(photon_target, merged_mask)
    
                    if config["double_flash"]["merged_acc"]:
                        flash1_acc_ls, flash2_acc_ls = merged_twoflash_acc(merged_mask, hit_times, device, no_sum=True)
                        
                    if config["double_flash"]["merged_pure"]:
                        purity_ls = merged_class_purity(merged_mask, hit_times, device, no_sum=True)
    
                    for b, idx in enumerate(indices):
                        delta_results[name]["bin_counts"][idx] += 1
    
                        if config["double_flash"]["reco_frac"]:
                            mask = np.array(interval_true_sums[b]) > 0
                            valid_idx = np.where(mask)
                            if len(valid_idx[0]) >= 2:
                                delta_results[name]["reco_frac_flash1"][idx] += (interval_pred_sums[b][valid_idx[0][0]] / interval_true_sums[b][valid_idx[0][0]]).item()
                                delta_results[name]["reco_frac_flash2"][idx] += (interval_pred_sums[b][valid_idx[0][1]] / interval_true_sums[b][valid_idx[0][1]]).item()
                        delta_results[name]["merge_acc_flash1"][idx] += flash1_acc_ls[b]
                        delta_results[name]["merge_acc_flash2"][idx] += flash2_acc_ls[b]
                        delta_results[name]["merge_pure"][idx] += purity_ls[b]
    
    for name in models.keys():
        counts = delta_results[name]["bin_counts"].clone()
        counts[counts == 0] = 1  # prevent division by zero
        for key in delta_results[name]:
            if key != "bin_counts":
                delta_results[name][key] /= counts

        print("total merge pure: ", delta_results[name]["merge_pure"].mean().item())
    np.save("notebooks/performance_analysis/conv5drop_plus_100phot_deltastats.npy", delta_results, allow_pickle=True)