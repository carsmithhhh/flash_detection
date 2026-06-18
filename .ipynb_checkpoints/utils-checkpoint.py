from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import wandb

from model import UNet1D
from hybrid_loss import *
import evaluation
from evaluation import *

def train_regression(model, train_loader, criterion, optimizer, device, good_conformer, use_wandb=False, epochs=500):
    model.train()
    
    reg_output_list = []
    reg_loss_list = []

    if use_wandb:
        wandb.init(project="conformer_oct", name="v5_drop_gain", config={
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__
        })
        wandb.watch(model, log="all", log_freq=1000)
        
    for epoch in range(epochs):
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0)

        for batch_idx, (data, target, hit_times, photon_target, photon_list) in enumerate(train_progress):
            data = data.to(device)
            target = target.to(device)
            photon_target = photon_target.to(device)

            # forward pass
            class_output, _ = good_conformer(data, mode='bce') # use trained classification for reg sampling
            _, reg_output = model(data, mode='bce')

            # do regression on merged windows
            merged_mask = merge_bins(class_output, skip_tol=5)
            summed_interval_output = sum_photons_in_intervals_vecwgrad(reg_output, merged_mask)   # list of list of tensors
            summed_interval_target = sum_photons_in_intervals_vecwgrad(photon_target, merged_mask) # list of list of tensors (no grad needed)
            masked_reg_output = torch.cat([t.flatten() for sublist in summed_interval_output for t in sublist])
            masked_photon_target = torch.cat([t.flatten() for sublist in summed_interval_target for t in sublist])
                
            # loss
            reg_loss = criterion(F.softplus(masked_reg_output), masked_photon_target)
            
            # backward pass
            optimizer.zero_grad()
            reg_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # OPTIONAL: CLIP GRADIENTS
            optimizer.step()

            if use_wandb:
                norm = evaluation._compute_grad_norm(model.parameters(), norm_type=2)
                wandb.log({
                    "train/reg_loss": reg_loss.item(),
                    "epoch": epoch,
                    "grad_norm": norm,
                })

            # logging
            if batch_idx == 0:
                reg_loss_list.append(reg_loss.item())
                reg_output_list.append(reg_output.detach().cpu())

    if use_wandb:
        wandb.finish()

    return reg_loss_list, reg_output_list

# 10.9 TESTING --> good_conformer for pure classification filtering
def train_model_2(model, train_loader, val_loader, scheduler, optimizer, device, epochs, mode='bce', wandb_logger=None, mse=False):
    model.train()
    optimizer.zero_grad()

    results = {}
    results['train_loss'] = []
    results['train_acc'] = []
    results['train_pure'] = []
    results['eval_loss'] = []
    results['eval_acc'] = []
    results['eval_pure'] = []
    results['train_reg_rmse'] = []
    results['eval_reg_rmse'] = []
    results['reg_output'] = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_pure = 0.0
        train_merged_pure = 0.0
        train_reg_rmse = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)

        for i, (data, target, hit_times, photon_target, photon_list) in enumerate(train_progress):
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device) # both [B, 1, 16000]
            class_output, reg_output = model(data, mode='bce')
            # PURE CLASSIFICATION
            # filtered_class_output, _ = good_conformer(data, mode='bce') # use trained classification for reg sampling
            # class_output, reg_output = model(data, mode='bce')

            if mode == 'mined_bce':
                ### Ensure class_output matches data shape - hack added for unet since 1000 us is not power of 2
                if class_output.shape[-1] != data.shape[-1]:
                    diff = data.shape[-1] - class_output.shape[-1]
                    if diff > 0:
                        class_output = F.pad(class_output, (0, diff))
                        reg_output = F.pad(reg_output, (0, diff))
                    else:
                        class_output = class_output[..., :data.shape[-1]]
                        reg_output = reg_output[..., :data.shape[-1]]
                
                acc = overall_class_acc(hit_times, class_output, device)
                purity = overall_class_purity(hit_times, class_output, device)
                
                include_reg = True
                if epoch > 1:
                    include_reg = True
                loss, _, _, _, _, _, _, _ = mined_bce_loss(data, hit_times, photon_list, class_output, reg_output, epoch, device, include_reg)
                reg_rmse = regression_rmse(hit_times, photon_target, reg_output, class_output, device, mse=mse)

            # step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # averaged over the batch already
            train_loss += loss.item() 
            train_acc += acc
            train_pure += purity
            train_reg_rmse += reg_rmse
            
            train_progress.set_postfix({"train_loss": train_loss/(i+1), "train_acc": train_acc/(i+1), "train_pure": train_pure/(i+1)})

        # divide by number of batches seen
        train_loss /= len(train_loader) 
        train_acc /= len(train_loader)
        train_pure /= len(train_loader)
        train_reg_rmse /= len(train_loader)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['train_pure'].append(train_pure)
        results['train_reg_rmse'].append(train_reg_rmse)

        if wandb_logger is not None:
            norm = evaluation._compute_grad_norm(model.parameters(), norm_type=2)
            wandb_logger.log({
                "epoch": epoch+30,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_pure": train_pure,
                "train_reg_rmse": train_reg_rmse,
                "grad_norm": norm
            })
        
        # Do validation every 5 epochs
        if (epoch+1) % 5 == 0:
            model.eval()

            val_loss = 0.0
            val_acc = 0.0
            val_pure = 0.0
            val_reg_rmse = 0.0
            val_progress = tqdm(val_loader, leave=False, position=0)

            for i, (val_data, val_target, val_hit_times, photon_target, photon_list) in enumerate(val_progress):
                val_data, val_target = val_data.to(device), val_target.to(device) # both [B, 1, 16000]
                # val_class_output, _ = good_conformer(val_, mode='bce') # use trained classification for reg sampling
                val_class_output, val_reg_output = model(val_data, mode='bce')

                if mode == 'mined_bce':
                    ### Ensure class_output matches data shape - hack added for unet since 1000 us is not power of 2
                    if val_class_output.shape[-1] != val_data.shape[-1]:
                        diff = val_data.shape[-1] - val_class_output.shape[-1]
                        if diff > 0:
                            val_class_output = F.pad(val_class_output, (0, diff))
                            val_reg_output = F.pad(val_reg_output, (0, diff))
                        else:
                            val_class_output = val_class_output[..., :val_data.shape[-1]]
                            val_reg_output = val_reg_output[..., :val_data.shape[-1]]
                    ##########################################################################################
                    loss, _, _, _, _, _, _, _ = mined_bce_loss(val_data, val_hit_times, photon_list, val_class_output, val_reg_output, epoch, device)
                    acc = overall_class_acc(val_hit_times, val_class_output, device)
                    pure = overall_class_purity(val_hit_times, val_class_output, device)
                    reg_rmse = regression_rmse(val_hit_times, photon_target, val_reg_output, val_class_output, device, mse=mse)
                elif mode == 'bce':
                    loss, _ = bce_loss(val_data, val_hit_times, val_class_output, device)
                    acc = overall_class_acc(val_hit_times, val_class_output, device)
                    pure = overall_class_purity(val_hit_times, val_class_output, device)
                
                # averaged over the batch already
                val_loss += loss.item() 
                val_acc += acc
                val_pure += pure
                val_reg_rmse += reg_rmse

                val_progress.set_postfix({"val_loss": val_loss/(i+1), "val_acc": val_acc/(i+1)})

            # divide by number of batches seen
            val_loss /= len(val_loader) 
            val_acc /= len(val_loader)
            val_pure /= len(val_loader)
            val_reg_rmse /= len(val_loader)

            results['eval_loss'].append(val_loss)
            results['eval_acc'].append(val_acc)
            results['eval_pure'].append(val_pure)
            results['eval_reg_rmse'].append(val_reg_rmse)

            if wandb_logger is not None:
                wandb_logger.log({
                    "epoch": epoch,
                    "eval_loss": val_loss,
                    "eval_acc": val_acc,
                    "eval_pure": val_pure,
                    "eval_reg_rmse": val_reg_rmse
                })
            scheduler.step(val_loss)

        if (epoch+1) % 5 == 0:
            # save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"/sdf/home/c/carsmith/sdf_data/flash_detection_data/conformer_ckpts/conformer_v5_drop_NEWunmerged/{epoch}.pth")

    return results

def test_model(model, test_loader, criterion, device):
    # 1. calculate average loss over the test set
    # 2. do argmax over test set model class_outputs and calculate average delta from correct start time
    # 3. for a few waveforms return: predicted start time, actual start time, original waveform shape to plot
    model.eval()
    test_loss = 0.0
    avg_delta = 0.0
    test_acc = 0.0
    n_items = 0.0
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc=f"Test Epoch {1}", leave=False, position=1)
        for test_data, test_target, test_hits in test_progress:
            test_data, test_target = test_data.to(device), test_target.to(device) # [B, 1, 16000], [B, 1, 16000]
            test_class_output = model(test_data)

            # Compute BCE loss as a weighted average of loss for positive truth and negative truth time bins
            # find all indices where target equals 1.0
            positive_indices = torch.where(test_target == 1.0)

            # gather the predictions and targets at all positive (true) time bins
            if len(positive_indices[0]) > 0:
                positive_pred = test_class_output[positive_indices]  # shape [num_positives]
                positive_target = test_target[positive_indices]  # shape [num_positives]
            else:
                # if no positive indices found
                positive_pred = torch.empty(0, device=test_class_output.device)
                positive_target = torch.empty(0, device=test_target.device)

            # find negative bins by taking all non-positive indices
            negative_mask = torch.ones_like(test_target, dtype=torch.bool)
            if len(positive_indices[0]) > 0:
                negative_mask[positive_indices] = False  
            negative_pred = test_class_output[negative_mask]  # shape [num_negatives]
            negative_target = test_target[negative_mask]  # should be all zeros

            # Compute BCE loss for positive and negative truth time bins
            if len(positive_pred) > 0:
                positive_loss = criterion(positive_pred, positive_target)
                positive_weight = test_data.shape[0] / (len(positive_pred)) # weight is inversely proportional to number of positive bins
            else:
                positive_loss = torch.tensor(0.0, device=test_class_output.device)
                positive_weight = 0.0
                
            negative_loss = criterion(negative_pred, negative_target)
            negative_weight = test_data.shape[0] / (len(negative_pred)) # weight is inversely proportional to number of negative bins
            
            loss = (positive_loss * positive_weight + negative_loss * negative_weight) #/ (positive_weight + negative_weight)

            test_loss += loss.item()
            n_items += test_data.shape[0]

            # calculating delta t
            # avg_delta += (torch.abs(pred - target))
            # check if n largest probabilities in predicted values are same indices as true flashes
            # get array of the count of 1's per sample in the batch
            # Count the number of 1.0s per sample in the batch in test_target
            B = test_class_output.shape[0]
            test_class_output_flat = test_class_output.view(B, -1)  # Shape: [B, 16000]
            test_target_flat = test_target.view(B, -1)  # Shape: [B, 16000]
            ones_per_sample = (test_target_flat == 1.0).sum(dim=1)  # Shape: [B]

            largest_indices = []

            # Loop over batch — fast because only over B, not 16000
            for b in range(B):
                n = ones_per_sample[b].item()
                if n == 0:
                    largest_indices.append(torch.tensor([], device=test_class_output.device, dtype=torch.long))
                else:
                    topk = torch.topk(torch.abs(test_class_output_flat[b]), k=n)
                    largest_indices.append(topk.indices)

            # calculating accuracy 
            positive_indices = (test_target_flat == 1.0).nonzero(as_tuple=False)
            true_pos_by_batch = defaultdict(set)
            for b, idx in positive_indices:
                true_pos_by_batch[b.item()].add(idx.item())

            for b in range(B):
                test_acc += sum(i.item() in true_pos_by_batch[b] for i in largest_indices[b])
            test_progress.set_postfix({"test_loss": test_loss/n_items, "test_acc": test_acc/n_items})

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        return test_data, test_target, test_loss, test_acc