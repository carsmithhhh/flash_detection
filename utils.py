from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import UNet1D

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, wandb_logger=None):
    results = {}
    results['train_loss'] = []
    results['train_acc'] = []
    results['eval_loss'] = []
    results['eval_acc'] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0)
        n_items = 0.0
        for i, (data, target) in enumerate(train_progress):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # Do weighted BCE loss
            positive_indices = torch.where(target == 1.0)

            if len(positive_indices[0]) > 0:
                positive_pred = output[positive_indices]  # shape [num_positives]
                positive_target = target[positive_indices]  # shape [num_positives]
            else:
                # if no positive indices found
                positive_pred = torch.empty(0, device=output.device)
                positive_target = torch.empty(0, device=target.device)

            # find negative bins by taking all non-positive indices
            negative_mask = torch.ones_like(target, dtype=torch.bool)
            if len(positive_indices[0]) > 0:
                negative_mask[positive_indices] = False  
            negative_pred = output[negative_mask]  # shape [num_negatives]
            negative_target = target[negative_mask]  # should be all zeros

            if len(positive_pred) > 0:
                positive_loss = criterion(positive_pred, positive_target)
                positive_weight = data.shape[0] / (len(positive_pred)) # weight is inversely proportional to number of positive bins
            else:
                positive_loss = torch.tensor(0.0, device=output.device)
                positive_weight = 0.0

            negative_loss = criterion(negative_pred, negative_target)
            negative_weight = data.shape[0] / (len(negative_pred)) # weight is inversely proportional to number of negative bins
            
            loss = (positive_loss * positive_weight + negative_loss * negative_weight) #/ (positive_weight + negative_weight)

            train_loss += loss.item()
            
            # kind of redundant accuracy calculation
            # sees if top x probabilities per sample match flash indices from target
            B = output.shape[0]
            output_flat = output.view(B, -1)  # Shape: [B, 16000]
            target_flat = target.view(B, -1)  # Shape: [B, 16000]
            ones_per_sample = (target_flat == 1.0).sum(dim=1)  # Shape: [B]

            largest_indices = []

            # Loop over batch — fast because only over B, not 16000
            for b in range(B):
                n = ones_per_sample[b].item()
                if n == 0:
                    largest_indices.append(torch.tensor([], device=output.device, dtype=torch.long))
                else:
                    topk = torch.topk(output_flat[b], k=n)
                    largest_indices.append(topk.indices)

            # calculating accuracy 
            positive_indices = (target_flat == 1.0).nonzero(as_tuple=False)
            true_pos_by_batch = defaultdict(set)
            for b, idx in positive_indices:
                true_pos_by_batch[b.item()].add(idx.item())

            for b in range(B):
                train_acc += sum(i.item() in true_pos_by_batch[b] for i in largest_indices[b])

            # set progress bar to show loss and accuracy
            n_items += data.shape[0]
            train_progress.set_postfix({"train_loss": train_loss/n_items, "train_acc": train_acc/n_items})
        print(f"last batch positive weight: {positive_weight}, negative_weight: {negative_weight}")
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        # train_progress.set_postfix({"train_loss": train_loss, "train_acc": train_acc})
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)

        # Log to wandb
        if wandb_logger is not None:
            wandb_logger.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc
            })

        if (epoch + 1) % 5 == 0 and val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            n_val_items = 0.0
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False, position=1)
                for val_data, val_target in val_progress:
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_output = model(val_data)

                    positive_indices = torch.where(val_target == 1.0)

                    # gather the predictions and targets at all positive (true) time bins
                    if len(positive_indices[0]) > 0:
                        positive_pred = val_output[positive_indices]  # shape [num_positives]
                        positive_target = val_target[positive_indices]  # shape [num_positives]
                    else:
                        # if no positive indices found
                        positive_pred = torch.empty(0, device=val_output.device)
                        positive_target = torch.empty(0, device=val_target.device)

                    # find negative bins by taking all non-positive indices
                    negative_mask = torch.ones_like(val_target, dtype=torch.bool)
                    if len(positive_indices[0]) > 0:
                        negative_mask[positive_indices] = False  
                    negative_pred = val_output[negative_mask]  # shape [num_negatives]
                    negative_target = val_target[negative_mask]  # should be all zeros

                    # Compute BCE loss for positive and negative truth time bins
                    if len(positive_pred) > 0:
                        positive_loss = criterion(positive_pred, positive_target)
                        positive_weight = val_data.shape[0] / (len(positive_pred)) # weight is inversely proportional to number of positive bins
                    else:
                        positive_loss = torch.tensor(0.0, device=val_output.device)
                        positive_weight = 0.0
                        
                    negative_loss = criterion(negative_pred, negative_target)
                    negative_weight = val_data.shape[0] / (len(negative_pred)) # weight is inversely proportional to number of negative bins
                    
                    loss = (positive_loss * positive_weight + negative_loss * negative_weight)

                    n_val_items += val_data.shape[0]
                    val_loss += loss.item()
                    train_acc += (positive_pred > 0.95).all().item()
                    val_progress.set_postfix({"val_loss": val_loss/n_val_items, "val_acc": val_acc/n_val_items})

                val_loss /= len(val_loader.dataset)
                val_acc /= len(val_loader.dataset)
                results['eval_loss'].append(val_loss)
                results['eval_acc'].append(val_acc)

                # Log validation metrics
                if wandb_logger is not None:
                    wandb_logger.log({
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })

        model.train()
    return results

def test_model(model, test_loader, criterion, device):
    # 1. calculate average loss over the test set
    # 2. do argmax over test set model outputs and calculate average delta from correct start time
    # 3. for a few waveforms return: predicted start time, actual start time, original waveform shape to plot
    model.eval()
    test_loss = 0.0
    avg_delta = 0.0
    test_acc = 0.0
    n_items = 0.0
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc=f"Test Epoch {1}", leave=False, position=1)
        for test_data, test_target in test_progress:
            test_data, test_target = test_data.to(device), test_target.to(device) # [B, 1, 16000], [B, 1, 16000]
            test_output = model(test_data)

            # Compute BCE loss as a weighted average of loss for positive truth and negative truth time bins
            # find all indices where target equals 1.0
            positive_indices = torch.where(test_target == 1.0)

            # gather the predictions and targets at all positive (true) time bins
            if len(positive_indices[0]) > 0:
                positive_pred = test_output[positive_indices]  # shape [num_positives]
                positive_target = test_target[positive_indices]  # shape [num_positives]
            else:
                # if no positive indices found
                positive_pred = torch.empty(0, device=test_output.device)
                positive_target = torch.empty(0, device=test_target.device)

            # find negative bins by taking all non-positive indices
            negative_mask = torch.ones_like(test_target, dtype=torch.bool)
            if len(positive_indices[0]) > 0:
                negative_mask[positive_indices] = False  
            negative_pred = test_output[negative_mask]  # shape [num_negatives]
            negative_target = test_target[negative_mask]  # should be all zeros

            # Compute BCE loss for positive and negative truth time bins
            if len(positive_pred) > 0:
                positive_loss = criterion(positive_pred, positive_target)
                positive_weight = test_data.shape[0] / (len(positive_pred)) # weight is inversely proportional to number of positive bins
            else:
                positive_loss = torch.tensor(0.0, device=test_output.device)
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
            B = test_output.shape[0]
            test_output_flat = test_output.view(B, -1)  # Shape: [B, 16000]
            test_target_flat = test_target.view(B, -1)  # Shape: [B, 16000]
            ones_per_sample = (test_target_flat == 1.0).sum(dim=1)  # Shape: [B]

            largest_indices = []

            # Loop over batch — fast because only over B, not 16000
            for b in range(B):
                n = ones_per_sample[b].item()
                if n == 0:
                    largest_indices.append(torch.tensor([], device=test_output.device, dtype=torch.long))
                else:
                    topk = torch.topk(test_output_flat[b], k=n)
                    largest_indices.append(topk.indices)

            # calculating accuracy 
            positive_indices = (test_target_flat == 1.0).nonzero(as_tuple=False)
            true_pos_by_batch = defaultdict(set)
            for b, idx in positive_indices:
                true_pos_by_batch[b.item()].add(idx.item())

            for b in range(B):
                test_acc += sum(i.item() in true_pos_by_batch[b] for i in largest_indices[b])
            test_progress.set_postfix({"test_loss": test_loss/n_items, "test_acc": test_acc/n_items})

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)

        return test_data, test_target, test_loss, test_acc

    

    
