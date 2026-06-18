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

def train_conformer_w_dist(model, teacher, train_loader, val_loader, scheduler, optimizer, device, epochs, wandb_logger=None):
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
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)
            class_output, reg_output = model(data)
            