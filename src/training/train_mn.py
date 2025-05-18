import os
import logging
import random
import gc
import time
import warnings
import sys
import contextlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import wandb
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, multilabel_confusion_matrix
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import math

from config import config
from src.models.efficient_at.mn.model import get_model as get_efficient_at_model
from src.training.losses import FocalLossBCE
from src.datasets.birdclef_dataset import BirdCLEFDataset, _load_adain_per_freq_stats, _apply_adain_transformation
# Attempt to import the user's EfficientNet model
# The user needs to ensure this file and class exist.
# Example: src/models/birdclef_model.py contains class BirdCLEFModel
try:
    from src.models.en_model import get_en_model as get_efficientnet_model
except ImportError:
    BirdCLEFModel = None # Will be checked before use
    print("INFO: BirdCLEFModel not found or could not be imported. EfficientNet architecture will not be available unless fixed.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description="BirdCLEF Training Script")
parser.add_argument("--run_name", type=str, default=None, help="Custom name for the W&B run.")
cmd_args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    """Custom collate function."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("Warning: collate_fn received empty batch, returning None.")
        return None

    # Check for essential keys first
    # 'sample_weight' is optional and only present if distance weighting is on and mode is train
    essential_keys = {'melspec', 'target', 'filename', 'samplename', 'source'}
    if not all(essential_keys.issubset(item.keys()) for item in batch):
         print("Warning: Batch items missing one or more essential keys. Returning None.")
         # Optionally, print which keys are missing from which item for detailed debugging
         # for i, item in enumerate(batch):
         #    if not essential_keys.issubset(item.keys()):
         #        print(f"Item {i} missing keys: {essential_keys - item.keys()}")
         return None

    result = {key: [] for key in batch[0].keys()} # Initialize with all keys from the first item
    has_sample_weight = 'sample_weight' in batch[0] # Check if first item has it

    for item in batch:
        for key, value in item.items():
            result[key].append(value)
        # If an item is missing sample_weight (e.g. validation batch), ensure it's handled if key was added from first item
        if has_sample_weight and 'sample_weight' not in item:
            # This case should ideally not happen if only train batches have weights and val don't, 
            # and collate_fn processes batches of same type. 
            # If it can happen, assign a default weight or handle as error.
            print("Warning: Inconsistent 'sample_weight' in batch. Appending 1.0.")
            result['sample_weight'].append(torch.tensor(1.0, dtype=torch.float32)) 

    try:
        result['target'] = torch.stack(result['target'])
        result['melspec'] = torch.stack(result['melspec'])
        if has_sample_weight and result['sample_weight']: # Ensure list is not empty
            result['sample_weight'] = torch.stack(result['sample_weight'])
        elif 'sample_weight' in result and not result['sample_weight']: # key exists but list is empty
            del result['sample_weight'] # remove if it ended up empty

    except RuntimeError as e:
        print(f"Error stacking tensors in collate_fn: {e}. Returning None.")
        return None
    except Exception as e:
        print(f"Unexpected error in collate_fn: {e}. Returning None.")
        return None

    # Final check for consistent batch dimensions
    if result['melspec'].shape[0] != len(batch) or result['target'].shape[0] != len(batch):
        print("Warning: Collated melspec/target tensors have incorrect batch dimension. Returning None.")
        return None
    if has_sample_weight and 'sample_weight' in result and result['sample_weight'].shape[0] != len(batch):
        print("Warning: Collated sample_weight tensor has incorrect batch dimension. Returning None.")
        return None

    return result

def mixup_data(x, targets, sources, alpha, device):
    """Applies mixup augmentation, ensuring mixing only occurs between samples from the same source.
    Returns mixed inputs, targets_a, targets_b, and lambda.
    'sources' is a list of strings indicating the data source for each sample in the batch.
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # Create new_indices for permutation, initialized to self-loops
    new_indices = list(range(batch_size))
    unique_sources_in_batch = set(sources)

    for source_type in unique_sources_in_batch:
        # Get original indices of samples belonging to the current source_type
        indices_of_this_source = [i for i, src in enumerate(sources) if src == source_type]
        
        if len(indices_of_this_source) > 1:
            # Shuffle these specific indices to find mixup partners from the same source
            shuffled_partners_for_this_source = random.sample(indices_of_this_source, len(indices_of_this_source))
            for i, original_idx in enumerate(indices_of_this_source):
                new_indices[original_idx] = shuffled_partners_for_this_source[i]
        # If len(indices_of_this_source) <= 1, new_indices[original_idx] remains original_idx (mix with self)

    permuted_indices = torch.tensor(new_indices, device=device)

    mixed_x = lam * x + (1 - lam) * x[permuted_indices]
    targets_a, targets_b = targets, targets[permuted_indices]
    return mixed_x, targets_a, targets_b, lam

def rand_bbox(size, lam):
    """Generates random bounding box coordinates based on lambda."""
    W = size[2] # Width (time dimension)
    H = size[3] # Height (frequency dimension)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate box coordinates, clamping to image boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, targets, sources, alpha, device):
    """Applies CutMix augmentation, ensuring mixing only occurs between samples from the same source.
    Returns mixed inputs, targets_a, targets_b, and lambda.
    'sources' is a list of strings indicating the data source for each sample in the batch.
    """
    batch_size = x.size(0)

    # Create new_indices for permutation, initialized to self-loops
    new_indices = list(range(batch_size))
    unique_sources_in_batch = set(sources)

    for source_type in unique_sources_in_batch:
        indices_of_this_source = [i for i, src in enumerate(sources) if src == source_type]
        if len(indices_of_this_source) > 1:
            shuffled_partners_for_this_source = random.sample(indices_of_this_source, len(indices_of_this_source))
            for i, original_idx in enumerate(indices_of_this_source):
                new_indices[original_idx] = shuffled_partners_for_this_source[i]
    
    permuted_indices = torch.tensor(new_indices, device=device)
    targets_a, targets_b = targets, targets[permuted_indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    # Apply CutMix using the permuted indices from the same source
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[permuted_indices, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, targets_a, targets_b, lam

def get_optimizer(model, config):
    """Creates optimizer based on config settings."""
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'SGD':
         optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer '{config.optimizer}' not implemented")
    return optimizer

def get_scheduler(optimizer, config):
    """Creates learning rate scheduler based on config settings."""
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.min_lr
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', # Usually val_loss
            factor=0.5,
            patience=2,
            min_lr=config.min_lr,
            verbose=True
        )
    elif config.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.epochs // 3),
            gamma=0.5
        )
    elif config.scheduler is None or config.scheduler.lower() == 'none':
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler '{config.scheduler}' not implemented")
    return scheduler

def get_criterion(config):
    """Creates loss criterion based on config settings."""
    reduction_type = 'mean'
    if hasattr(config, 'ENABLE_DISTANCE_WEIGHTING') and config.ENABLE_DISTANCE_WEIGHTING:
        print("INFO: Distance weighting enabled, using reduction='none' for loss criterion.")
        reduction_type = 'none'
    
    if config.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction=reduction_type)
    elif config.criterion == 'FocalLossBCE':
        print(f"INFO: Using FocalLossBCE with reduction='{reduction_type}'.")
        criterion = FocalLossBCE(config=config, reduction=reduction_type)
    else:
        raise NotImplementedError(f"Criterion '{config.criterion}' not implemented")
    return criterion

def calculate_auc(targets, outputs, sample_weights=None):
    """Calculates macro-averaged ROC AUC.
    Can apply sample weights to the AUC calculation for each class if provided.
    """
    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0: # Check if class has positive samples
            try:
                # Use sample_weight if provided
                current_class_weights = None
                if sample_weights is not None:
                    # Ensure sample_weights is 1D and matches the number of samples for this class
                    if sample_weights.ndim == 1 and sample_weights.shape[0] == targets.shape[0]:
                        current_class_weights = sample_weights
                    else:
                        print(f"Warning: calculate_auc received sample_weights with unexpected shape ({sample_weights.shape}). Ignoring for class {i}.")
                
                class_auc = roc_auc_score(targets[:, i], probs[:, i], sample_weight=current_class_weights)
                aucs.append(class_auc)
            except ValueError as e:
                # This can happen if only one class is present in targets[:, i] after filtering by weights (if weights are zero for one class)
                # Or if all targets for a class are the same.
                # print(f"ValueError calculating AUC for class {i}: {e}. Skipping class.")
                pass # Silently skip or log as needed
            except Exception as e_calc_auc: # Catch other errors
                print(f"Error calculating AUC for class {i}: {e_calc_auc}. Skipping class.")
                pass

    return np.mean(aucs) if aucs else 0.0

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, 
                      logged_original_samplenames_for_spectrograms, num_samples_to_log,
                      scheduler=None):
    """Runs one epoch of training with optional mixed precision."""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    use_amp = scaler.is_enabled()
    # Check if batch augmentation is globally enabled
    batch_augment_active = config.batch_augment_prob > 0
    distance_weighting_enabled = hasattr(config, 'ENABLE_DISTANCE_WEIGHTING') and config.ENABLE_DISTANCE_WEIGHTING

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if batch is None:
            print(f"Warning: Skipping None batch at step {step}")
            continue

        try:
            inputs_orig = batch['melspec'].to(device)
            targets_orig = batch['target'].to(device)
            sources_orig = batch['source'] # List of strings, no .to(device)
            sample_weights_batch = None
            if distance_weighting_enabled and 'sample_weight' in batch:
                sample_weights_batch = batch['sample_weight'].to(device)
            elif distance_weighting_enabled:
                # This case should ideally not occur if dataset returns weights for train and collate handles it.
                # If it does, it means sample_weight was expected but not found in this training batch.
                print("Warning: Distance weighting enabled, but 'sample_weight' not found in training batch. Using weight 1.0 for all samples in batch.")
                sample_weights_batch = torch.ones(inputs_orig.size(0), device=device, dtype=torch.float32)

        except (AttributeError, TypeError) as e:
            print(f"Error: Skipping batch {step} due to unexpected format: {e}")
            continue

        optimizer.zero_grad()

        # --- Decide whether to apply batch augmentation --- #
        apply_augmentation_this_batch = (batch_augment_active and 
                                       random.random() < config.batch_augment_prob)
        
        apply_mixup = False
        apply_cutmix = False # Ensure apply_cutmix is always initialized

        if apply_augmentation_this_batch:
            # --- Decide which augmentation: Mixup or CutMix --- #
            use_mixup_decision = random.random() < config.mixup_vs_cutmix_ratio

            if use_mixup_decision:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs_orig, targets_orig, sources_orig, config.mixup_alpha, device
                )
                apply_mixup = True
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs_orig, targets_orig, sources_orig, config.cutmix_alpha, device
                )
                apply_cutmix = True
        else:
            # No augmentation for this batch
            inputs = inputs_orig
            targets_a = targets_orig
            lam = 1.0 # Ensure lambda is 1 when no augmentation

        # --- Forward pass and Loss Calculation --- #
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            model_output = model(inputs)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output # Get logits

            if apply_mixup or apply_cutmix:
                raw_loss_a = criterion(logits, targets_a) # Per-sample loss
                raw_loss_b = criterion(logits, targets_b) # Per-sample loss
                if distance_weighting_enabled and sample_weights_batch is not None:
                    # Ensure weights are correctly broadcastable: (batch_size, 1) for (batch_size, num_classes) loss
                    weights_expanded = sample_weights_batch.unsqueeze(1) if raw_loss_a.ndim > sample_weights_batch.ndim else sample_weights_batch
                    weighted_loss_a = raw_loss_a * weights_expanded
                    weighted_loss_b = raw_loss_b * weights_expanded 
                    loss = (lam * weighted_loss_a.mean()) + ((1 - lam) * weighted_loss_b.mean())
                else:
                    loss = lam * raw_loss_a.mean() + (1 - lam) * raw_loss_b.mean()
            else:
                raw_loss = criterion(logits, targets_a) # targets_a is targets_orig here. This is per-sample loss.
                if distance_weighting_enabled and sample_weights_batch is not None:
                    weights_expanded = sample_weights_batch.unsqueeze(1) if raw_loss.ndim > sample_weights_batch.ndim else sample_weights_batch
                    weighted_loss = raw_loss * weights_expanded
                    loss = weighted_loss.mean()
                else:
                    loss = raw_loss.mean() # Default behavior if not weighting or weights missing

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Use original targets for AUC calculation, applying ceil for label smoothing
        outputs_np = logits.detach().float().cpu().numpy()
        targets_np = torch.ceil(targets_orig).detach().cpu().numpy() 

        all_outputs.append(outputs_np)
        all_targets.append(targets_np)
        losses.append(loss.item())

        # --- Log Spectrograms ---
        if wandb.run is not None and len(logged_original_samplenames_for_spectrograms) < num_samples_to_log:
            with torch.no_grad(): # Ensure no gradients are calculated for this section
                for j in range(inputs.size(0)): # Iterate through batch
                    if len(logged_original_samplenames_for_spectrograms) >= num_samples_to_log:
                        break # Stop if we've logged enough globally

                    original_samplename = batch['samplename'][j] # Assuming 'samplename' is in batch output from collate_fn
                    
                    if original_samplename not in logged_original_samplenames_for_spectrograms:
                        try:
                            # inputs[j] is (3, H, W), normalized. Plot first channel.
                            # We need to denormalize or be aware colors might be off, or just plot as is.
                            # For simplicity, plot as is. It shows what model gets before backbone.
                            img_tensor_first_channel = inputs[j, 0, :, :].cpu()
                            
                            # If you want to try to roughly denormalize for viewing (optional, can be complex):
                            # mean = torch.tensor([0.485]).view(1, 1).to(img_tensor_first_channel.device)
                            # std = torch.tensor([0.229]).view(1, 1).to(img_tensor_first_channel.device)
                            # img_to_plot_denorm = img_tensor_first_channel * std + mean
                            # img_to_plot_np = torch.clamp(img_to_plot_denorm, 0, 1).numpy()
                            img_to_plot_np = img_tensor_first_channel.numpy()

                            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                            ax.imshow(img_to_plot_np, aspect='auto', origin='lower', cmap='viridis')
                            
                            caption = f"Sample: {original_samplename}"
                            if apply_mixup or apply_cutmix:
                                if j < len(targets_a) and hasattr(targets_a[j], 'cpu'): # Check if valid tensor for original ID
                                    # To get the ID of the mixed sample, we need to trace back through `indices`
                                    # original_samplename_b might not be straightforward if `indices` is not available here
                                    # For simplicity, just indicate it's mixed.
                                    caption += f" (Mixed, lam: {lam:.2f} approx for batch)"
                                else: # Fallback if targets_a is not as expected
                                    caption += f" (Mixed)"
                            caption += " \n(1st chan, post-all-augs & norm)"
                            
                            ax.set_title(caption, fontsize=8)
                            ax.axis('off')
                            plt.tight_layout()

                            wandb.log({
                                f"training_augmented_spectrograms/sample_{original_samplename}": wandb.Image(fig)
                            }, commit=True) # Commit immediately for simplicity
                            
                            plt.close(fig)
                            logged_original_samplenames_for_spectrograms.append(original_samplename)
                        except Exception as e_log_spec:
                            print(f"Warning (W&B Log Spec in train_one_epoch): Failed for {original_samplename}: {e_log_spec}")
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

        if config.debug and (step + 1) >= config.debug_limit_batches:
            print(f"DEBUG: Stopping training epoch early after {config.debug_limit_batches} batches.")
            break

    if not all_targets or not all_outputs:
        print("Warning: No targets or outputs collected during training epoch.")
        return 0.0, 0.0

    all_outputs_cat = np.concatenate(all_outputs)
    all_targets_cat = np.concatenate(all_targets)
    auc = calculate_auc(all_targets_cat, all_outputs_cat)
    avg_loss = np.mean(losses)

    return avg_loss, auc

def validate(model, loader, criterion, device):
    """Runs validation with optional mixed precision for inference."""
    model.eval()
    losses = []
    all_targets_list = [] 
    all_outputs_list = [] 
    all_sample_weights_list = [] # Initialize list to store sample weights per batch for validation
    use_amp = config.use_amp
    species_ids = loader.dataset.species_ids 
    num_classes = len(species_ids)
    # Access the reduction type from the criterion object itself
    criterion_reduction_type = criterion.reduction if hasattr(criterion, 'reduction') else 'mean'
    # Determine if distance weighting is active for this validation run
    distance_weighting_enabled_for_val = hasattr(config, 'ENABLE_DISTANCE_WEIGHTING') and config.ENABLE_DISTANCE_WEIGHTING

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Validation")):
            #print(f"DEBUG: Validation step {step}, processing batch...") # DEBUG PRINT
            if batch is None:
                print(f"Warning: Skipping None validation batch at step {step}")
                continue

            try:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                sample_weights_batch_val = None
                if distance_weighting_enabled_for_val and 'sample_weight' in batch:
                    sample_weights_batch_val = batch['sample_weight'].to(device)
                    if sample_weights_batch_val is not None: # Ensure it's not None before trying to append
                        all_sample_weights_list.append(sample_weights_batch_val.cpu().numpy())
                elif distance_weighting_enabled_for_val:
                    # This might occur if collate_fn or dataset had an issue, or if weighting is on but a specific batch misses weights
                    print(f"Warning: Distance weighting for validation enabled, but 'sample_weight' not found in batch {step}. Using weight 1.0 for this batch.")
                    sample_weights_batch_val = torch.ones(inputs.size(0), device=device, dtype=torch.float32)
                #if 'filename' not in batch or 'samplename' not in batch:
                    #print(f"DEBUG: Validation step {step}, batch loaded. No filename/samplename key in batch.")

            except (AttributeError, TypeError) as e:
                print(f"Error: Skipping validation batch {step} due to unexpected format: {e}")
                continue

            #print(f"DEBUG: Validation step {step}, about to run model forward pass.") # DEBUG PRINT
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                model_output = model(inputs)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output # Get logits
                loss_val = criterion(outputs, targets) # This might be a tensor if reduction is 'none'

            # If criterion returned per-element losses (because reduction was 'none'),
            # take the mean for logging and for scheduler if ReduceLROnPlateau is used with val_loss.
            batch_mean_loss = 0.0 # Initialize
            if criterion_reduction_type == 'none' and loss_val.ndim > 0:
                if distance_weighting_enabled_for_val and sample_weights_batch_val is not None:
                    # Apply sample weights to per-sample losses
                    # Ensure weights are correctly broadcastable: (batch_size, 1) for (batch_size, num_classes) loss
                    weights_expanded_val = sample_weights_batch_val.unsqueeze(1) if loss_val.ndim > sample_weights_batch_val.ndim else sample_weights_batch_val
                    weighted_loss_val = loss_val * weights_expanded_val
                    batch_mean_loss = weighted_loss_val.mean()
                else:
                    # If not weighting or weights are missing for some reason, take unweighted mean
                    batch_mean_loss = loss_val.mean()
            else: # criterion reduction is 'mean' or loss_val is already scalar
                batch_mean_loss = loss_val # It's already a scalar or reduction was 'mean'

            all_outputs_list.append(outputs.float().cpu().numpy())
            all_targets_list.append(torch.ceil(targets).cpu().numpy()) #use ceil to convert to 0/1 targets for label smoothing
            losses.append(batch_mean_loss.item())

            if config.debug and (step + 1) >= config.debug_limit_batches:
                print(f"DEBUG: Stopping validation early after {config.debug_limit_batches} batches.")
                break

    if not all_targets_list or not all_outputs_list:
        print("Warning: No targets or outputs collected during validation.")
        # Return empty structures or zeros for all expected values
        per_class_metrics = [{'species_id': sid, 'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for sid in species_ids]
        return 0.0, 0.0, per_class_metrics, np.array([]), np.array([])

    all_outputs_cat = np.concatenate(all_outputs_list)
    all_targets_cat = np.concatenate(all_targets_list)
    
    sample_weights_for_metrics = None
    if distance_weighting_enabled_for_val and all_sample_weights_list:
        # Check if any actual weights were collected (i.e., not all batches were missing them)
        # Filter out None entries if any were appended due to missing weights in some batches (though current logic appends ones)
        valid_weights_collected = [w for w in all_sample_weights_list if w is not None]
        if valid_weights_collected:
            try:
                sample_weights_for_metrics = np.concatenate(valid_weights_collected)
                if sample_weights_for_metrics.shape[0] != all_targets_cat.shape[0]:
                    print(f"WARNING: Concatenated sample_weights_for_metrics shape ({sample_weights_for_metrics.shape[0]}) mismatch with targets ({all_targets_cat.shape[0]}). Disabling for metrics.")
                    sample_weights_for_metrics = None
                else:
                    print(f"INFO: Using {sample_weights_for_metrics.sum():.2f} total weight for {len(sample_weights_for_metrics)} samples in validation metric calculation.")
            except ValueError as e_concat_weights: # Handle potential error during concatenation (e.g. empty list if all batches missed weights)
                print(f"WARNING: Error concatenating sample weights for metrics: {e_concat_weights}. Proceeding unweighted.")
                sample_weights_for_metrics = None
        else:
            print("INFO: Distance weighting for validation enabled, but no valid sample weights were collected from batches. Proceeding unweighted for metrics.")

    # macro_auc = calculate_auc(all_targets_cat, all_outputs_cat) # Existing macro AUC calculation
    # Modified calculate_auc to accept sample_weights
    macro_auc = calculate_auc(all_targets_cat, all_outputs_cat, 
                              sample_weights_for_metrics if distance_weighting_enabled_for_val and sample_weights_for_metrics is not None else None)
    avg_loss = np.mean(losses) if losses else 0.0

    # --- Per-Class Metrics Calculation ---
    per_class_metrics_list = []
    probabilities_cat = 1 / (1 + np.exp(-all_outputs_cat)) # Sigmoid probabilities
    predictions_binary_cat = (probabilities_cat >= 0.5).astype(int)

    # Get TP, FP, FN, TN for all classes using multilabel_confusion_matrix
    try:
        # multilabel_confusion_matrix does not accept sample_weight directly for TP/FP/FN counts in older sklearn.
        # For weighted TP/FP/FN/TN, one might need to calculate them manually per class or use a library that supports it.
        # For now, TP/FP/FN/TN will remain unweighted.
        mcm = multilabel_confusion_matrix(all_targets_cat, predictions_binary_cat, labels=list(range(num_classes)))
    except Exception as e_mcm:
        print(f"Error calculating multilabel_confusion_matrix: {e_mcm}. Per-class TP/FP/FN/TN will be zero.")
        mcm = np.zeros((num_classes, 2, 2), dtype=int) # Fallback

    for i in range(num_classes):
        class_targets = all_targets_cat[:, i]
        class_probs = probabilities_cat[:, i]
        class_preds_binary = predictions_binary_cat[:, i]
        species_name = species_ids[i]

        class_auc = 0.0
        if np.sum(class_targets) > 0 and np.sum(1 - class_targets) > 0: # Ensure both classes are present for AUC
            try:
                # Pass sample weights to roc_auc_score if available and enabled
                current_sample_weights = None
                if distance_weighting_enabled_for_val and sample_weights_for_metrics is not None:
                    # Assuming sample_weights_for_metrics is a 1D array aligned with all_targets_cat
                     if sample_weights_for_metrics.shape[0] == class_targets.shape[0]:
                        current_sample_weights = sample_weights_for_metrics
                     else:
                        # This case implies an issue with collecting/aligning weights, log and proceed without.
                        print(f"WARNING: Sample weights for class {species_name} AUC calculation have incorrect shape. Proceeding unweighted.")

                class_auc = roc_auc_score(class_targets, class_probs, sample_weight=current_sample_weights)
            except ValueError:
                class_auc = 0.0 # Or handle as NaN, but 0.0 is simpler for aggregation
            except Exception as e_auc: # Catch other potential errors like shape mismatches if not caught by check
                print(f"Error calculating weighted AUC for class {species_name}: {e_auc}. Proceeding unweighted.")
                class_auc = roc_auc_score(class_targets, class_probs) # Fallback to unweighted

        # If class_targets are all 0, precision/recall/f1 for label 1 will be 0 due to zero_division=0
        # Pass sample weights to precision_recall_fscore_support if available and enabled
        current_sample_weights_prfs = None
        if distance_weighting_enabled_for_val and sample_weights_for_metrics is not None:
            if sample_weights_for_metrics.shape[0] == class_targets.shape[0]:
                current_sample_weights_prfs = sample_weights_for_metrics
            else:
                print(f"WARNING: Sample weights for class {species_name} PRFS calculation have incorrect shape. Proceeding unweighted.")

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            class_targets, class_preds_binary, average='binary', pos_label=1, zero_division=0,
            sample_weight=current_sample_weights_prfs
        )

        # mcm[i] is [[TN, FP], [FN, TP]] for class i
        tn, fp, fn, tp = mcm[i].ravel()

        per_class_metrics_list.append({
            'species_id': species_name,
            'auc': class_auc,
            'precision': precision, 
            'recall': recall,
            'f1': f1_score,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })

    return avg_loss, macro_auc, per_class_metrics_list, all_outputs_cat, all_targets_cat

def run_training(df, config, trial=None, all_spectrograms=None, 
                 val_spectrogram_data=None, # New parameter
                 custom_run_name=None, hpo_step_offset=0):
    """Runs the training loop. 
    
    Accepts pre-loaded spectrograms via the all_spectrograms argument.
    Accepts dedicated pre-loaded validation spectrograms via val_spectrogram_data.

    If trial is provided (from Optuna), runs only the single fold specified 
    in config.selected_folds, enables pruning, and returns the best validation 
    AUC for that fold.
    
    If trial is None, runs the folds specified in config.selected_folds 
    (can be multiple) without pruning and returns the mean OOF AUC.
    """
    is_hpo_trial = trial is not None
    
    # --- wandb initialization ---
    wandb_run = None # Initialize to None
    if not is_hpo_trial: # Only init if not an HPO trial
        # Determine run_name only for non-HPO runs that will use W&B
        actual_run_name = custom_run_name if custom_run_name else f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project="BirdCLEF-2025", 
            config=config.get_wandb_config(), 
            name=actual_run_name,
            group="full_training", # Always "full_training" for non-HPO
            job_type="train",
            reinit=True 
        )
    # No wandb.config.update for HPO trials as W&B is not initialized for them.

    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")
    print(f"Using Seed: {config.seed}")
    print(f"Load Preprocessed Data: {config.LOAD_PREPROCESSED_DATA}")
    print(f"run_training received {len(all_spectrograms)} pre-loaded samples.")

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    working_df = df.copy()

    # --- Filter pseudo-labels based on usage threshold --- 
    if config.USE_PSEUDO_LABELS and 'data_source' in working_df.columns and 'confidence' in working_df.columns:
        usage_threshold = config.pseudo_label_usage_threshold
        initial_count = len(working_df)
        pseudo_count_before = len(working_df[working_df['data_source'] == 'pseudo'])
        
        # Identify pseudo rows below threshold
        rows_to_drop = working_df[
            (working_df['data_source'] == 'pseudo') &
            (working_df['confidence'] < usage_threshold)
        ].index
        
        if not rows_to_drop.empty:
            working_df = working_df.drop(rows_to_drop).reset_index(drop=True)
            pseudo_count_after = len(working_df[working_df['data_source'] == 'pseudo'])
            print(f"Applied pseudo-label usage threshold ({usage_threshold}):")
            print(f"Removed {len(rows_to_drop)} pseudo labels (Confidence < {usage_threshold}).")
            print(f"Pseudo count: {pseudo_count_before} -> {pseudo_count_after}")
            print(f"Total training df size: {initial_count} -> {len(working_df)}")
        else:
            print(f"No pseudo-labels found below usage threshold {usage_threshold}. All {pseudo_count_before} pseudo labels kept.")
    elif config.USE_PSEUDO_LABELS:
         print("Warning: Could not filter pseudo-labels by confidence. 'data_source' or 'confidence' column missing.")
    # --- End filtering --- 

    if config.USE_RARE_DATA:
        print("\n--- Loading Rare Species Data (USE_RARE_DATA=True) ---")
        try:
            rare_train_df_full = pd.read_csv(config.train_rare_csv_path)
            rare_train_df_full['filepath'] = rare_train_df_full['filename'].apply(lambda f: os.path.join(config.train_audio_rare_dir, f))
            rare_train_df_full['samplename'] = rare_train_df_full.filename.map(
                lambda x: str(x.split('/')[0]) + '-' + str(x.split('/')[-1].split('.')[0])
            )
            rare_train_df_full['data_source'] = 'rare' # Add data source identifier
            # Ensure secondary_labels exists, even if empty, for consistent concatenation
            if 'secondary_labels' not in rare_train_df_full.columns:
                    rare_train_df_full['secondary_labels'] = [[] for _ in range(len(rare_train_df_full))]

            required_cols_rare = ['samplename', 'primary_label', 'secondary_labels', 'filepath', 'filename', 'data_source']
            rare_train_df = rare_train_df_full[required_cols_rare].copy()
            print(f"Loaded and selected columns for {len(rare_train_df)} rare training samples.")
            del rare_train_df_full; gc.collect()
            
            # Concatenate with main training data
            working_df = pd.concat([working_df, rare_train_df], ignore_index=True)
            print(f"Combined DataFrame size (main + rare): {len(working_df)} samples.")

        except Exception as e:
            print(f"CRITICAL ERROR loading or processing rare labels CSV {config.train_rare_csv_path}: {e}")
            sys.exit(1)
    else:
        print("\nSkipping rare-species data (USE_RARE_DATA=False).")

    # Final check on combined dataframe and spectrograms
    print(f"\nFinal training dataframe size: {len(working_df)} samples.")
    
    # --- Filter training_df based on loaded spectrogram keys if configured to load them --- #
    if config.LOAD_PREPROCESSED_DATA:
        print(f"\nConfigured to load preprocessed data. Filtering dataframe...")
        print(f"Total pre-loaded spectrogram keys available: {len(all_spectrograms)}")
        
        original_count = len(working_df)

        loaded_keys = set(all_spectrograms.keys())
        working_df = working_df[working_df['samplename'].isin(loaded_keys)].reset_index(drop=True)
        filtered_count = len(working_df)
        
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"  WARNING: Removed {removed_count} samples from training_df because their spectrograms were not found in the loaded NPZ file(s).")
        
        print(f"  Final training_df size after filtering: {filtered_count} samples.")
    else:
        print("\nWarning: all_spectrograms is None. Cannot filter training_df by loaded keys (which is expected if not loading preprocessed data).")

    # --- BEGIN: Global Oversampling of Rare Classes (Before K-Fold Split) ---
    if config.USE_GLOBAL_OVERSAMPLING:
        min_samples_per_class_global = config.GLOBAL_OVERSAMPLING_MIN_SAMPLES
        print(f"\nApplying global oversampling for classes with < {min_samples_per_class_global} samples...")
        
        class_counts_before_oversampling = working_df['primary_label'].value_counts()
        labels_to_oversample = class_counts_before_oversampling[class_counts_before_oversampling < min_samples_per_class_global].index
        
        oversampled_dfs = [working_df] # Start with the original df
        
        if not labels_to_oversample.empty:
            print(f"  Found {len(labels_to_oversample)} classes to oversample: {list(labels_to_oversample)}")
            for label in tqdm(labels_to_oversample, desc="Global Oversampling"):
                class_df = working_df[working_df['primary_label'] == label]
                current_class_count = len(class_df) # This should be from original df, not accumulating
                
                # Re-fetch class_df from the original (non-concatenated) working_df to get correct current_class_count
                # This is important if a class is oversampled in multiple iterations (though less likely with simple threshold)
                original_class_df = oversampled_dfs[0][oversampled_dfs[0]['primary_label'] == label] # oversampled_dfs[0] is the original working_df
                current_class_count = len(original_class_df)

                num_to_add = min_samples_per_class_global - current_class_count
                
                if num_to_add > 0 and not original_class_df.empty:
                    # Simple replication: sample with replacement from the existing samples of this class
                    replicated_samples = original_class_df.sample(n=num_to_add, replace=True, random_state=config.seed)
                    oversampled_dfs.append(replicated_samples)
            
            if len(oversampled_dfs) > 1: # Only concat if new samples were actually added
                working_df = pd.concat(oversampled_dfs, ignore_index=True)
                print(f"  Finished global oversampling. New working_df size: {len(working_df)}")
            
        else:
            print("  No classes found below the global oversampling threshold.")
    else:
        print("\nGlobal oversampling disabled or not configured (USE_GLOBAL_OVERSAMPLING).")
    # --- END: Global Oversampling of Rare Classes ---

    # --- Create groups for StratifiedGroupKFold with conditional ungrouping --- 
    GROUP_DIVERSITY_THRESHOLD = 30 # Classes with < this many unique groups will be ungrouped

    print("\nCreating groups for StratifiedGroupKFold with conditional ungrouping...")
    
    # 1. Create initial group_id based on lat/lon/author
    working_df['initial_group_id'] = None
    mask_complete_info = working_df['latitude'].notna() & \
                            working_df['longitude'].notna() & \
                            working_df['author'].notna()

    working_df.loc[mask_complete_info, 'initial_group_id'] = (
        working_df.loc[mask_complete_info, 'latitude'].astype(str) + '_' + 
        working_df.loc[mask_complete_info, 'longitude'].astype(str) + '_' + 
        working_df.loc[mask_complete_info, 'author'].astype(str)
    )
    # Assign unique group ID for rows with missing lat/lon/author initially
    # These will also be treated as distinct groups unless their class is common enough
    # to be subject to potential ungrouping (which is unlikely for these unique IDs).
    working_df.loc[~mask_complete_info, 'initial_group_id'] = "MISSINGKEY_" + working_df.loc[~mask_complete_info, 'samplename'].astype(str)

    # 2. Calculate per-class group counts using 'initial_group_id'
    print(f"  Calculating group diversity per class (using threshold: < {GROUP_DIVERSITY_THRESHOLD} unique groups for ungrouping)...")
    class_group_counts = working_df.groupby('primary_label')['initial_group_id'].nunique()
    
    classes_to_ungroup = class_group_counts[class_group_counts < GROUP_DIVERSITY_THRESHOLD].index.tolist()
    print(f"  Found {len(classes_to_ungroup)} classes to be ungrouped (samples will get unique group IDs). Classes: {classes_to_ungroup if classes_to_ungroup else 'None'}")

    # 3. Create final 'group_id' column for splitting
    # Initialize final_group_id with initial_group_id
    working_df['final_group_id'] = working_df['initial_group_id']

    # For classes identified for ungrouping, assign a TRULY unique group ID to each sample row
    # This ensures StratifiedGroupKFold can split them to maintain stratification for these rare-group classes.
    ungroup_mask = working_df['primary_label'].isin(classes_to_ungroup)
    if ungroup_mask.any(): # Check if there are any samples to ungroup
        # Create a Series of unique IDs for the rows that need ungrouping
        # Using a combination of a prefix and the DataFrame index ensures uniqueness for these rows.
        unique_row_ids = working_df.index[ungroup_mask].astype(str)
        working_df.loc[ungroup_mask, 'final_group_id'] = "UNGROUPED_ROWID_" + unique_row_ids
    
    num_actually_ungrouped_samples = ungroup_mask.sum()
    if num_actually_ungrouped_samples > 0:
        print(f"  Assigned unique group IDs to {num_actually_ungrouped_samples} samples belonging to these {len(classes_to_ungroup)} classes.")

    groups = working_df['final_group_id']
    num_unique_final_groups = groups.nunique()
    print(f"  Total number of unique groups for StratifiedGroupKFold (after conditional ungrouping): {num_unique_final_groups}")

    if num_unique_final_groups < config.n_fold:
        print(f"  WARNING: Total unique final groups ({num_unique_final_groups}) is less than n_fold ({config.n_fold}). StratifiedGroupKFold might error or behave unexpectedly.")

    print(f"  Attempting to use StratifiedGroupKFold with {config.n_fold} splits using these final groups.")
    try:
        sgkf = StratifiedGroupKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
        fold_iterator = sgkf.split(X=working_df, y=working_df['primary_label'], groups=groups)
        print(f"  Successfully using StratifiedGroupKFold with conditional ungrouping.")
    except ValueError as e_sgkf:
        print(f"  WARNING: StratifiedGroupKFold with conditional ungrouping failed: {e_sgkf}")
        print(f"  This can happen if a group is too small or if a class is present in too few groups even after ungrouping strategy.")
        print(f"  Falling back to simple StratifiedKFold as a last resort for this run.")
        # Fallback to StratifiedKFold if StratifiedGroupKFold still fails with modified groups
        skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
        fold_iterator = skf.split(X=working_df, y=working_df['primary_label'])
        print(f"  Using StratifiedKFold instead.")

    all_folds_history = []
    best_epoch_details_all_folds = [] 
    single_fold_best_auc = 0.0 
    overall_species_ids_for_run = None # Initialize to store species_ids
    
    # New tracker for spectrograms logged in train_one_epoch
    logged_original_samplenames_for_spectrograms = [] # Using a list to preserve order of first N logged
    
    try: # Wrap the main training loop in try/finally for wandb.finish()
        for fold, (train_idx, val_idx) in enumerate(fold_iterator):
            if fold not in config.selected_folds:
                continue

            # --- wandb: Define custom step for this fold's metrics ---
            if wandb_run: # Check if wandb run is active
                wandb.define_metric(f"fold_{fold}/epoch")
                wandb.define_metric(f"fold_{fold}/*", step_metric=f"fold_{fold}/epoch")

            print(f'\n{"="*30} Fold {fold} {"="*30}')
            # --- Initialize history for the CURRENT fold --- #
            fold_history = {
                'epochs': [],
                'train_loss': [], 'val_loss': [],
                'train_auc': [], 'val_auc': []
            }

            train_df_fold = working_df.iloc[train_idx].reset_index(drop=True)
            val_df_fold = working_df.iloc[val_idx].reset_index(drop=True)

            # --- Filter Validation Set: Allow 'main' or 'rare' data sources (existing filter, keep if needed) ---
            original_val_count_before_source_filter = len(val_df_fold)
            if 'data_source' in val_df_fold.columns:
                allowed_sources = ['main', 'rare'] 
                val_df_fold = val_df_fold[val_df_fold['data_source'].isin(allowed_sources)].reset_index(drop=True)
                print(f"Fold {fold}: Filtered validation set by data_source ('main', 'rare'). Count: {original_val_count_before_source_filter} -> {len(val_df_fold)}")
            # --- End Validation Set Filtering by source ---
            
            print(f'Training set size: {len(train_df_fold)} samples')
            print(f'Initial validation set size (after source filter, before NPZ filter): {len(val_df_fold)} samples')

            # Determine which spectrogram dictionary to use for validation and filter val_df_fold
            current_val_spectrogram_source_for_dataset = all_spectrograms # Default to main spectrograms for dataset instantiation

            if val_spectrogram_data is not None and len(val_spectrogram_data) > 0:
                print(f"Fold {fold}: Using dedicated validation spectrograms from PREPROCESSED_NPZ_PATH_VAL.")
                original_val_fold_count_before_npz_filter = len(val_df_fold)
                
                # Filter val_df_fold to include only samplenames present in val_spectrogram_data
                val_df_fold = val_df_fold[val_df_fold['samplename'].isin(val_spectrogram_data.keys())].reset_index(drop=True)
                
                filtered_val_fold_count_after_npz_filter = len(val_df_fold)
                print(f"  Filtered val_df_fold to {filtered_val_fold_count_after_npz_filter} samples (from {original_val_fold_count_before_npz_filter}) based on keys in dedicated validation NPZ.")
                
                if filtered_val_fold_count_after_npz_filter == 0 and original_val_fold_count_before_npz_filter > 0:
                    print(f"  WARNING: Fold {fold} validation set became empty after filtering against dedicated validation NPZ keys. Check samplenames and VAL NPZ content.")
                
                current_val_spectrogram_source_for_dataset = val_spectrogram_data # Use these for val_dataset

            elif val_spectrogram_data is not None and len(val_spectrogram_data) == 0:
                print(f"Fold {fold}: Dedicated validation spectrograms NPZ was loaded but is empty. Using training set spectrograms for validation if samplenames match.")
            else: # val_spectrogram_data is None
                print(f"Fold {fold}: No dedicated validation spectrograms provided or loaded. Using training set spectrograms for validation if samplenames match.")

            # --- DIAGNOSTIC PRINTS FOR CLASS DISTRIBUTION (on the now potentially filtered val_df_fold) ---
            print(f"\n--- Fold {fold} Class Distribution Diagnostics ---")
            # Training set distribution
            train_label_counts = train_df_fold['primary_label'].value_counts().sort_index()
            print(f"Fold {fold} - Training set primary_label distribution ({len(train_label_counts)} classes):")

            # Validation set distribution
            val_label_counts = val_df_fold['primary_label'].value_counts().sort_index()
            print(f"Fold {fold} - Validation set primary_label distribution ({len(val_label_counts)} classes):")

            if not train_label_counts.empty and not val_label_counts.empty:
                train_classes = set(train_label_counts.index)
                val_classes = set(val_label_counts.index)

                classes_in_train_only = train_classes - val_classes
                classes_in_val_only = val_classes - train_classes # Should be rare

                if classes_in_train_only:
                    print(f"WARNING: Fold {fold} - Classes in TRAIN but NOT in VALIDATION ({len(classes_in_train_only)} classes)")
                if classes_in_val_only:
                    print(f"WARNING: Fold {fold} - Classes in VALIDATION but NOT in TRAIN ({len(classes_in_val_only)} classes)")

                # Check for classes with very few validation samples (e.g., < 5, or configurable)
                low_sample_threshold = 5 # Can be adjusted
                low_sample_val_classes = val_label_counts[val_label_counts < low_sample_threshold]
                if not low_sample_val_classes.empty:
                    print(f"WARNING: Fold {fold} - Classes in VALIDATION with < {low_sample_threshold} samples ({len(low_sample_val_classes)} classes):")
            print(f"--- End Fold {fold} Class Distribution Diagnostics ---\n")
            # --- END DIAGNOSTIC PRINTS ---

            # Pass the pre-loaded dictionary (or None) to the Dataset
            # NOW, pass both the hardcoded targets AND the shared tracking set (REMOVED THESE)
            train_dataset = BirdCLEFDataset(train_df_fold, config, mode='train', all_spectrograms=all_spectrograms)
            # Use the potentially filtered val_df_fold and the correct spectrogram source for val_dataset
            val_dataset = BirdCLEFDataset(val_df_fold, config, mode='valid', all_spectrograms=current_val_spectrogram_source_for_dataset)

            # Get species_ids from the first validation dataset instance created
            if overall_species_ids_for_run is None and hasattr(val_dataset, 'species_ids'):
                overall_species_ids_for_run = val_dataset.species_ids

            train_sampler = None
            shuffle_train_loader = True

            if config.USE_WEIGHTED_SAMPLING:
                print("Using WeightedRandomSampler for training data based on 'samplename'.")
                # Extract species identifiers from the 'samplename' column of the current fold's training data
                # Assumes train_df_fold['samplename'] exists and is in format 'species_id-other_info'
                try:
                    all_species_ids_for_fold_samples = [sn.split('-')[0] for sn in train_df_fold['samplename']]
                    species_id_counts_in_fold = Counter(all_species_ids_for_fold_samples)
                    
                    # Calculate log-based weights: 1 / log(count + 1)
                    log_weights_raw = []
                    for sid in all_species_ids_for_fold_samples:
                        count = species_id_counts_in_fold[sid] 
                        weight = 1.0 / math.log(count + 1) 
                        log_weights_raw.append(weight)

                    weights_tensor = torch.tensor(log_weights_raw, dtype=torch.float)
                    train_sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
                    shuffle_train_loader = False # Sampler handles shuffling
                    print(f"  WeightedRandomSampler created for fold {fold} with {len(weights_tensor)} log-based weights.")
                except Exception as e_sampler:
                    print(f"ERROR creating WeightedRandomSampler for fold {fold}: {e_sampler}. Defaulting to shuffle=True")
                    train_sampler = None
                    shuffle_train_loader = True
            else:
                print("Not using WeightedRandomSampler for training data. Standard shuffling will be used.")

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                shuffle=shuffle_train_loader,
                num_workers=config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
                persistent_workers=True if config.num_workers > 0 else False,
                sampler=train_sampler
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.val_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False,
                persistent_workers=True if config.num_workers > 0 else False
            )

            print("\nSetting up model, optimizer, criterion, scheduler...")
            
            # --- Model Instantiation based on config.model_name ---
            if "mn" in config.model_name.lower():
                print(f"Loading EfficientAT MobileNet model: {config.model_name}")
                with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
                    model = get_efficient_at_model(
                        num_classes=config.num_classes,          
                        pretrained_name=config.model_name, # e.g., 'mn10_as', from config
                        width_mult=config.width_mult,      # from config
                        head_type="mlp",                         
                        input_dim_f=config.TARGET_SHAPE[0], # Use TARGET_SHAPE from config
                        input_dim_t=config.TARGET_SHAPE[1], # Use TARGET_SHAPE from config
                        dropout=config.dropout # General dropout from config
                    ).to(config.device)
                print(f"  MobileNet '{config.model_name}' instantiated with input shape {config.TARGET_SHAPE}.")
            elif "efficientnet" in config.model_name.lower():
                print(f"Loading BirdCLEFModel (EfficientNet): {config.model_name}")
                model = get_efficientnet_model(config=config).to(config.device) # Pass the whole config
                print(f"  EfficientNet (BirdCLEFModel wrapper for '{config.model_name}') instantiated. Expects input {config.TARGET_SHAPE} (after dataset prep).")
            else:
                raise ValueError(f"Unsupported config.model_name: '{config.model_name}'. Must contain 'mn' or 'efficientnet'.")
            
            optimizer = get_optimizer(model, config)
            criterion = get_criterion(config)
            scheduler = get_scheduler(optimizer, config)

            scaler = GradScaler(enabled=config.use_amp)

            print(f"Automatic Mixed Precision (AMP): {'Enabled' if config.use_amp and scaler.is_enabled() else 'Disabled'}")

            best_val_auc = 0.0
            best_epoch = 0

            best_epoch_val_per_class_metrics = None
            best_epoch_val_all_outputs = None
            best_epoch_val_all_targets = None

            # --- Epoch Loop --- #
            for epoch in range(config.epochs):
                print(f"\nEpoch {epoch + 1}/{config.epochs}")

                train_loss, train_auc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    config.device,
                    scaler,
                    logged_original_samplenames_for_spectrograms,
                    config.NUM_SPECTROGRAM_SAMPLES_TO_LOG,
                    scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
                )

                # Modified to receive new metrics
                val_loss, val_auc, val_per_class_metrics, val_all_outputs, val_all_targets = validate(
                    model,
                    val_loader,
                    criterion,
                    config.device
                )

                if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                print(f"Epoch {epoch+1} Summary:")
                print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
                print(f"Val Loss:   {val_loss:.4f}, Val AUC:   {val_auc:.4f}")

                # --- Append metrics for the CURRENT fold's history --- #
                fold_history['epochs'].append(epoch + 1)
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_auc'].append(train_auc)
                fold_history['val_auc'].append(val_auc)

                # --- wandb logging for epoch metrics ---
                log_metrics = {
                    f'fold_{fold}/epoch': epoch, # Log the actual epoch number for this fold
                    f'fold_{fold}/train_loss': train_loss,
                    f'fold_{fold}/train_auc': train_auc,
                    f'fold_{fold}/val_loss': val_loss,
                    f'fold_{fold}/val_auc': val_auc,
                    f'fold_{fold}/lr': optimizer.param_groups[0]['lr']
                }
                if wandb_run: # Check if wandb run is active
                    wandb.log(log_metrics) # Let wandb use the defined step_metric

                # --- HPO Pruning --- #
                if is_hpo_trial:
                    is_first_hpo_fold_in_trial = (hpo_step_offset == 0)
                    local_epoch_in_fold = epoch # 0-indexed epoch within the current fold
                    current_global_step = hpo_step_offset + local_epoch_in_fold
                    
                    # Number of initial epochs in non-first HPO folds to skip reporting for.
                    # config.epochs here is the number of epochs set for this HPO fold.
                    hpo_fold_skip_report_epochs = config.epochs // 2
                                                     
                    should_report_this_hpo_step = True
                    if not is_first_hpo_fold_in_trial and local_epoch_in_fold < hpo_fold_skip_report_epochs:
                        should_report_this_hpo_step = False
                        # Optional: print a message for clarity during HPO runs
                        # print(f"DEBUG HPO: Trial {trial.number}, SKIPPING report for global_step {current_global_step} (local_epoch {local_epoch_in_fold} of a non-first HPO fold).")

                    if should_report_this_hpo_step:
                        # print(f"DEBUG HPO: Trial {trial.number}, REPORTING val_auc {val_auc:.4f} for global_step {current_global_step}.") # Optional debug
                        trial.report(val_auc, current_global_step) # Report intermediate val_auc with global step
                        if trial.should_prune():
                            print(f"  Pruning HPO trial {trial.number} based on value at global_step {current_global_step} (val_auc: {val_auc:.4f}).")
                            # Clean up before pruning to release GPU memory if possible for next trial
                            del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
                            if 'train_df_fold' in locals(): del train_df_fold
                            if 'val_df_fold' in locals(): del val_df_fold
                            torch.cuda.empty_cache()
                            gc.collect()
                            raise optuna.TrialPruned() # Raise exception to stop training

                # --- Model Checkpointing --- #
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    # New: Store detailed metrics for the best epoch
                    best_epoch_val_per_class_metrics = val_per_class_metrics
                    best_epoch_val_all_outputs = val_all_outputs
                    best_epoch_val_all_targets = val_all_targets
                    
                    print(f"✨ New best AUC: {best_val_auc:.4f} at epoch {best_epoch}. Saving model...")

                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'epoch': epoch,
                        'val_auc': best_val_auc,
                        'train_auc': train_auc,
                        # Optionally, store per-class metrics in checkpoint if desired, though it can make files large
                        # 'val_per_class_metrics': val_per_class_metrics 
                    }
                    save_path = os.path.join(config.MODEL_OUTPUT_DIR, f"{config.model_name}_fold{fold}_best.pth")
                    try:
                        torch.save(checkpoint, save_path)
                        print(f"  Model saved to {save_path}")
                    except Exception as e:
                        print(f"  Error saving model checkpoint: {e}") # Removed artifact logging from here
                # --- EPOCH LOOP ENDS HERE ---

            # --- Code to run AFTER all epochs for the current FOLD are done ---
            print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
            
            # New: Store the best epoch details for this fold for later OOF aggregation
            if best_epoch_val_per_class_metrics is not None: # Ensure we have data from a best epoch
                best_epoch_details_all_folds.append({
                    'fold': fold,
                    'best_val_auc': best_val_auc,
                    'best_epoch': best_epoch,
                    'per_class_metrics': best_epoch_val_per_class_metrics,
                    'outputs': best_epoch_val_all_outputs,
                    'targets': best_epoch_val_all_targets
                })

            # Log best AUC for this fold to wandb summary for easy viewing
            if wandb_run: # Check if wandb run is active
                wandb.summary[f'fold_{fold}_best_val_auc'] = best_val_auc
            single_fold_best_auc = best_val_auc

            all_folds_history.append(fold_history)

            # Clean up resources for the current fold before starting the next one
            del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
            del train_df_fold, val_df_fold
            torch.cuda.empty_cache()
            gc.collect()
            # --- End of FOLD specific cleanup ---

        # --- FOLD LOOP ENDS HERE (or continues to next fold) ---

        # --- Code to run AFTER ALL SELECTED FOLDS are done ---
        if not is_hpo_trial:
            # --- Averaged Training History Plot (existing code) ---
            if all_folds_history:
                print("\n--- Generating Average Training History Plot Across Folds ---")

                # Calculate average metrics per epoch
                num_epochs = config.epochs
                avg_train_loss = np.zeros(num_epochs)
                avg_val_loss = np.zeros(num_epochs)
                avg_train_auc = np.zeros(num_epochs)
                avg_val_auc = np.zeros(num_epochs)
                counts_per_epoch = np.zeros(num_epochs, dtype=int)

                for fold_hist in all_folds_history:
                    # Use the actual number of epochs recorded in the history for this fold
                    epochs_ran = len(fold_hist['epochs'])
                    for i in range(epochs_ran):
                        epoch_idx = i
                        if epoch_idx < num_epochs:
                            avg_train_loss[epoch_idx] += fold_hist['train_loss'][i]
                            avg_val_loss[epoch_idx] += fold_hist['val_loss'][i]
                            avg_train_auc[epoch_idx] += fold_hist['train_auc'][i]
                            avg_val_auc[epoch_idx] += fold_hist['val_auc'][i]
                            counts_per_epoch[epoch_idx] += 1

                # Avoid division by zero if no folds ran or epochs were skipped
                valid_counts_mask = counts_per_epoch > 0
                avg_train_loss[valid_counts_mask] /= counts_per_epoch[valid_counts_mask]
                avg_val_loss[valid_counts_mask] /= counts_per_epoch[valid_counts_mask]
                avg_train_auc[valid_counts_mask] /= counts_per_epoch[valid_counts_mask]
                avg_val_auc[valid_counts_mask] /= counts_per_epoch[valid_counts_mask]

                epochs_axis = list(range(1, num_epochs + 1))

                # Generate Plot
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))

                # Loss Plot
                ax[0].plot(epochs_axis, avg_train_loss, label='Avg Train Loss')
                ax[0].plot(epochs_axis, avg_val_loss, label='Avg Validation Loss')
                ax[0].set_title('Average Loss vs. Epochs Across Folds')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].legend()
                ax[0].grid(True)

                # AUC Plot
                ax[1].plot(epochs_axis, avg_train_auc, label='Avg Train AUC')
                ax[1].plot(epochs_axis, avg_val_auc, label='Avg Validation AUC')
                ax[1].set_title('Average AUC vs. Epochs Across Folds')
                ax[1].set_xlabel('Epoch')
                ax[1].set_ylabel('AUC')
                ax[1].legend()
                ax[1].grid(True)

                plt.tight_layout()

                plot_dir = os.path.join(config.OUTPUT_DIR, "plots", "training_curves")
                os.makedirs(plot_dir, exist_ok=True)
                plot_save_path = os.path.join(plot_dir, "all_folds_training_plot.png")

                try:
                    plt.savefig(plot_save_path)
                    print(f"Saved average plot to: {plot_save_path}")
                    # --- wandb log training plot ---
                    if wandb_run: # Check if wandb run is active
                        wandb.log({"training_plot": wandb.Image(plot_save_path)})
                except Exception as e:
                    print(f"Error saving average plot to {plot_save_path}: {e}")
                plt.close(fig)
            else:
                print("\nNo fold histories recorded, skipping average plot generation.")

            # --- New: Aggregate and Log Per-Class Metrics and OOF Metrics ---
            if best_epoch_details_all_folds:
                print("\n--- Aggregating and Logging Per-Class & OOF Metrics ---")
                
                # Use the stored overall_species_ids_for_run
                if overall_species_ids_for_run is None:
                    print("ERROR: Could not determine species_ids for metric aggregation. Skipping.")
                    # Potentially skip the rest of this block or handle error appropriately
                else:
                    species_ids = overall_species_ids_for_run
                    num_classes = len(species_ids)

                    # 1. Average Per-Class Metrics from Best Epoch of Each Fold
                    # Initialize a dictionary to sum metrics for each class across folds
                    sum_per_class_metrics = {sid: {m: 0.0 for m in ['auc', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn']} for sid in species_ids}
                    num_folds_with_details = len(best_epoch_details_all_folds)

                    for fold_details in best_epoch_details_all_folds:
                        for class_metric in fold_details['per_class_metrics']:
                            sid = class_metric['species_id']
                            if sid in sum_per_class_metrics:
                                sum_per_class_metrics[sid]['auc'] += class_metric['auc']
                                sum_per_class_metrics[sid]['precision'] += class_metric['precision']
                                sum_per_class_metrics[sid]['recall'] += class_metric['recall']
                                sum_per_class_metrics[sid]['f1'] += class_metric['f1']
                                sum_per_class_metrics[sid]['tp'] += class_metric['tp']
                                sum_per_class_metrics[sid]['fp'] += class_metric['fp']
                                sum_per_class_metrics[sid]['fn'] += class_metric['fn']
                                sum_per_class_metrics[sid]['tn'] += class_metric['tn']
                    
                    avg_per_class_metrics_list = []
                    if num_folds_with_details > 0:
                        for sid in species_ids:
                            avg_metrics = {metric: val / num_folds_with_details for metric, val in sum_per_class_metrics[sid].items()}
                            avg_metrics['species_id'] = sid
                            avg_per_class_metrics_list.append(avg_metrics)
                    
                    if avg_per_class_metrics_list:
                        avg_per_class_df = pd.DataFrame(avg_per_class_metrics_list)
                        if wandb_run:
                            try:
                                wandb.log({"avg_best_epoch_per_class_metrics": wandb.Table(dataframe=avg_per_class_df)})
                                print("  Logged average best epoch per-class metrics to W&B table.")
                            except Exception as e_wandb_table:
                                print(f"  Error logging avg_best_epoch_per_class_metrics table to W&B: {e_wandb_table}")

                    # 2. Calculate OOF Per-Class Metrics
                    oof_all_targets = []
                    oof_all_outputs = [] # logits
                    for fold_details in best_epoch_details_all_folds:
                        if fold_details['targets'] is not None and fold_details['outputs'] is not None:
                            oof_all_targets.append(fold_details['targets'])
                            oof_all_outputs.append(fold_details['outputs'])

                    if oof_all_targets and oof_all_outputs:
                        oof_targets_cat = np.concatenate(oof_all_targets)
                        oof_outputs_cat = np.concatenate(oof_all_outputs) # These are logits
                        
                        oof_per_class_metrics_list = []
                        oof_probabilities_cat = 1 / (1 + np.exp(-oof_outputs_cat)) # Sigmoid probabilities
                        oof_predictions_binary_cat = (oof_probabilities_cat >= 0.5).astype(int)

                        try:
                            oof_mcm = multilabel_confusion_matrix(oof_targets_cat, oof_predictions_binary_cat, labels=list(range(num_classes)))
                        except Exception as e_oof_mcm:
                            print(f"  Error calculating OOF multilabel_confusion_matrix: {e_oof_mcm}. TP/FP/FN/TN will be zero.")
                            oof_mcm = np.zeros((num_classes, 2, 2), dtype=int) # Fallback

                        for i in range(num_classes):
                            class_targets = oof_targets_cat[:, i]
                            class_probs = oof_probabilities_cat[:, i]
                            class_preds_binary = oof_predictions_binary_cat[:, i]
                            species_name = species_ids[i]

                            class_auc = 0.0
                            if np.sum(class_targets) > 0 and np.sum(1 - class_targets) > 0:
                                try: class_auc = roc_auc_score(class_targets, class_probs)
                                except ValueError: class_auc = 0.0
                            
                            precision, recall, f1_score, _ = precision_recall_fscore_support(
                                class_targets, class_preds_binary, average='binary', pos_label=1, zero_division=0
                            )
                            tn, fp, fn, tp = oof_mcm[i].ravel()

                            oof_per_class_metrics_list.append({
                                'species_id': species_name, 'auc': class_auc, 'precision': precision,
                                'recall': recall, 'f1': f1_score, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
                            })
                        
                        if oof_per_class_metrics_list:
                            oof_per_class_df = pd.DataFrame(oof_per_class_metrics_list)
                            if wandb_run:
                                try:
                                    wandb.log({"oof_per_class_metrics": wandb.Table(dataframe=oof_per_class_df)})
                                    print("  Logged OOF per-class metrics to W&B table.")
                                except Exception as e_wandb_table:
                                    print(f"  Error logging oof_per_class_metrics table to W&B: {e_wandb_table}")
                    else:
                        print("  No OOF details found, skipping OOF per-class metrics calculation.")
            else:
                print("\nNo best_epoch_details recorded, skipping per-class & OOF metric aggregation.")

            # --- Non-HPO Summary (existing code, ensure it uses the correct data) ---
            if all_folds_history: # This check remains for the general OOF AUC from history
                oof_scores_from_hist = [max(h['val_auc']) for h in all_folds_history if h['val_auc']] # Get best AUC from each fold history
                mean_oof_auc = np.mean(oof_scores_from_hist) if oof_scores_from_hist else 0.0
                print("\n" + "="*60)
                print("Cross-Validation Training Summary:")
                num_folds_run = len(all_folds_history)
                for i in range(num_folds_run):
                     fold_num = config.selected_folds[i]
                     best_fold_auc = max(all_folds_history[i]['val_auc']) if all_folds_history[i]['val_auc'] else 0.0
                     print(f"  Fold {fold_num}: Best Val AUC = {best_fold_auc:.4f}")
                print(f"\nMean OOF AUC across {len(oof_scores_from_hist)} trained folds: {mean_oof_auc:.4f}")
                print("="*60)
                if wandb_run: # Check if wandb run is active
                    wandb.summary['mean_oof_auc'] = mean_oof_auc # Log overall mean OOF AUC
            else:
                print("\nNo folds were trained.")
                print("="*60)

        if is_hpo_trial:
            print(f"\nReturning best AUC for HPO Trial (Fold {config.selected_folds[0]}): {single_fold_best_auc:.4f}")
            # For HPO, wandb.finish() will be called in the finally block
            return single_fold_best_auc
        else:
            # For standard runs, calculate and return the mean OOF AUC if multiple folds ran
            # (This part is largely for printing, wandb summary already updated)
            mean_oof_auc_final = 0.0
            if all_folds_history:
                oof_scores_from_hist = [max(h['val_auc']) for h in all_folds_history if h.get('val_auc')] # Safer access
                if oof_scores_from_hist: # Ensure list is not empty before mean
                    mean_oof_auc_final = np.mean(oof_scores_from_hist)

                # Update summary if the run is still considered active by the W&B library
                if wandb_run: # Use the local wandb_run variable
                    try:
                        wandb.summary['mean_oof_auc'] = mean_oof_auc_final
                        print(f"DEBUG: Updated wandb.summary['mean_oof_auc'] = {mean_oof_auc_final:.4f}")
                    except Exception as e_summary:
                        print(f"DEBUG: Error updating wandb.summary for mean_oof_auc: {e_summary}")

            # Determine return value after the finally block has executed
            mean_oof_auc_to_return = 0.0
            if all_folds_history:
                oof_scores_from_hist_return = [max(h['val_auc']) for h in all_folds_history if h.get('val_auc')] # Safer access
                if oof_scores_from_hist_return:
                    mean_oof_auc_to_return = np.mean(oof_scores_from_hist_return)

            print("\n" + "="*60)
            if all_folds_history and any(h.get('val_auc') for h in all_folds_history):
                print("Final Cross-Validation Training Summary:")
                # Ensure selected_folds has enough elements if all_folds_history is populated
                num_folds_actually_run = len(all_folds_history)
                for i in range(num_folds_actually_run):
                     fold_num_display = config.selected_folds[i] if i < len(config.selected_folds) else f"FoldIndex_{i}"
                     # Get val_auc history for the current fold, check if it's not empty
                     current_fold_val_auc_history = all_folds_history[i].get('val_auc', [])
                     best_fold_auc = max(current_fold_val_auc_history) if current_fold_val_auc_history else 0.0
                     print(f"  Fold {fold_num_display}: Best Val AUC = {best_fold_auc:.4f}")
                print(f"\nMean OOF AUC across {len(oof_scores_from_hist_return) if oof_scores_from_hist_return else 0} trained folds: {mean_oof_auc_to_return:.4f}")
            else:
                print("No folds were trained or no validation AUCs recorded.")
            print("="*60)
            return mean_oof_auc_to_return
    finally:
        if wandb_run:
            wandb_run.finish() # Ensure wandb run is finished

if __name__ == "__main__":
    print("\n--- Initializing Training Script ---")
    print(f"Using configuration: LOAD_PREPROCESSED_DATA={config.LOAD_PREPROCESSED_DATA}, USE_PSEUDO_LABELS={config.USE_PSEUDO_LABELS}")

    set_seed(config.seed)

    all_spectrograms = None 

    print("Loading main training metadata...")
    try:
        main_train_df_full = pd.read_csv(config.train_csv_path)
        main_train_df_full['filepath'] = main_train_df_full['filename'].apply(lambda f: os.path.join(config.train_audio_dir, f))
        # Add samplename: e.g. 1139490/CSA36385.ogg -> 1139490-CSA36385
        main_train_df_full['samplename'] = main_train_df_full.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        main_train_df_full['data_source'] = 'main'
        
        # Select only necessary columns for training
        required_cols_main = ['samplename', 'primary_label', 'secondary_labels', 'filepath', 'filename', 'data_source', 'latitude', 'longitude', 'author'] # Added author
        main_train_df = main_train_df_full[required_cols_main].copy()
        print(f"Loaded and selected columns for {len(main_train_df)} main training samples.")
        del main_train_df_full; gc.collect()
    except Exception as e:
        print(f"CRITICAL ERROR loading main training CSV {config.train_csv_path}: {e}. Exiting.")
        sys.exit(1)

    training_df = main_train_df 

    # --- Load Preprocessed Spectrograms (if configured) --- #
    if config.LOAD_PREPROCESSED_DATA:
        all_spectrograms = {} 
        
        # Load PRIMARY spectrograms
        primary_npz_path = config.PREPROCESSED_NPZ_PATH
        print(f"Attempting to load primary spectrograms from: {primary_npz_path}")
        try:
            start_load_time = time.time()
            with np.load(primary_npz_path) as data_archive:
                primary_specs = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Primary Specs")}
            end_load_time = time.time()
            all_spectrograms.update(primary_specs)
            print(f"Successfully loaded {len(primary_specs)} primary samples in {end_load_time - start_load_time:.2f} seconds.")
            del primary_specs; gc.collect()
        except Exception as e:
            print(f"CRITICAL ERROR loading primary NPZ file {primary_npz_path}: {e}")
            sys.exit(1)

        # --- Conditionally Load PSEUDO-LABEL Data & Spectrograms --- #
        if config.USE_PSEUDO_LABELS:
            print("\n--- Loading Pseudo-Label Data (USE_PSEUDO_LABELS=True) ---")
            
            try:
                pseudo_labels_df_full = pd.read_csv(config.train_pseudo_csv_path)
                if not pseudo_labels_df_full.empty:
                    # Create required derived columns
                    pseudo_labels_df_full['samplename'] = pseudo_labels_df_full.apply(
                        lambda row: f"{row['filename']}_{int(row['start_time'])}_{int(row['end_time'])}", axis=1
                    )
                    pseudo_labels_df_full['filepath'] = pseudo_labels_df_full['filename'].apply(lambda f: os.path.join(config.unlabeled_audio_dir, f))
                    pseudo_labels_df_full['data_source'] = 'pseudo' # Add data source identifier
                    
                    # Select necessary columns including confidence for filtering later
                    required_cols_pseudo = ['samplename', 'primary_label', 'filepath', 'filename', 'data_source', 'confidence'] # Include data_source AND confidence
                    pseudo_labels_df = pseudo_labels_df_full[required_cols_pseudo].copy()
                    print(f"Loaded and selected columns (including confidence) for {len(pseudo_labels_df)} pseudo labels.")
                    del pseudo_labels_df_full; gc.collect()

                    # Load pseudo spectrograms
                    pseudo_npz_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, 'pseudo_spectrograms.npz')
                    print(f"Attempting to load pseudo spectrograms from: {pseudo_npz_path}")
                    
                    try:
                        start_load_time = time.time()
                        with np.load(pseudo_npz_path) as data_archive:
                            pseudo_specs = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Pseudo Specs")}
                        end_load_time = time.time()
                        all_spectrograms.update(pseudo_specs) # Merge into the main dictionary
                        print(f"Successfully loaded and merged {len(pseudo_specs)} pseudo samples in {end_load_time - start_load_time:.2f} seconds.")
                        del pseudo_specs; gc.collect()
                        
                        training_df = pd.concat([training_df, pseudo_labels_df], ignore_index=True)
                        print(f"Combined DataFrame size: {len(training_df)} samples.")
                        
                    except Exception as e:
                         print(f"ERROR loading pseudo NPZ file {pseudo_npz_path}: {e}")
                         print("Continuing training without pseudo-labels due to NPZ loading error.")
                else:
                    print("Pseudo labels CSV found but is empty. Skipping.")

            except Exception as e:
                print(f"CRITICAL ERROR loading or processing pseudo labels CSV {config.train_pseudo_csv_path}: {e}")
                sys.exit(1)
        else:
            print("\nSkipping pseudo-label data (USE_PSEUDO_LABELS=False).")

    # Final check on combined dataframe and spectrograms
    print(f"\nFinal training dataframe size: {len(training_df)} samples.")
    
    # --- Filter training_df based on loaded spectrogram keys if configured to load them --- #
    if config.LOAD_PREPROCESSED_DATA:
        print(f"\nConfigured to load preprocessed data. Filtering dataframe...")
        print(f"Total pre-loaded spectrogram keys available: {len(all_spectrograms)}")
        
        original_count = len(training_df)

        loaded_keys = set(all_spectrograms.keys())
        training_df = training_df[training_df['samplename'].isin(loaded_keys)].reset_index(drop=True)
        filtered_count = len(training_df)
        
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"  WARNING: Removed {removed_count} samples from training_df because their spectrograms were not found in the loaded NPZ file(s).")
        
        print(f"  Final training_df size after filtering: {filtered_count} samples.")
    else:
        print("\nWarning: all_spectrograms is None. Cannot filter training_df by loaded keys (which is expected if not loading preprocessed data).")

    # --- Load Preprocessed Validation Spectrograms (if available) ---
    loaded_val_spectrograms = None # Initialize
    if hasattr(config, 'PREPROCESSED_NPZ_PATH_VAL') and config.PREPROCESSED_NPZ_PATH_VAL:
        val_npz_path = config.PREPROCESSED_NPZ_PATH_VAL
        print(f"\nAttempting to load DEDICATED VALIDATION spectrograms from: {val_npz_path}")
        if os.path.exists(val_npz_path):
            try:
                start_load_time_val = time.time()
                with np.load(val_npz_path) as data_archive_val:
                    # Ensure tqdm is available if you use it here, or remove for simplicity if not essential for this loading step
                    loaded_val_spectrograms = {key: data_archive_val[key] for key in tqdm(data_archive_val.keys(), desc="Loading DEDICATED VAL Specs")}
                end_load_time_val = time.time()
                print(f"Successfully loaded {len(loaded_val_spectrograms) if loaded_val_spectrograms is not None else 0} DEDICATED VALIDATION sample entries in {end_load_time_val - start_load_time_val:.2f} seconds.")
                if loaded_val_spectrograms is not None and not loaded_val_spectrograms: # Check if loaded but empty
                    print("Warning: DEDICATED VALIDATION NPZ file was loaded but contained no spectrograms.")
            except Exception as e_val_load:
                print(f"ERROR loading DEDICATED VALIDATION NPZ file {val_npz_path}: {e_val_load}. Proceeding without dedicated validation specs.")
                loaded_val_spectrograms = None # Ensure it's None on error
        else:
            print(f"Info: DEDICATED VALIDATION NPZ file not found at {val_npz_path}. Validation may use training spectrogram processing rules or be empty if filtered by missing keys.")
    else:
        print("\nInfo: config.PREPROCESSED_NPZ_PATH_VAL not defined. Validation may use training spectrogram processing rules.")

    # Pass the loaded_val_spectrograms to run_training
    run_training(training_df, config, all_spectrograms=all_spectrograms, 
                 val_spectrogram_data=loaded_val_spectrograms, # Pass the loaded validation specs
                 custom_run_name=cmd_args.run_name)

    print("\nTraining script finished!")