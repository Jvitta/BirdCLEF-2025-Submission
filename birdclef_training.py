import os
import logging
import random
import gc
import time
import math
import warnings
from pathlib import Path
import sys
import glob
import io
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
from losses import FocalLossBCE
import torchvision.transforms as transforms

from tqdm.auto import tqdm

import timm
import matplotlib.pyplot as plt
import optuna
import wandb

from config import config
import birdclef_utils as utils
from google.cloud import storage
import cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def config_to_dict(cfg):
    return {key: value for key, value in cfg.__dict__.items() if not key.startswith('__') and not callable(value)}

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.seed)

class BirdCLEFDataset(Dataset):
    """Dataset class for BirdCLEF.
    Handles loading pre-computed spectrograms from a pre-loaded dictionary
    or generating them on-the-fly.
    """
    def __init__(self, df, config, mode="train", all_spectrograms=None):
        self.df = df.copy()
        self.config = config
        self.mode = mode
        self.all_spectrograms = all_spectrograms

        # Load taxonomy to map labels to indices
        try:
            taxonomy_df = pd.read_csv(self.config.taxonomy_path)
            self.species_ids = taxonomy_df['primary_label'].tolist()
            if len(self.species_ids) != self.config.num_classes:
                print(f"Warning: Taxonomy file has {len(self.species_ids)} species, but config.num_classes is {self.config.num_classes}. Using value from taxonomy.")
                self.num_classes = len(self.species_ids)
            else:
                self.num_classes = self.config.num_classes
            self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}
        except Exception as e:
            print(f"Error loading taxonomy CSV from {self.config.taxonomy_path}: {e}")
            raise

        if self.all_spectrograms is not None:
            print(f"Dataset mode '{self.mode}': Using pre-loaded spectrogram dictionary.")
            print(f"Found {len(self.all_spectrograms)} samplenames with precomputed chunks.")
            # Optional: Add check if all df samplenames are in all_spectrograms keys
            missing_keys = set(self.df['samplename']) - set(self.all_spectrograms.keys())
            if missing_keys:
                 print(f"Warning: {len(missing_keys)} samplenames from the dataframe are missing in the loaded spectrograms dictionary.")
                 print(f"Examples: {list(missing_keys)[:5]}")
        elif self.config.LOAD_PREPROCESSED_DATA:
            print(f"Dataset mode '{self.mode}': ERROR - Configured to load preprocessed data, but none provided. Dataset will be empty.")
        else:
            print(f"Dataset mode '{self.mode}': Configured for on-the-fly generation from {len(self.df)} files.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        primary_label = row['primary_label']
        secondary_labels = row.get('secondary_labels', [])
        filename_for_error = row.get('filename', samplename) # Use filename if available
        
        spec = None # This will hold the final (H_5s, W_5s) processed chunk
        target_h_5s, target_w_5s = self.config.TARGET_SHAPE # Expected final shape, e.g., (256, 256)

        if samplename in self.all_spectrograms:
            spec_data_from_npz = self.all_spectrograms[samplename]
            raw_selected_chunk_2d = None # This will be the 2D chunk selected, potentially >5s wide

            # --- Dequantization Step --- 
            if isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.dtype == np.uint16:
                # Dequantize from uint16 [0, 65535] to float32 [0, 1]
                spec_data_from_npz = spec_data_from_npz.astype(np.float32) / 65535.0
            elif isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.dtype != np.float32:
                # If it's some other non-float32 type we weren't expecting after uint16,
                # try to convert to float32. This is a fallback.
                print(f"WARNING: Samplename '{samplename}' has unexpected dtype {spec_data_from_npz.dtype}. Converting to float32.")
                spec_data_from_npz = spec_data_from_npz.astype(np.float32)
            # If it was already float32, spec_data_from_npz is unchanged.

            if isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.ndim == 3:
                # Assumes data is always (N, H, W_chunk)
                # N is number of chunks, H is consistent height, W_chunk is width of EACH chunk.
                num_available_chunks = spec_data_from_npz.shape[0]
                
                if num_available_chunks > 0:
                    selected_idx = 0
                    if self.mode == 'train' and num_available_chunks > 1:
                        selected_idx = random.randint(0, num_available_chunks - 1)
                    raw_selected_chunk_2d = spec_data_from_npz[selected_idx]
                else:
                    print(f"WARNING: Data for '{samplename}' is a 3D array but has 0 chunks. Using zeros.")

            elif isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.ndim == 2:
                # This case should ideally be phased out if preprocessing always saves as 3D (1, H, W)
                print(f"WARNING: Data for '{samplename}' is a 2D array. Preprocessing should save as 3D (e.g., (1, H, W)). Attempting to use directly.")
                raw_selected_chunk_2d = spec_data_from_npz
            else:
                ndim_info = spec_data_from_npz.ndim if isinstance(spec_data_from_npz, np.ndarray) else "Not an ndarray"
                print(f"WARNING: Data for '{samplename}' has unexpected ndim {ndim_info} or type. Expected 3D ndarray. Using zeros.")

            # Now, raw_selected_chunk_2d should be a single 2D spectrogram.
            # Crop it if it's wider than the target 5s width.
            if raw_selected_chunk_2d is not None:
                # NEW LOGIC TO PROCESS raw_selected_chunk_2d INTO spec
                final_target_shape_tuple = tuple(self.config.TARGET_SHAPE)
                native_5s_frames = math.ceil(self.config.TARGET_DURATION * self.config.FS / self.config.HOP_LENGTH)
                
                current_h, current_w = raw_selected_chunk_2d.shape
                processed_intermediate_spec = None

                if current_h == self.config.N_MELS: # Native mel spec (e.g., 136, W_variable)
                    img_5s_native = None
                    if current_w > native_5s_frames: # Wider than 5s native, needs crop
                        if self.mode == 'train':
                            max_offset = current_w - native_5s_frames
                            offset = random.randint(0, max_offset)
                            img_5s_native = raw_selected_chunk_2d[:, offset:offset + native_5s_frames]
                        else: # eval/test: center crop
                            offset = (current_w - native_5s_frames) // 2
                            img_5s_native = raw_selected_chunk_2d[:, offset:offset + native_5s_frames]
                    elif current_w == native_5s_frames: # Already 5s native width
                        img_5s_native = raw_selected_chunk_2d
                    else: # Narrower than 5s native width, needs padding
                        padding_width = native_5s_frames - current_w
                        background_val = self.config.BACKGROUND_VALUE_MELSPEC if hasattr(self.config, 'BACKGROUND_VALUE_MELSPEC') else 0.0
                        img_5s_native = np.pad(raw_selected_chunk_2d, ((0,0), (0, padding_width)), mode='constant', constant_values=background_val)
                    
                    # Resize this (N_MELS, native_5s_frames) segment to TARGET_SHAPE.
                    if img_5s_native.shape != final_target_shape_tuple: 
                        processed_intermediate_spec = cv2.resize(img_5s_native, (self.config.TARGET_SHAPE[1], self.config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
                    else:
                        processed_intermediate_spec = img_5s_native # Should be TARGET_SHAPE if N_MELS was already TARGET_SHAPE[0] and width was TARGET_SHAPE[1]

                elif current_h == self.config.TARGET_SHAPE[0]: # Already height-resized chunk (e.g., 256, W_target_shape)
                    if raw_selected_chunk_2d.shape == final_target_shape_tuple:
                        processed_intermediate_spec = raw_selected_chunk_2d
                    else: # Fallback: width might not match, so resize to be sure
                        processed_intermediate_spec = cv2.resize(raw_selected_chunk_2d, (self.config.TARGET_SHAPE[1], self.config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    print(f"ERROR: Chunk for '{samplename}' (shape {raw_selected_chunk_2d.shape}) has unexpected height {current_h}. Expected {self.config.N_MELS} or {self.config.TARGET_SHAPE[0]}. Using zeros.")
                    processed_intermediate_spec = np.zeros(final_target_shape_tuple, dtype=np.float32)
                
                spec = processed_intermediate_spec # Assign to the 'spec' variable
                # END OF NEW LOGIC
            
            # Fallback if spec is still None or issues occurred during processing
            if spec is None or spec.shape != self.config.TARGET_SHAPE:
                 original_shape_info = spec_data_from_npz.shape if isinstance(spec_data_from_npz, np.ndarray) else type(spec_data_from_npz)
                 current_spec_shape_info = spec.shape if spec is not None else "None"
                 print(f"Fallback: Using zeros for '{samplename}'. Raw NPZ shape: {original_shape_info}, Processed spec shape before fallback: {current_spec_shape_info}.")
                 spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)
        
        else: # samplename not found in the pre-loaded spectrogram dictionary
            print(f"ERROR: Samplename '{samplename}' not found in pre-loaded dictionary! Using zeros.")
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)

        # --- Final Shape Guarantee --- (important for downstream code)
        if not isinstance(spec, np.ndarray) or spec.shape != tuple(self.config.TARGET_SHAPE):
             print(f"CRITICAL WARNING: Final spec for '{samplename}' has wrong shape/type ({spec.shape if isinstance(spec, np.ndarray) else type(spec)}) before unsqueeze. Forcing zeros.")
             spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)

        # Ensure spec is float32 before augmentations/tensor conversion
        spec = spec.astype(np.float32)

        # Apply manual SpecAugment (Time/Freq Mask, Contrast) on NumPy array
        if self.mode == "train":
            spec = self.apply_spec_augmentations(spec)

        # Convert to tensor, add channel dimension, repeat
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # Normalize the (potentially augmented) tensor
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        spec_tensor = normalize(spec_tensor)

        # Encode labels retrieved from the dataframe row
        target = self.encode_label(primary_label)
        if secondary_labels and secondary_labels not in [[''], None, np.nan]:
             # Ensure secondary_labels is a list 
             if isinstance(secondary_labels, str):
                 try: 
                     eval_labels = eval(secondary_labels) 
                     if isinstance(eval_labels, list): secondary_labels = eval_labels
                     else: secondary_labels = []
                 except: secondary_labels = []
             elif not isinstance(secondary_labels, list): secondary_labels = []
             
             # Apply valid secondary labels
             for label in secondary_labels:
                  if label in self.label_to_idx:
                       target[self.label_to_idx[label]] = 1.0 - self.config.label_smoothing_factor

        return {
            'melspec': spec_tensor,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': filename_for_error 
        }

    def apply_spec_augmentations(self, spec_np):
        """Apply augmentations to a single-channel numpy spectrogram."""
        # Time masking (horizontal stripes)
        if random.random() < self.config.time_mask_prob:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, self.config.max_time_mask_width)
                start = random.randint(0, max(0, spec_np.shape[1] - width))
                spec_np[:, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < self.config.freq_mask_prob:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, self.config.max_freq_mask_height)
                start = random.randint(0, max(0, spec_np.shape[0] - height))
                spec_np[start:start+height, :] = 0

        # Random brightness/contrast
        if random.random() < self.config.contrast_prob:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec_np = spec_np * gain + bias
            spec_np = np.clip(spec_np, 0, 1)

        return spec_np

    def encode_label(self, label):
        """Encode primary label to one-hot vector."""
        target = np.zeros(self.num_classes, dtype=np.float32)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0 - self.config.label_smoothing_factor
        return target

def collate_fn(batch):
    """Custom collate function."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("Warning: collate_fn received empty batch, returning None.")
        return None

    expected_keys = {'melspec', 'target', 'filename'}
    if not all(expected_keys.issubset(item.keys()) for item in batch):
         print("Warning: Batch items have inconsistent keys. Returning None.")
         return None

    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    try:
        result['target'] = torch.stack(result['target'])
        result['melspec'] = torch.stack(result['melspec'])
    except RuntimeError as e:
        print(f"Error stacking tensors in collate_fn: {e}. Returning None.")
        return None
    except Exception as e:
        print(f"Unexpected error in collate_fn: {e}. Returning None.")
        return None

    if result['melspec'].shape[0] != len(batch) or result['target'].shape[0] != len(batch):
        print("Warning: Collated tensors have incorrect batch dimension. Returning None.")
        return None

    return result

class BirdCLEFModel(nn.Module):
    """BirdCLEF model using timm backbone."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            drop_rate=0.2, # Consider making these configurable
            drop_path_rate=0.2 # Consider making these configurable
        )

        # Use original backbone feature extraction logic
        if 'efficientnet' in config.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, config.num_classes)

        self.mixup_enabled = config.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = config.mixup_alpha
            print(f"Mixup enabled with alpha={self.mixup_alpha}")

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits

def mixup_data(x, targets, alpha, device):
    """Applies mixup augmentation.
    Returns mixed inputs, targets_a, targets_b, and lambda.
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[indices]
    targets_a, targets_b = targets, targets[indices]
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

def cutmix_data(x, targets, alpha, device):
    """Applies CutMix augmentation.
    Returns mixed inputs, targets_a, targets_b, and lambda.
    """
    batch_size = x.size(0)
    indices = torch.randperm(batch_size, device=device)
    targets_a, targets_b = targets, targets[indices]

    # Generate lambda using beta distribution
    lam = np.random.beta(alpha, alpha)

    # Get bounding box coordinates
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Create mixed inputs by pasting the patch
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda based on actual patch area relative to image area
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
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
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
    if config.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config.criterion == 'FocalLossBCE':
        print("INFO: Using FocalLossBCE with parameters hardcoded in losses.py")
        criterion = FocalLossBCE(config=config)
    else:
        raise NotImplementedError(f"Criterion '{config.criterion}' not implemented")
    return criterion

def calculate_auc(targets, outputs):
    """Calculates macro-averaged ROC AUC."""
    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except ValueError as e:
                pass

    return np.mean(aucs) if aucs else 0.0

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler=None):
    """Runs one epoch of training with optional mixed precision."""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    use_amp = scaler.is_enabled()
    # Check if batch augmentation is globally enabled
    batch_augment_active = config.batch_augment_prob > 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if batch is None:
            print(f"Warning: Skipping None batch at step {step}")
            continue

        try:
            inputs_orig = batch['melspec'].to(device)
            targets_orig = batch['target'].to(device)
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
                    inputs_orig, targets_orig, config.mixup_alpha, device
                )
                apply_mixup = True
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs_orig, targets_orig, config.cutmix_alpha, device
                )
                apply_cutmix = True
        else:
            # No augmentation for this batch
            inputs = inputs_orig
            targets_a = targets_orig
            lam = 1.0 # Ensure lambda is 1 when no augmentation

        # --- Forward pass and Loss Calculation --- #
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)

            if apply_mixup or apply_cutmix:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets_a) # targets_a is targets_orig here

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Use original targets for AUC calculation, applying ceil for label smoothing
        outputs_np = logits.detach().float().cpu().numpy()
        targets_np = torch.ceil(targets_orig).detach().cpu().numpy() 

        all_outputs.append(outputs_np)
        all_targets.append(targets_np)
        losses.append(loss.item())

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
    all_targets = []
    all_outputs = []
    use_amp = config.use_amp

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Validation")):
            if batch is None:
                print(f"Warning: Skipping None validation batch at step {step}")
                continue

            try:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
            except (AttributeError, TypeError) as e:
                print(f"Error: Skipping validation batch {step} due to unexpected format: {e}")
                continue

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            all_outputs.append(outputs.float().cpu().numpy())
            all_targets.append(torch.ceil(targets).cpu().numpy()) #use ceil to convert to 0/1 targets for label smoothing
            losses.append(loss.item())

            if config.debug and (step + 1) >= config.debug_limit_batches:
                print(f"DEBUG: Stopping validation early after {config.debug_limit_batches} batches.")
                break

    if not all_targets or not all_outputs:
        print("Warning: No targets or outputs collected during validation.")
        return 0.0, 0.0

    all_outputs_cat = np.concatenate(all_outputs)
    all_targets_cat = np.concatenate(all_targets)

    auc = calculate_auc(all_targets_cat, all_outputs_cat)
    avg_loss = np.mean(losses)

    return avg_loss, auc

def run_training(df, config, trial=None, all_spectrograms=None):
    """Runs the training loop. 
    
    Accepts pre-loaded spectrograms via the all_spectrograms argument.

    If trial is provided (from Optuna), runs only the single fold specified 
    in config.selected_folds, enables pruning, and returns the best validation 
    AUC for that fold.
    
    If trial is None, runs the folds specified in config.selected_folds 
    (can be multiple) without pruning and returns the mean OOF AUC.
    """
    is_hpo_trial = trial is not None
    
    # --- wandb initialization ---
    run_name = f"trial_{trial.number}" if is_hpo_trial else f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    group_name = "hpo" if is_hpo_trial else "full_training"
    
    wandb_run = wandb.init(
        project="BirdCLEF-2025", # Or your preferred project name
        config=config_to_dict(config),
        name=run_name,
        group=group_name,
        job_type="train",
        reinit=True # Allows re-initialization if running multiple times in one script (e.g. HPO)
    )
    # Log specific HPO trial parameters if applicable
    if is_hpo_trial:
        wandb.config.update(trial.params, allow_val_change=True)

    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")
    print(f"Using Seed: {config.seed}")
    print(f"Load Preprocessed Data: {config.LOAD_PREPROCESSED_DATA}")

    if all_spectrograms is not None:
        print(f"run_training received {len(all_spectrograms)} pre-loaded samples.")
    else:
        print("Warning: run_training received no pre-loaded samples")

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

    if 'samplename' not in working_df.columns:
        print("CRITICAL ERROR: 'samplename' column missing from DataFrame passed to run_training. Exiting.")
        sys.exit(1)

    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
   
    all_folds_history = []
    single_fold_best_auc = 0.0 
    
    try: # Wrap the main training loop in try/finally for wandb.finish()
        for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
            if fold not in config.selected_folds: continue

            print(f'\n{"="*30} Fold {fold} {"="*30}')
            # --- Initialize history for the CURRENT fold --- #
            fold_history = {
                'epochs': [],
                'train_loss': [], 'val_loss': [],
                'train_auc': [], 'val_auc': []
            }

            train_df_fold = working_df.iloc[train_idx].reset_index(drop=True)
            val_df_fold = working_df.iloc[val_idx].reset_index(drop=True)

            # --- Filter Validation Set: Ensure only 'main' data source is used --- 
            original_val_count = len(val_df_fold)
            if 'data_source' in val_df_fold.columns:
                val_df_fold = val_df_fold[val_df_fold['data_source'] == 'main'].reset_index(drop=True)
                print(f"Filtered validation set to include only 'main' data source.")
                print(f"  Original val count: {original_val_count}, Filtered val count: {len(val_df_fold)}")
            else:
                print("Warning: 'data_source' column not found in validation fold. Cannot filter.")

            print(f'Training set: {len(train_df_fold)} samples (includes main and potentially pseudo)')
            print(f'Validation set: {len(val_df_fold)} samples (main data only)')

            # Pass the pre-loaded dictionary (or None) to the Dataset
            train_dataset = BirdCLEFDataset(train_df_fold, config, mode='train', all_spectrograms=all_spectrograms)
            val_dataset = BirdCLEFDataset(val_df_fold, config, mode='valid', all_spectrograms=all_spectrograms)

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                shuffle=True,
                num_workers=config.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.val_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False
            )

            print("\nSetting up model, optimizer, criterion, scheduler...")
            model = BirdCLEFModel(config).to(config.device)
            optimizer = get_optimizer(model, config)
            criterion = get_criterion(config)
            scheduler = get_scheduler(optimizer, config)

            scaler = torch.amp.GradScaler(device='cuda', enabled=config.use_amp)
            print(f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}")


            best_val_auc = 0.0
            best_epoch = 0

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
                    scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
                )

                val_loss, val_auc = validate(
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

                # --- HPO Pruning --- #
                if is_hpo_trial:
                    trial.report(val_auc, epoch) # Report intermediate val_auc
                    if trial.should_prune():
                        print(f"  Pruning trial based on intermediate value at epoch {epoch+1}.")
                        raise optuna.TrialPruned() # Raise exception to stop training
                
                # --- Model Checkpointing ---
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    print(f"✨ New best AUC: {best_val_auc:.4f} at epoch {best_epoch}. Saving model...")

                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'epoch': epoch,
                        'val_auc': best_val_auc,
                        'train_auc': train_auc,
                    }
                    save_path = os.path.join(config.MODEL_OUTPUT_DIR, f"{config.model_name}_fold{fold}_best.pth")
                    try:
                        torch.save(checkpoint, save_path)
                        print(f"  Model saved to {save_path}")

                        # --- wandb log model artifact ---
                        artifact_name = f"{config.model_name}_fold{fold}_best"
                        model_artifact = wandb.Artifact(
                            artifact_name,
                            type="model",
                            description=f"Best model for fold {fold} based on validation AUC.",
                            metadata={
                                "fold": fold,
                                "epoch": best_epoch,
                                "val_auc": best_val_auc,
                                "train_auc": train_auc,
                                "model_name": config.model_name,
                                "seed": config.seed
                            }
                        )
                        model_artifact.add_file(save_path)
                        wandb_run.log_artifact(model_artifact)
                        print(f"  Logged model artifact {artifact_name} to wandb.")

                    except Exception as e:
                        print(f"  Error saving model checkpoint or logging artifact: {e}")

                print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                # Log best AUC for this fold to wandb summary for easy viewing
                wandb.summary[f'fold_{fold}_best_val_auc'] = best_val_auc
                wandb.summary[f'fold_{fold}_best_epoch'] = best_epoch
                single_fold_best_auc = best_val_auc 

                all_folds_history.append(fold_history)

                del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
                del train_df_fold, val_df_fold 
                torch.cuda.empty_cache()
                gc.collect()

            print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
            single_fold_best_auc = best_val_auc 

            all_folds_history.append(fold_history)

        if not is_hpo_trial:
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

                plot_dir = os.path.join(config.OUTPUT_DIR, "training_curves")
                os.makedirs(plot_dir, exist_ok=True)
                plot_save_path = os.path.join(plot_dir, "all_folds_training_plot.png")

                try:
                    plt.savefig(plot_save_path)
                    print(f"Saved average plot to: {plot_save_path}")
                    # --- wandb log training plot ---
                    wandb.log({"training_plot": wandb.Image(plot_save_path)})
                except Exception as e:
                    print(f"Error saving average plot to {plot_save_path}: {e}")
                plt.close(fig)
            else:
                print("\nNo fold histories recorded, skipping average plot generation.")
            
            # --- Non-HPO Summary --- #
            if all_folds_history:
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
                oof_scores_from_hist = [max(h['val_auc']) for h in all_folds_history if h['val_auc']] 
                mean_oof_auc_final = np.mean(oof_scores_from_hist) if oof_scores_from_hist else 0.0
                # Ensure summary is updated if it wasn't before (e.g. if only one fold ran and it wasn't HPO)
                if 'mean_oof_auc' not in wandb.summary:
                     wandb.summary['mean_oof_auc'] = mean_oof_auc_final
                print("\n" + "="*60)
                print("Final Cross-Validation Training Summary (already printed during training):")
                for i, fold_hist in enumerate(all_folds_history):
                     fold_num = config.selected_folds[i]
                     best_fold_auc = max(fold_hist['val_auc']) if fold_hist['val_auc'] else 0.0
                     print(f"  Fold {fold_num}: Best Val AUC = {best_fold_auc:.4f}")
                print(f"\nMean OOF AUC across {len(oof_scores_from_hist)} trained folds: {mean_oof_auc_final:.4f}")
                print("="*60)
                return mean_oof_auc_final
            else:
                print("No folds were trained.")
                print("="*60)
                if 'mean_oof_auc' not in wandb.summary:
                     wandb.summary['mean_oof_auc'] = 0.0
                return 0.0
    finally:
        if wandb_run:
            wandb_run.finish() # Ensure wandb run is finished

if __name__ == "__main__":
    print("\n--- Initializing Training Script ---")
    print(f"Using configuration: LOAD_PREPROCESSED_DATA={config.LOAD_PREPROCESSED_DATA}, USE_PSEUDO_LABELS={config.USE_PSEUDO_LABELS}")

    all_spectrograms = None 

    print("Loading main training metadata...")
    try:
        main_train_df_full = pd.read_csv(config.train_csv_path)
        main_train_df_full['filepath'] = main_train_df_full['filename'].apply(lambda f: os.path.join(config.train_audio_dir, f))
        main_train_df_full['samplename'] = main_train_df_full.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        main_train_df_full['data_source'] = 'main'
        
        # Select only necessary columns for training
        required_cols_main = ['samplename', 'primary_label', 'secondary_labels', 'filepath', 'filename', 'data_source']
        main_train_df = main_train_df_full[required_cols_main].copy()
        print(f"Loaded and selected columns for {len(main_train_df)} main training samples.")
        del main_train_df_full; gc.collect()
    except Exception as e:
        print(f"CRITICAL ERROR loading main training CSV {config.train_csv_path}: {e}. Exiting.")
        sys.exit(1)

    training_df = main_train_df 

    # --- Load Preprocessed Spectrograms (if configured) --- #
    if config.LOAD_PREPROCESSED_DATA:
        all_spectrograms = {} # Initialize as empty dict if loading
        
        # Load PRIMARY spectrograms
        primary_npz_path = config.PREPROCESSED_NPZ_PATH
        print(f"Attempting to load primary spectrograms from: {primary_npz_path}")
        if os.path.exists(primary_npz_path):
            try:
                start_load_time = time.time()
                with np.load(primary_npz_path) as data_archive:
                    primary_specs = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Primary Specs")}
                end_load_time = time.time()
                all_spectrograms.update(primary_specs)
                print(f"Successfully loaded {len(primary_specs)} primary samples in {end_load_time - start_load_time:.2f} seconds.")
                del primary_specs; gc.collect()
            except Exception as e:
                print(f"ERROR loading primary NPZ file {primary_npz_path}: {e}")
                # Decide if this is critical - maybe continue without preloaded?
                print("Cannot continue without primary preloaded data. Exiting.")
                sys.exit(1)
        else:
            print(f"ERROR: Primary NPZ file {primary_npz_path} not found, but LOAD_PREPROCESSED_DATA is True. Exiting.")
            sys.exit(1)

        # --- Conditionally Load PSEUDO-LABEL Data & Spectrograms --- #
        if config.USE_PSEUDO_LABELS:
            print("\n--- Loading Pseudo-Label Data (USE_PSEUDO_LABELS=True) ---")
            
            # Load pseudo metadata
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

                    # Load pseudo spectrograms (check path exists first)
                    pseudo_npz_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, 'pseudo_spectrograms.npz')
                    print(f"Attempting to load pseudo spectrograms from: {pseudo_npz_path}")
                    if os.path.exists(pseudo_npz_path):
                        try:
                            start_load_time = time.time()
                            with np.load(pseudo_npz_path) as data_archive:
                                pseudo_specs = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Pseudo Specs")}
                            end_load_time = time.time()
                            all_spectrograms.update(pseudo_specs) # Merge into the main dictionary
                            print(f"Successfully loaded and merged {len(pseudo_specs)} pseudo samples in {end_load_time - start_load_time:.2f} seconds.")
                            del pseudo_specs; gc.collect()
                            
                            # Concatenate DataFrames only AFTER successfully loading specs
                            training_df = pd.concat([training_df, pseudo_labels_df], ignore_index=True)
                            print(f"Combined DataFrame size: {len(training_df)} samples.")
                            
                        except Exception as e:
                             print(f"ERROR loading pseudo NPZ file {pseudo_npz_path}: {e}")
                             print("Continuing training without pseudo-labels due to NPZ loading error.")
                    else:
                        # Critical Error if NPZ is missing but was expected
                        print(f"CRITICAL ERROR: Pseudo NPZ file {pseudo_npz_path} not found, but USE_PSEUDO_LABELS is True. Exiting.")
                        sys.exit(1)
                else:
                    print("Pseudo labels CSV found but is empty. Skipping.")

            except FileNotFoundError:
                 # Critical Error if CSV is missing but was expected
                print(f"CRITICAL ERROR: Pseudo labels CSV {config.train_pseudo_csv_path} not found, but USE_PSEUDO_LABELS is True. Exiting.")
                sys.exit(1)
            except Exception as e:
                print(f"CRITICAL ERROR loading or processing pseudo labels CSV {config.train_pseudo_csv_path}: {e}")
                print("Exiting due to pseudo-label loading error.")
                sys.exit(1)
        else:
            print("\nSkipping pseudo-label data (USE_PSEUDO_LABELS=False).")

    # Final check on combined dataframe and spectrograms
    print(f"\nFinal training dataframe size: {len(training_df)} samples.")
    if all_spectrograms is not None:
         print(f"Total pre-loaded spectrogram keys available: {len(all_spectrograms)}")
         # Optional: Check for missing keys between final df and loaded specs
         missing_keys = set(training_df['samplename']) - set(all_spectrograms.keys())
         if missing_keys:
              print(f"  WARNING: {len(missing_keys)} samplenames in the final dataframe are missing from the loaded spectrograms!")

    # --- Filter training_df to only include samples with loaded spectrograms --- #
    if all_spectrograms is not None:
        print("\nFiltering training dataframe based on loaded spectrogram keys...")
        original_count = len(training_df)
        loaded_keys = set(all_spectrograms.keys())
        training_df = training_df[training_df['samplename'].isin(loaded_keys)].reset_index(drop=True)
        filtered_count = len(training_df)
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"  Removed {removed_count} samples from training_df because their spectrograms were not found in the loaded NPZ file(s).")
        print(f"  Final training_df size after filtering: {filtered_count} samples.")
    else:
        print("\nWarning: all_spectrograms is None, cannot filter training_df by loaded keys.")

    # --- Run Training --- #
    run_training(training_df, config, all_spectrograms=all_spectrograms)

    print("\nTraining script finished!")