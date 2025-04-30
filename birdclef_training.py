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

from tqdm.auto import tqdm

import timm
import matplotlib.pyplot as plt
import optuna

from config import config
import birdclef_utils as utils
from google.cloud import storage

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

gcs_client = None
if config.IS_CUSTOM_JOB:
    try:
        gcs_client = storage.Client()
        print("INFO: Google Cloud Storage client initialized successfully.")
    except Exception as e:
        print(f"CRITICAL WARNING: Failed to initialize Google Cloud Storage client: {e}")
        # Training will likely fail if client is needed and not initialized.
# --- End GCS Client Init --- #

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
        spec = None

        # --- Load preprocessed data --- #
        if self.all_spectrograms is None:
            print(f"CRITICAL ERROR: all_spectrograms is None in __getitem__ for {samplename}! Cannot proceed.")
            raise ValueError("BirdCLEFDataset received None for all_spectrograms, but preprocessed data is required.")
            
        if samplename in self.all_spectrograms:
            spec_data = self.all_spectrograms[samplename] # Can be (3, 256, 256) or (256, 256)
            selected_spec_np = None # To hold the final (256, 256) numpy array
            expected_shape_2d = tuple(self.config.TARGET_SHAPE)

            # --- Handle based on retrieved data type/shape --- 
            if isinstance(spec_data, np.ndarray):
                # Case 1: Primary Data (Multiple Chunks stacked) - Shape (3, 256, 256)
                if spec_data.ndim == 3 and spec_data.shape[1:] == expected_shape_2d:
                    if self.mode == 'train':
                        # Randomly select one chunk (np array iterates over first dim)
                        selected_spec_np = random.choice(spec_data) 
                    else:
                        # Validation/Test: Take the first chunk
                        selected_spec_np = spec_data[0]
                    
                # Case 2: Pseudo Data (Single Chunk) - Shape (256, 256)
                elif spec_data.ndim == 2 and spec_data.shape == expected_shape_2d:
                    selected_spec_np = spec_data
                
                # Case 3: Unexpected ndarray shape
                else:
                    print(f"WARNING: Ndarray '{samplename}' has unexpected shape {spec_data.shape}. Attempting reshape or using zeros.")
                    try:
                        selected_spec_np = spec_data.reshape(expected_shape_2d)
                        print(f"  Successfully reshaped '{samplename}' to {selected_spec_np.shape}")
                    except:
                        print(f"  Reshape failed for '{samplename}'. Using zeros.")

            if selected_spec_np is not None and selected_spec_np.shape == expected_shape_2d:
                spec = selected_spec_np
            else:
                spec = np.zeros(expected_shape_2d, dtype=np.float32)
                shape_info = selected_spec_np.shape if isinstance(selected_spec_np, np.ndarray) else type(selected_spec_np)
                print(f"Fallback: Using zeros for '{samplename}' due to processing issues (Final shape before fallback: {shape_info}).")
        else:
            print(f"ERROR: Samplename '{samplename}' not found in the pre-loaded spectrogram dictionary! Using zeros.")
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)

        # --- Final Shape Guarantee --- 
        expected_shape = tuple(self.config.TARGET_SHAPE)
        if not isinstance(spec, np.ndarray) or spec.shape != expected_shape:
             print(f"CRITICAL WARNING: Final spec for '{samplename}' has wrong shape/type ({spec.shape if isinstance(spec, np.ndarray) else type(spec)}) before unsqueeze. Forcing zeros.")
             spec = np.zeros(expected_shape, dtype=np.float32)

        # Ensure spec is float32 before converting to tensor
        spec = spec.astype(np.float32)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train":
            spec = self.apply_spec_augmentations(spec)

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
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': filename_for_error 
        }

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram."""
        # Time masking (horizontal stripes)
        if random.random() < self.config.time_mask_prob:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, self.config.max_time_mask_width) 
                start = random.randint(0, max(0, spec.shape[2] - width))
                spec[0, :, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < self.config.freq_mask_prob: 
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, self.config.max_freq_mask_height) 
                start = random.randint(0, max(0, spec.shape[1] - height))
                spec[0, start:start+height, :] = 0

        # Random brightness/contrast
        if random.random() < self.config.contrast_prob:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 

        return spec

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
        # Return None for empty batch, handled in train/val loops
        print("Warning: collate_fn received empty batch, returning None.")
        return None

    # Check for consistent keys before trying to access batch[0]
    expected_keys = {'melspec', 'target', 'filename'}
    if not all(expected_keys.issubset(item.keys()) for item in batch):
         print("Warning: Batch items have inconsistent keys. Returning None.")
         return None

    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    # Stack tensors
    try:
        result['target'] = torch.stack(result['target'])
        # Assuming melspec should always be stackable due to TARGET_SHAPE
        result['melspec'] = torch.stack(result['melspec'])
    except RuntimeError as e:
        # This might happen if melspecs somehow have different shapes despite TARGET_SHAPE
        print(f"Error stacking tensors in collate_fn: {e}. Returning None.")
        return None
    except Exception as e:
        print(f"Unexpected error in collate_fn: {e}. Returning None.")
        return None

    # Check batch size consistency after stacking
    if result['melspec'].shape[0] != len(batch) or result['target'].shape[0] != len(batch):
        print("Warning: Collated tensors have incorrect batch dimension. Returning None.")
        return None

    return result

class BirdCLEFModel(nn.Module):
    """BirdCLEF model using timm backbone."""
    def __init__(self, config): # Pass config object
        super().__init__()
        self.config = config

        # Ensure num_classes is available
        if not hasattr(config, 'num_classes') or config.num_classes <= 0:
             raise ValueError("config.num_classes must be set to a positive integer.")

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
        elif 'resnet' in config.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # Assuming this path handles other timm models with get_classifier
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, config.num_classes)

        # Mixup setup - use config directly
        self.mixup_enabled = hasattr(config, 'mixup_alpha') and config.mixup_alpha > 0
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

# --- Helper Functions --- #

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

def get_optimizer(model, config): # Pass config object
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
            momentum=0.9, # Common default
            weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer '{config.optimizer}' not implemented")
    return optimizer

def get_scheduler(optimizer, config, steps_per_epoch=None): # Pass config, potentially steps_per_epoch
    """Creates learning rate scheduler based on config settings."""
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs, # T_max in epochs
            eta_min=config.min_lr
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', # Use min mode as per user's version
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
    elif config.scheduler == 'OneCycleLR':
        # Use original implementation detail (scheduler=None handled in run_training)
        scheduler = None # Let run_training handle the specific OneCycleLR setup
    elif config.scheduler is None or config.scheduler.lower() == 'none':
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler '{config.scheduler}' not implemented")
    return scheduler

def get_criterion(config): # Pass config object
    """Creates loss criterion based on config settings."""
    if config.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config.criterion == 'FocalLossBCE':
        # Uses hardcoded defaults in losses.py (alpha=0.25, gamma=2, bce_w=0.6, focal_w=1.4) 
        # as per user preference
        print("INFO: Using FocalLossBCE with parameters hardcoded in losses.py")
        criterion = FocalLossBCE(config=config)
    else:
        raise NotImplementedError(f"Criterion '{config.criterion}' not implemented")
    return criterion

def calculate_auc(targets, outputs):
    """Calculates macro-averaged ROC AUC (User's version)."""
    num_classes = targets.shape[1]
    aucs = []

    # Use user's calculation logic
    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except ValueError as e:
                pass

    return np.mean(aucs) if aucs else 0.0

# --- Training and Validation Loops (Using User's versions, simplified batch handling) --- #

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler=None):
    """Runs one epoch of training with optional mixed precision."""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    use_amp = scaler.is_enabled() # Check if AMP is active via the scaler
    mixup_active = model.mixup_enabled # Check if model has mixup enabled

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

        # --- AMP: autocast context --- #
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Apply mixup BEFORE model if enabled
            if mixup_active:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs_orig, targets_orig, model.mixup_alpha, device
                )
            else:
                inputs = inputs_orig
            
            # Model only returns logits now
            logits = model(inputs)

            # Calculate loss AFTER model, potentially using mixup targets
            if mixup_active:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets_orig)
        # --- End autocast --- #

        # --- AMP: Scale loss and step --- #
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # --- End AMP --- #

        # Use original targets for AUC calculation
        outputs_np = logits.detach().float().cpu().numpy() # Ensure float32 for numpy ops
        targets_np = targets_orig.detach().cpu().numpy()

        # Scheduler step (only for OneCycleLR here, matching original)
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs_np)
        all_targets.append(targets_np)
        losses.append(loss.item())

        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

        # --- Debugging: Limit batches --- #
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
    use_amp = config.use_amp # Check config directly for validation

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

            # --- AMP: autocast context for validation --- #
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            # --- End autocast --- #

            all_outputs.append(outputs.float().cpu().numpy()) # Ensure float32 for numpy ops
            all_targets.append(targets.cpu().numpy())
            losses.append(loss.item())

            # --- Debugging: Limit batches --- #
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
    if is_hpo_trial:
        print("\n--- Starting HPO Training Trial --- (Pruning Enabled)")
    else:
        print("\n--- Starting Standard Training Run --- (No Pruning)")
        
    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")
    print(f"Using Seed: {config.seed}")
    print(f"Load Preprocessed Data: {config.LOAD_PREPROCESSED_DATA}")

    # --- NPZ Data Loading moved to the main script block --- #
    # Removed the NPZ loading logic from here. 
    # `all_spectrograms` is now expected as an argument.
    if all_spectrograms is not None:
        print(f"run_training received {len(all_spectrograms)} pre-loaded samples.")
    elif config.LOAD_PREPROCESSED_DATA:
        print("Warning: run_training received no pre-loaded samples, but LOAD_PREPROCESSED_DATA is True.")
    else:
        print("run_training configured for on-the-fly generation (all_spectrograms is None).")
    # --- End NPZ Data Loading Removal --- #

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    working_df = df.copy()

    # --- Samplename generation moved to the main script block --- #
    # Removed samplename generation logic
    if 'samplename' not in working_df.columns:
        print("CRITICAL ERROR: 'samplename' column missing from DataFrame passed to run_training. Exiting.")
        sys.exit(1)
    # --- End Samplename Generation Removal --- #

    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
   
    all_folds_history = []
    single_fold_best_auc = 0.0 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
        if fold not in config.selected_folds: continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        # --- Initialize history for the CURRENT fold --- #
        fold_history = {
            'epochs': [],
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': []
        }
        # --- End History Init --- #

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
        # --- End Validation Set Filtering --- 

        print(f'Training set: {len(train_df_fold)} samples (includes main and potentially pseudo)')
        print(f'Validation set: {len(val_df_fold)} samples (main data only)')

        # Pass the pre-loaded dictionary (or None) to the Dataset
        train_dataset = BirdCLEFDataset(train_df_fold, config, mode='train', all_spectrograms=all_spectrograms)
        val_dataset = BirdCLEFDataset(val_df_fold, config, mode='valid', all_spectrograms=all_spectrograms)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers, # Can still use workers for CPU tasks
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
            # REMOVED worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers, # Can still use workers for CPU tasks
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
            # REMOVED worker_init_fn
        )

        print("\nSetting up model, optimizer, criterion, scheduler...")
        model = BirdCLEFModel(config).to(config.device)
        optimizer = get_optimizer(model, config)
        criterion = get_criterion(config)
        scheduler = get_scheduler(optimizer, config)

        # --- AMP: Initialize GradScaler --- #
        scaler = torch.amp.GradScaler(device='cuda', enabled=config.use_amp)
        print(f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}")
        # --- End AMP Init --- #

        best_val_auc = 0.0
        best_epoch = 0
        # Removed patience counter / early stopping for simplicity, can be added back

        # --- Epoch Loop --- #
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch + 1}/{config.epochs}")

            # Pass scaler to train_one_epoch
            train_loss, train_auc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.device,
                scaler, # <-- Pass scaler
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            # Validation uses autocast internally based on config
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
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val AUC:   {val_auc:.4f}")

            # --- Append metrics for the CURRENT fold's history --- #
            fold_history['epochs'].append(epoch + 1)
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['train_auc'].append(train_auc)
            fold_history['val_auc'].append(val_auc)
            # --- End Appending ---

            # --- HPO Pruning --- #
            if is_hpo_trial:
                trial.report(val_auc, epoch) # Report intermediate val_auc
                if trial.should_prune():
                    print(f"  Pruning trial based on intermediate value at epoch {epoch+1}.")
                    raise optuna.TrialPruned() # Raise exception to stop training
            # --- End HPO Pruning ---
            
            # --- Model Checkpointing ---
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                print(f"  ✨ New best AUC: {best_val_auc:.4f} at epoch {best_epoch}. Saving model...")

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
                except Exception as e:
                    print(f"  Error saving model checkpoint: {e}")

        print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
        # Store the best AUC for this fold (will be the only one if is_hpo_trial)
        single_fold_best_auc = best_val_auc 

        # --- Store history for this fold (only relevant if not HPO or if analyzing HPO runs later) --- #
        all_folds_history.append(fold_history)
        # --- End Store History --- #

        del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
        del train_df_fold, val_df_fold 
        torch.cuda.empty_cache()
        gc.collect()

    # --- Plotting & Summary (Only for Non-HPO Runs) --- #
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
                    epoch_idx = i # 0-based index for arrays
                    if epoch_idx < num_epochs: # Safety check
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
            os.makedirs(plot_dir, exist_ok=True) # Ensure the directory exists
            plot_save_path = os.path.join(plot_dir, "all_folds_training_plot.png")

            try:
                plt.savefig(plot_save_path)
                print(f"Saved average plot to: {plot_save_path}")
            except Exception as e:
                print(f"Error saving average plot to {plot_save_path}: {e}")
            plt.close(fig) # Close the figure to free memory
        else:
            print("\nNo fold histories recorded, skipping average plot generation.")
        # --- End Plotting --- #
        
        # --- Non-HPO Summary --- #
        if all_folds_history:
            oof_scores_from_hist = [max(h['val_auc']) for h in all_folds_history if h['val_auc']] # Get best AUC from each fold history
            mean_oof_auc = np.mean(oof_scores_from_hist) if oof_scores_from_hist else 0.0
            print("\n" + "="*60)
            print("Cross-Validation Training Summary:")
            # Use actual selected folds list length for iteration, assuming it matches history order for non-hpo
            num_folds_run = len(all_folds_history) 
            for i in range(num_folds_run):
                 fold_num = config.selected_folds[i] # Index based on number of folds actually run
                 best_fold_auc = max(all_folds_history[i]['val_auc']) if all_folds_history[i]['val_auc'] else 0.0
                 print(f"  Fold {fold_num}: Best Val AUC = {best_fold_auc:.4f}")
            print(f"\nMean OOF AUC across {len(oof_scores_from_hist)} trained folds: {mean_oof_auc:.4f}")
            print("="*60)
        else:
            print("\nNo folds were trained.")
            print("="*60)
        # --- End Non-HPO Summary ---

    # --- Final Return Value --- #
    if is_hpo_trial:
        # For HPO, return the best AUC achieved in the single fold run
        print(f"\nReturning best AUC for HPO Trial (Fold {config.selected_folds[0]}): {single_fold_best_auc:.4f}")
        return single_fold_best_auc
    else:
        # For standard runs, calculate and return the mean OOF AUC if multiple folds ran
        if all_folds_history:
            # Recalculate mean OOF from the best scores recorded in histories if needed
            oof_scores_from_hist = [max(h['val_auc']) for h in all_folds_history if h['val_auc']] # Get best AUC from each fold history
            mean_oof_auc = np.mean(oof_scores_from_hist) if oof_scores_from_hist else 0.0
            print("\n" + "="*60)
            print("Cross-Validation Training Summary:")
            for i, fold_hist in enumerate(all_folds_history):
                 fold_num = config.selected_folds[i] # Assumes selected_folds matches history order
                 best_fold_auc = max(fold_hist['val_auc']) if fold_hist['val_auc'] else 0.0
                 print(f"  Fold {fold_num}: Best Val AUC = {best_fold_auc:.4f}")
            print(f"\nMean OOF AUC across {len(oof_scores_from_hist)} trained folds: {mean_oof_auc:.4f}")
            print("="*60)
            return mean_oof_auc
        else:
            print("No folds were trained.")
            print("="*60)
            return 0.0

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
                    
                    # Select only necessary columns for training (note: secondary_labels will be NaN)
                    required_cols_pseudo = ['samplename', 'primary_label', 'filepath', 'filename', 'data_source'] # Include data_source
                    pseudo_labels_df = pseudo_labels_df_full[required_cols_pseudo].copy()
                    print(f"Loaded and selected columns for {len(pseudo_labels_df)} pseudo labels.")
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
              print(f"    Examples: {list(missing_keys)[:10]}")
              # Potentially filter df: training_df = training_df[training_df['samplename'].isin(all_spectrograms.keys())]

    # --- Run Training --- #
    run_training(training_df, config, all_spectrograms=all_spectrograms)

    print("\nTraining script finished!")