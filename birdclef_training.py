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

from tqdm.auto import tqdm

import timm
import matplotlib.pyplot as plt

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
        self.all_spectrograms = all_spectrograms # Store the pre-loaded dictionary

        # Load taxonomy info once
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

        # Ensure necessary columns exist
        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.df['filename'].apply(lambda f: os.path.join(self.config.train_audio_dir, f))
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        # Print status based on whether spectrograms were pre-loaded
        if self.all_spectrograms is not None:
            print(f"Dataset mode '{self.mode}': Using pre-loaded spectrogram dictionary with {len(self.all_spectrograms)} samples.")
        elif self.config.LOAD_PREPROCESSED_DATA:
            print(f"Dataset mode '{self.mode}': WARNING - Configured to load preprocessed data, but no dictionary provided. Will attempt on-the-fly.")
        else:
            print(f"Dataset mode '{self.mode}': Configured to generate spectrograms on-the-fly.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        # 1. Try loading from the pre-loaded dictionary
        if self.all_spectrograms is not None:
            if samplename in self.all_spectrograms:
                spec = self.all_spectrograms[samplename]
            else:
                # If using pre-loaded, missing sample is an issue
                print(f"ERROR: Sample '{samplename}' not found in the pre-loaded spectrogram dictionary! Using zeros.")
                # Fallback to zeros, or could raise an error
                spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)

        # 2. Attempt on-the-fly generation ONLY if not using pre-loaded data
        #    (or if pre-loading failed and all_spectrograms is None, handled in __init__ print)
        if spec is None and not self.config.LOAD_PREPROCESSED_DATA:
            if self.mode == "train":
                print(f"Debug: Generating spec on-the-fly for {samplename}")
            try:
                spec = utils.process_audio_file(row['filepath'], row['filename'], self.config, {}, {})
            except Exception as e_gen:
                 print(f"Error generating spectrogram on-the-fly for {samplename}: {e_gen}")
                 spec = None

        # 3. Handle cases where spec is still None (generation failed or pre-load error)
        if spec is None:
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Failed to load or generate spec for {samplename}. Using zeros.")

        # Ensure spec is float32 before converting to tensor
        spec = spec.astype(np.float32)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply augmentations only in training mode
        if self.mode == "train" and random.random() < self.config.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        # Encode primary and secondary labels
        target = self.encode_label(row['primary_label'])
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                try:
                    secondary_labels = eval(row['secondary_labels'])
                    if not isinstance(secondary_labels, list): secondary_labels = []
                except: secondary_labels = []
            else: secondary_labels = row['secondary_labels']
            if isinstance(secondary_labels, list):
                 for label in secondary_labels:
                     if label in self.label_to_idx: target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram (User's version)."""
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            # Using fixed width from user version
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, max(0, spec.shape[2] - width))
                spec[0, :, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            # Using fixed height from user version
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, max(0, spec.shape[1] - height))
                spec[0, start:start+height, :] = 0

        # Random brightness/contrast (User's version)
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def encode_label(self, label):
        """Encode primary label to one-hot vector."""
        target = np.zeros(self.num_classes, dtype=np.float32)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    """Custom collate function (User's version)."""
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

    def forward(self, x, targets=None):
        # Use original forward logic
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits,
                                       logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup augmentation (User's version)."""
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function (User's version)."""
        # Assuming criterion is like BCEWithLogitsLoss which handles reduction internally
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Helper Functions --- #

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
                # This can happen if only one class is present in targets for this class
                # print(f"Warning: AUC calculation error for class {i}: {e}")
                # Don't append anything if calculation fails
                pass # Suppress warning for cleaner logs

    return np.mean(aucs) if aucs else 0.0

# --- Training and Validation Loops (Using User's versions, simplified batch handling) --- #

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler=None):
    """Runs one epoch of training with optional mixed precision."""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    use_amp = scaler.is_enabled() # Check if AMP is active via the scaler

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if batch is None:
            print(f"Warning: Skipping None batch at step {step}")
            continue

        try:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
        except (AttributeError, TypeError) as e:
            print(f"Error: Skipping batch {step} due to unexpected format: {e}")
            continue

        optimizer.zero_grad()

        # --- AMP: autocast context --- #
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Pass targets only if mixup is enabled
            outputs = model(inputs, targets if model.training and model.mixup_enabled else None)

            # Handle model output (logits or logits, loss)
            if isinstance(outputs, tuple):
                logits, loss = outputs # Assume loss is already calculated correctly (e.g., with mixup)
            else:
                logits = outputs
                loss = criterion(logits, targets)
        # --- End autocast --- #

        # --- AMP: Scale loss and step --- #
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # --- End AMP --- #

        outputs_np = logits.detach().float().cpu().numpy() # Ensure float32 for numpy ops
        targets_np = targets.detach().cpu().numpy()

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

def run_training(df, config, resume_fold=0):
    """Runs the cross-validation training loop."""
    print("\n--- Starting Training Run ---")
    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")
    print(f"Load Preprocessed Data: {config.LOAD_PREPROCESSED_DATA}")

    all_spectrograms = None # Initialize
    # --- Pre-load NPZ if configured --- #
    if config.LOAD_PREPROCESSED_DATA:
        npz_path = config.PREPROCESSED_NPZ_PATH
        print(f"Attempting to pre-load NPZ file into RAM: {npz_path}")
        if not os.path.exists(npz_path):
             print(f"Error: LOAD_PREPROCESSED_DATA is True, but NPZ file {npz_path} does not exist.")
             print("       Please run preprocessing.py first.")
             sys.exit(1) # Exit if preprocessed data is required but missing
        else:
            try:
                print("Loading... (This might take a moment for large files)")
                start_load_time = time.time()
                # Load the entire NPZ into a dictionary in RAM
                # Note: np.load returns a lazy NpzFile object, explicitly convert to dict
                with np.load(npz_path) as data_archive:
                    all_spectrograms = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading NPZ into RAM")}
                end_load_time = time.time()
                print(f"Successfully pre-loaded {len(all_spectrograms)} samples into RAM in {end_load_time - start_load_time:.2f} seconds.")
            except Exception as e:
                print(f"Error loading NPZ file into RAM: {e}")
                print("Cannot continue without preloaded data when LOAD_PREPROCESSED_DATA is True.")
                sys.exit(1)
    else:
        print("\nConfigured to generate spectrograms on-the-fly (no pre-loading).")
    # --- End Pre-loading --- #

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    working_df = df.copy()

    if 'filepath' not in working_df.columns:
        working_df['filepath'] = working_df['filename'].apply(lambda f: os.path.join(config.train_audio_dir, f))
    if 'samplename' not in working_df.columns:
        working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    oof_scores = []
    # --- Modify History Tracking --- #
    all_folds_history = [] # Store history dict from each fold
    # --- End Modify History Tracking --- #

    for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
        if fold < resume_fold: continue
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

        print(f'Training set: {len(train_df_fold)} samples')
        print(f'Validation set: {len(val_df_fold)} samples')

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
            # --- End Appending --- #

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
        oof_scores.append(best_val_auc)

        # --- Store history for this fold --- #
        all_folds_history.append(fold_history)
        # --- End Store History --- #

        del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
        del train_df_fold, val_df_fold 
        torch.cuda.empty_cache()
        gc.collect()

    # --- Plotting Average Metrics Across Folds --- #
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

    print("\n" + "="*60)
    print("Cross-Validation Training Summary:")
    if oof_scores:
        mean_oof_auc = np.mean(oof_scores)
        for i, score in enumerate(oof_scores):
            fold_num = config.selected_folds[i]
            print(f"  Fold {fold_num}: Best Val AUC = {score:.4f}")
        print(f"\nMean OOF AUC across {len(oof_scores)} trained folds: {mean_oof_auc:.4f}")
    else:
        print("No folds were trained.")
        mean_oof_auc = 0.0 
    print("="*60)

    # Return mean_oof_auc (might be used if called from elsewhere later)
    return mean_oof_auc

if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="BirdCLEF Training Script")
    parser.add_argument(
        '--resume_fold', 
        type=int, 
        default=0, 
        help='Fold number to resume training from (0-indexed). Skips folds < resume_fold.'
    )
    args = parser.parse_args()
    print(f"Resume Fold specified: {args.resume_fold}")
    # --- End Argument Parsing --- #

    print("\n--- Initializing Training Script ---")

    print("Loading main training metadata...")
    try:
        main_train_df = pd.read_csv(config.train_csv_path)
    except FileNotFoundError:
        print(f"Error: Main training CSV not found at {config.train_csv_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading main training CSV: {e}. Exiting.")
        sys.exit(1)

    # Pass the resume_fold argument to run_training
    run_training(main_train_df, config, args.resume_fold) 

    print("\nTraining script finished!")