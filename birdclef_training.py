import os
import logging
import random
import gc
import time
import warnings
from pathlib import Path
import sys
import glob
# import zipfile # No longer needed

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import timm

from config import config
import birdclef_utils as utils
import optuna
import matplotlib.pyplot as plt # Import for plotting

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

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

# set_seed(config.seed) # Removed from global scope

class BirdCLEFDataset(Dataset):
    """Dataset class for BirdCLEF.
    Handles loading pre-computed spectrogram CHUNKS from a directory structure
    based on chunked metadata.
    """
    def __init__(self, df, config, mode="train"):
        self.df = df.copy()
        self.config = config
        self.mode = mode
        # Store the base directory where chunk .npy files are located
        self.chunk_dir_path = config.PREPROCESSED_CHUNK_DIR 
        # self.zip_path = config.PREPROCESSED_ZIP_PATH # Removed zip logic
        # self.zip_ref = None # Removed zip logic

        # Check if chunk directory exists
        if not os.path.isdir(self.chunk_dir_path):
             raise FileNotFoundError(f"Preprocessed chunk directory not found at: {self.chunk_dir_path}")
        print(f"Dataset mode '{self.mode}': Reading chunks from directory {self.chunk_dir_path}")

        # --- Removed zip opening logic --- 

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
            # No zip_ref to close here
            raise

        required_cols = ['chunk_key', 'primary_label', 'secondary_labels', 'filename']
        if not all(col in self.df.columns for col in required_cols):
            # No zip_ref to close here
            raise ValueError(f"Chunked DataFrame missing required columns: {required_cols}")
        
        # Verification of files vs dir could be added here if needed (might be slow)
        # Example: Check if first few chunk_key.npy files exist? 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chunk_key = row['chunk_key'] # e.g., 'label/XC123.ogg_chunk0'
        spec = None
        
        # Construct the full path to the .npy file
        npy_filepath = os.path.join(self.chunk_dir_path, f"{chunk_key}.npy")

        try:
            # Load directly from the .npy file path
            spec = np.load(npy_filepath)
        except FileNotFoundError:
            print(f"Warning: Spectrogram chunk file not found: {npy_filepath}. Using zeros.")
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float16)
        except Exception as e:
            print(f"Error loading chunk file {npy_filepath}: {e}. Using zeros.")
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float16)

        # Load as float16 initially, then cast to float32 for the model
        spec = torch.tensor(spec, dtype=torch.float16) 
        spec = spec.float() # Cast to float32
        spec = spec.unsqueeze(0) # Add channel dimension

        if self.mode == "train" and random.random() < self.config.aug_prob:
            # Ensure augmentations work with float32 or handle casting internally
            spec = self.apply_spec_augmentations(spec) # Now feeding float32 to augmentations
            # --- End Apply Augmentations ---

        target = self.encode_label(row['primary_label'])
        
        # Handle secondary labels (check for NaN before processing)
        secondary_labels_val = row.get('secondary_labels', None)
        if pd.notna(secondary_labels_val) and secondary_labels_val not in [[''], None]:
            if isinstance(secondary_labels_val, str):
                try:
                    secondary_labels = eval(secondary_labels_val)
                    if not isinstance(secondary_labels, list):
                         secondary_labels = []
                except: # Catch potential eval errors
                    secondary_labels = []
            elif isinstance(secondary_labels_val, list):
                secondary_labels = secondary_labels_val
            else:
                secondary_labels = [] # Fallback for unexpected types
            
            # Process list of secondary labels
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0 # Target remains float32

        return {
            'melspec': spec, # This is float16
            'target': torch.tensor(target, dtype=torch.float32), # Target is float32
            'chunk_key': chunk_key
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
    # ... (Check expected keys should now include chunk_key, remove filename if not needed) ...
    expected_keys = {'melspec', 'target', 'chunk_key'}
    if not all(expected_keys.issubset(item.keys()) for item in batch):
         print(f"Warning: Batch items have inconsistent keys (Expected: {expected_keys}). Returning None.")
         return None

    # ... (Rest of collate_fn should be okay, stacking melspec and target) ...

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

def get_criterion(config, pos_weight=None): # Pass config object and optional pos_weight
    """Creates loss criterion based on config settings."""
    if config.criterion == 'BCEWithLogitsLoss':
        # Pass pos_weight tensor if provided
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

def train_one_epoch(model, loader, optimizer, criterion, device, config, scheduler=None):
    """Runs one epoch of training (User's version, simplified)."""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        # Handle None batches from collate_fn
        if batch is None:
            print(f"Warning: Skipping None batch at step {step}")
            continue

        # Assume batch is a dictionary with tensor values
        try:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
        except (AttributeError, TypeError) as e:
            print(f"Error: Skipping batch {step} due to unexpected format: {e}")
            continue

        optimizer.zero_grad()
        # Access mixup_enabled via model.module when using DataParallel
        apply_mixup = model.training and getattr(model, 'module', model).mixup_enabled
        outputs = model(inputs, targets if apply_mixup else None)

        # Handle model output (logits or logits, loss)
        if isinstance(outputs, tuple):
            logits, loss_tensor = outputs
            # When using DataParallel with loss computed inside model.forward,
            # the returned loss might be a tensor with loss per device. Average it.
            loss = loss_tensor.mean()
        else:
            # If mixup is off, calculate loss here (criterion handles reduction)
            logits = outputs
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        outputs_np = logits.detach().cpu().numpy()
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

def validate(model, loader, criterion, device, config):
    """Runs validation (User's version)."""
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

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

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            all_outputs.append(outputs.cpu().numpy())
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

# --- Main Training Orchestration --- #

def run_training(df, config, trial=None):
    """Runs the cross-validation training loop using GroupKFold and returns metrics and history."""
    print("\n--- Starting Training Run with Chunked Data ---")
    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")

    if not config.LOAD_PREPROCESSED_DATA:
         print("Error: LOAD_PREPROCESSED_DATA must be True in config for chunked data.")
         return 0.0, {}
         
    # These checks might need adjustment based on final config logic
    if config.PREPROCESSED_DATA_TYPE == "dir":
        print(f"Training will load chunks individually from directory: {config.PREPROCESSED_CHUNK_DIR}")
        if not os.path.isdir(config.PREPROCESSED_CHUNK_DIR):
             print(f"Error: Preprocessed chunk directory not found at {config.PREPROCESSED_CHUNK_DIR}. Check config or run preprocessing. Exiting.")
             return 0.0, {}
    elif config.PREPROCESSED_DATA_TYPE == "zip":
         print(f"Training will load chunks individually from: {config.PREPROCESSED_ZIP_PATH}")
         if not os.path.exists(config.PREPROCESSED_ZIP_PATH):
             print(f"Error: Preprocessed zip archive not found at {config.PREPROCESSED_ZIP_PATH}. Run preprocessing. Exiting.")
             return 0.0, {}
    else:
         print(f"Error: Invalid config.PREPROCESSED_DATA_TYPE: {config.PREPROCESSED_DATA_TYPE}")
         return 0.0, {}
         
    # The input df IS the chunked dataframe
    working_df = df.copy()

    # --- Calculate Class Weights (pos_weight) --- #
    print("Calculating class weights for BCEWithLogitsLoss...")
    pos_weight_tensor = None
    try:
        # Load the specific metadata file for weight calculation
        weight_meta_path = "/kaggle/input/train-metadata/train_metadata_chunked_full2.csv"
        print(f"Loading metadata for weights from: {weight_meta_path}")
        weight_df = pd.read_csv(weight_meta_path)
        
        # Load taxonomy to get label mapping
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        species_ids = taxonomy_df['primary_label'].tolist()
        num_classes = len(species_ids)
        label_to_idx = {label: idx for idx, label in enumerate(species_ids)}

        # Calculate primary label counts
        primary_counts_series = weight_df['primary_label'].value_counts()
        total_samples = len(weight_df)
        
        # Create positive counts tensor
        positive_counts = torch.zeros(num_classes)
        for label, count in primary_counts_series.items():
            if label in label_to_idx:
                idx = label_to_idx[label]
                positive_counts[idx] = count
            else:
                print(f"Warning: Label '{label}' from weight metadata not found in taxonomy.")

        # Calculate pos_weight: num_neg / num_pos = (total - num_pos) / num_pos
        epsilon = 1e-6 # Prevent division by zero
        pos_weight_tensor = (total_samples - positive_counts) / (positive_counts + epsilon)
        
        # Ensure tensor is on the correct device
        pos_weight_tensor = pos_weight_tensor.to(config.device)
        print(f"Calculated pos_weight tensor (shape: {pos_weight_tensor.shape}, device: {pos_weight_tensor.device})")
        # Optional: Clamp weights
        # pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=100.0)

    except FileNotFoundError:
        print(f"Error: Metadata file for weight calculation not found at {weight_meta_path}. Proceeding without class weights.")
    except Exception as e:
        print(f"Error calculating class weights: {e}. Proceeding without class weights.")
        pos_weight_tensor = None # Ensure it's None if calculation fails

    # --- Ensure Model Output Directory Exists --- #
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # --- Group K-Fold Cross-validation Setup --- 
    # Need a column identifying the original audio file for grouping
    # Assuming 'original_samplename' was created during preprocessing
    group_col = 'filename' # Use filename for grouping now
    if group_col not in working_df.columns:
         print(f"Error: Grouping column '{group_col}' not found in the chunked DataFrame. Cannot use GroupKFold.")
         return 0.0, {}
         
    print(f"Using GroupKFold with {config.n_fold} splits, grouping by '{group_col}'.")
    # Use GroupKFold - does not stratify, ensures groups stay together.
    gkf = GroupKFold(n_splits=config.n_fold)
    oof_scores = []
    # Initialize history storage
    fold_history = {f: {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
                    for f in config.selected_folds}

    # --- Fold Loop --- 
    # Pass the groups parameter to split()
    # Note: y parameter is not used by GroupKFold for splitting but required by API
    for fold, (train_idx, val_idx) in enumerate(gkf.split(working_df, y=working_df['primary_label'], groups=working_df[group_col])):
        if fold not in config.selected_folds:
            print(f"\nSkipping Fold {fold}...")
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')

        # --- Data Setup for Fold --- 
        train_df_fold = working_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = working_df.iloc[val_idx].reset_index(drop=True)

        # Report chunk counts for the fold
        print(f'Training set: {len(train_df_fold)} chunks')
        print(f'Validation set: {len(val_df_fold)} chunks')
        # Optionally report number of unique original files in each set
        print(f'  Unique original files in train: {train_df_fold[group_col].nunique()}')
        print(f'  Unique original files in valid: {val_df_fold[group_col].nunique()}')

        # Create datasets - Now loads directly from zip, no spectrograms dict passed
        train_dataset = BirdCLEFDataset(train_df_fold, config, mode='train')
        val_dataset = BirdCLEFDataset(val_df_fold, config, mode='valid')

        # Create dataloaders (rest of setup is the same)
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

        # --- Model, Optimizer, Criterion, Scheduler Setup for Fold ---
        print("\nSetting up model, optimizer, criterion, scheduler...")
        # Instantiate model on CPU first (if device changes occur later)
        model = BirdCLEFModel(config) 

        # Check if multiple GPUs are available and wrap with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel.")
            model = nn.DataParallel(model) # Wrap the model
        elif torch.cuda.is_available():
             print(f"Using single GPU: {config.device}")
        else:
            print("Using CPU")

        # Move the model (or wrapped model) to the primary CUDA device (or CPU)
        model.to(config.device)

        optimizer = get_optimizer(model, config) # Pass the potentially wrapped model
        # Pass the calculated pos_weight tensor to the criterion function
        criterion = get_criterion(config, pos_weight=pos_weight_tensor).to(config.device)

        # Handle OneCycleLR setup separately as per original logic
        if config.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.lr,
                steps_per_epoch=len(train_loader),
                epochs=config.epochs,
                pct_start=0.1 # Consider making pct_start configurable
            )
        else:
            scheduler = get_scheduler(optimizer, config) # Get other schedulers (or None)

        best_val_auc_fold = 0.0
        best_epoch_fold = 0

        # --- Epoch Loop ---
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch + 1}/{config.epochs}")

            # Training step - use user's train_one_epoch signature
            train_loss, train_auc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.device,
                config, # Pass config here
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            # Validation step - use user's validate signature
            val_loss, val_auc = validate(
                model,
                val_loader,
                criterion,
                config.device,
                config # Pass config here
            )

            # Step the scheduler (non-OneCycleLR)
            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    # Step ReduceLROnPlateau based on validation loss (matching user version)
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val AUC:   {val_auc:.4f}")

            # --- Store Metrics for Plotting --- #
            fold_history[fold]['train_loss'].append(train_loss)
            fold_history[fold]['train_auc'].append(train_auc)
            fold_history[fold]['val_loss'].append(val_loss)
            fold_history[fold]['val_auc'].append(val_auc)

            # --- Model Checkpointing ---
            if val_auc > best_val_auc_fold:
                best_val_auc_fold = val_auc
                best_epoch_fold = epoch + 1
                print(f"  ✨ New best AUC for fold {fold}: {best_val_auc_fold:.4f} at epoch {best_epoch_fold}. Saving model...")

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': best_val_auc_fold,
                    'train_auc': train_auc,
                }
                # Use config for output path and model name
                save_path = os.path.join(config.MODEL_OUTPUT_DIR, f"{config.model_name}_fold{fold}_best.pth")
                try:
                    torch.save(checkpoint, save_path)
                    print(f"  Model saved to {save_path}")
                except Exception as e:
                    print(f"  Error saving model checkpoint: {e}")

            # --- Optuna Pruning --- #
            # Only report and check for pruning after the first fold of the epoch
            if trial is not None and fold == config.selected_folds[0]:
                trial.report(val_auc, epoch) # Report intermediate result (using first fold's AUC)
                print(f"  Trial {trial.number} | Epoch {epoch + 1} | Val AUC: {val_auc:.4f}")
                if trial.should_prune():
                    print(f"  ❌ Trial pruned based on Fold {fold} at epoch {epoch + 1}.")
                    # Clean up memory before raising exception
                    # (Cleanup code is important here before raising)
                    del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
                    del train_df_fold, val_df_fold
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise optuna.TrialPruned() # Signal Optuna to prune

        # --- End of Fold --- #
        print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc_fold:.4f} at epoch {best_epoch_fold}")
        oof_scores.append(best_val_auc_fold)

        # Clean up memory
        del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
        del train_df_fold, val_df_fold # Explicitly delete fold dataframes
        torch.cuda.empty_cache()
        gc.collect()

    # --- End of Training --- #
    print("\n" + "="*60)
    print("Cross-Validation Training Summary:")
    mean_oof_score = 0.0 # Initialize
    if oof_scores:
        for i, score in enumerate(oof_scores):
            # Use config.selected_folds to report correct fold number
            fold_num = config.selected_folds[i]
            print(f"  Fold {fold_num}: Best Val AUC = {score:.4f}")
        mean_oof_score = np.mean(oof_scores) # Calculate mean
        print(f"\nMean OOF AUC across {len(oof_scores)} trained folds: {mean_oof_score:.4f}")
    else:
        print("No folds were trained.")
    print("="*60)

    # If the loop finished without pruning, return the final mean score
    return mean_oof_score, fold_history

# --- Plotting Function --- #
def plot_training_history(history, config):
    """Generates and saves plots for training/validation loss and AUC across folds."""
    print("\nGenerating training history plot...")
    try:
        epochs_ran = config.epochs # Assuming all folds run all epochs
        epoch_axis = range(1, epochs_ran + 1)

        # Aggregate history across folds
        # Filter out folds that might not have completed all epochs (e.g., if run stopped early)
        all_train_loss = np.array([history[f]['train_loss'] for f in config.selected_folds if len(history.get(f, {}).get('train_loss', [])) == epochs_ran])
        all_val_loss = np.array([history[f]['val_loss'] for f in config.selected_folds if len(history.get(f, {}).get('val_loss', [])) == epochs_ran])
        all_train_auc = np.array([history[f]['train_auc'] for f in config.selected_folds if len(history.get(f, {}).get('train_auc', [])) == epochs_ran])
        all_val_auc = np.array([history[f]['val_auc'] for f in config.selected_folds if len(history.get(f, {}).get('val_auc', [])) == epochs_ran])

        # Check if we actually have history data from at least one fold
        if all_train_loss.size > 0:
            mean_train_loss = np.mean(all_train_loss, axis=0)
            std_train_loss = np.std(all_train_loss, axis=0)
            mean_val_loss = np.mean(all_val_loss, axis=0)
            std_val_loss = np.std(all_val_loss, axis=0)
            mean_train_auc = np.mean(all_train_auc, axis=0)
            std_train_auc = np.std(all_train_auc, axis=0)
            mean_val_auc = np.mean(all_val_auc, axis=0)
            std_val_auc = np.std(all_val_auc, axis=0)

            # Create plot
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            fig.suptitle(f'Training History ({config.model_name}, {config.n_fold}-Fold CV)')

            # Loss subplot
            axs[0].plot(epoch_axis, mean_train_loss, label='Mean Train Loss', color='tab:blue')
            axs[0].fill_between(epoch_axis, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, color='tab:blue')
            axs[0].plot(epoch_axis, mean_val_loss, label='Mean Val Loss', color='tab:orange')
            axs[0].fill_between(epoch_axis, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, color='tab:orange')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].set_title('Loss vs. Epochs')
            axs[0].legend()
            axs[0].grid(True)

            # AUC subplot
            axs[1].plot(epoch_axis, mean_train_auc, label='Mean Train AUC', color='tab:blue')
            axs[1].fill_between(epoch_axis, mean_train_auc - std_train_auc, mean_train_auc + std_train_auc, alpha=0.3, color='tab:blue')
            axs[1].plot(epoch_axis, mean_val_auc, label='Mean Val AUC', color='tab:orange')
            axs[1].fill_between(epoch_axis, mean_val_auc - std_val_auc, mean_val_auc + std_val_auc, alpha=0.3, color='tab:orange')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('AUC')
            axs[1].set_title('AUC vs. Epochs')
            axs[1].legend()
            axs[1].grid(True)

            # Save the plot
            plot_save_path = os.path.join(config.OUTPUT_DIR, "training_metrics_plot.png")
            # Ensure output directory exists before saving
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(plot_save_path)
            print(f"Training plot saved to: {plot_save_path}")
            plt.close(fig) # Close the figure to free memory
        else:
            print("Skipping plot generation: No history data collected or data is incomplete.")

    except Exception as e:
        print(f"Error generating training plot: {e}")

if __name__ == "__main__":
    print("\n--- Initializing Training Script --- ")
    set_seed(config.seed) # Set seed for reproducibility

    # --- Load CHUNKED Metadata --- 
    print(f"Loading CHUNKED training metadata from: {config.CHUNKED_METADATA_PATH}")
    try:
        # Load the chunked dataframe generated by preprocessing.py
        main_chunked_train_df = pd.read_csv(config.CHUNKED_METADATA_PATH)
        print(f"Loaded {len(main_chunked_train_df)} chunk entries.")
    except FileNotFoundError:
        print(f"Error: Chunked metadata CSV not found at {config.CHUNKED_METADATA_PATH}. Run preprocessing first. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading chunked metadata CSV: {e}. Exiting.")
        sys.exit(1)

    # --- Run Training using the CHUNKED DataFrame --- 
    # Pass the chunked dataframe directly to run_training
    mean_oof, history = run_training(main_chunked_train_df, config)

    # --- Plot History --- 
    if history:
        plot_training_history(history, config)
    else:
        print("No history returned from training run, skipping plotting.")

    print("\nTraining script finished!")