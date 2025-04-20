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

from tqdm.auto import tqdm

import timm

from config import config
import birdclef_utils as utils

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

set_seed(config.seed)

class BirdCLEFDataset(Dataset):
    """Dataset class for BirdCLEF.
    Handles loading pre-computed spectrograms or generating them on-the-fly.
    """
    def __init__(self, df, config, spectrograms=None, mode="train"):
        self.df = df.copy() # Work on a copy to avoid modifying original df
        self.config = config # Use the imported config object
        self.mode = mode
        self.spectrograms = spectrograms

        # Load taxonomy info once
        try:
            taxonomy_df = pd.read_csv(self.config.taxonomy_path)
            self.species_ids = taxonomy_df['primary_label'].tolist()
            # Ensure num_classes in config matches taxonomy
            if len(self.species_ids) != self.config.num_classes:
                 print(f"Warning: Taxonomy file has {len(self.species_ids)} species, but config.num_classes is {self.config.num_classes}. Using value from taxonomy.")
                 self.num_classes = len(self.species_ids)
            else:
                self.num_classes = self.config.num_classes
            self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}
        except Exception as e:
            print(f"Error loading taxonomy CSV from {self.config.taxonomy_path}: {e}")
            raise # Re-raise exception as this is critical

        # Ensure necessary columns exist (filepath, samplename)
        if 'filepath' not in self.df.columns:
            # Use os.path.join for consistency
            self.df['filepath'] = self.df['filename'].apply(lambda f: os.path.join(self.config.train_audio_dir, f))
        if 'samplename' not in self.df.columns:
            # Use split/join method consistent with preprocessing.py
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        # Report on pre-computed spectrogram matching
        if self.spectrograms:
            sample_names = set(self.df['samplename'])
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Dataset mode '{self.mode}': Found {found_samples} matching pre-computed spectrograms for {len(self.df)} samples.")
        elif not self.config.LOAD_PREPROCESSED_DATA:
             print(f"Dataset mode '{self.mode}': Configured to generate spectrograms on-the-fly.")
        else:
             print(f"Dataset mode '{self.mode}': Configured to load pre-computed data from {self.config.PREPROCESSED_FILEPATH}, but no spectrogram dictionary provided to Dataset.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        # Try loading pre-computed first if configured and available
        if self.config.LOAD_PREPROCESSED_DATA and self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        # Otherwise, generate on-the-fly if configured
        elif not self.config.LOAD_PREPROCESSED_DATA:
            # Use the utility function
            spec = utils.process_audio_file(row['filepath'], self.config)

        # Handle cases where spec is still None (missing pre-computed or error in generation)
        if spec is None:
            spec = np.zeros(self.config.TARGET_SHAPE, dtype=np.float32)
            # Print warning only if generation was attempted or pre-computed was expected but missing
            if not self.config.LOAD_PREPROCESSED_DATA or (self.config.LOAD_PREPROCESSED_DATA and self.spectrograms is not None):
                 if self.mode == "train": # Often only critical during training
                    print(f"Warning: Spectrogram for {samplename} ({row['filepath']}) could not be loaded/generated. Using zeros.")

        # Ensure spec is float32 before converting to tensor
        spec = spec.astype(np.float32)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply augmentations only in training mode
        if self.mode == "train" and random.random() < self.config.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        # Encode primary and secondary labels
        target = self.encode_label(row['primary_label'])
        # Use the original logic for secondary labels from the user's version
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                try:
                    # Safely evaluate string representation
                    secondary_labels = eval(row['secondary_labels'])
                    if not isinstance(secondary_labels, list):
                         secondary_labels = [] # Ensure it's a list
                except:
                    secondary_labels = [] # Handle eval errors
            else:
                secondary_labels = row['secondary_labels'] # Assume it's already a list

            if isinstance(secondary_labels, list): # Check if it's a list before iterating
                 for label in secondary_labels:
                     if label in self.label_to_idx:
                         target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename'] # Keep filename
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

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
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
        # Pass targets only if mixup is enabled
        outputs = model(inputs, targets if model.training and model.mixup_enabled else None)

        # Handle model output (logits or logits, loss)
        if isinstance(outputs, tuple):
            logits, loss = outputs
        else:
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

def validate(model, loader, criterion, device):
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

def run_training(df, config): # Pass config object
    """Runs the cross-validation training loop (incorporating user logic)."""
    print("\n--- Starting Training Run ---")
    print(f"Using Device: {config.device}")
    print(f"Debug Mode: {config.debug}")
    print(f"Load Preprocessed Data: {config.LOAD_PREPROCESSED_DATA}")
    if config.LOAD_PREPROCESSED_DATA:
         print(f"Preprocessed data path: {config.PREPROCESSED_FILEPATH}")
    else:
         print("Generating spectrograms on-the-fly.")

    # Ensure model output directory exists
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # --- Restore Spectrogram Loading Logic --- #
    spectrograms = None
    # preprocessed_data_path_resolved = None # This variable was part of Kaggle specific logic, not needed here

    if config.LOAD_PREPROCESSED_DATA:
        print("\nAttempting to load pre-computed mel spectrograms...")
        # Use the PREPROCESSED_FILEPATH directly
        preprocessed_path = config.PREPROCESSED_FILEPATH
        print(f"Looking for pre-computed data at: {preprocessed_path}")
        if os.path.exists(preprocessed_path):
             try:
                 spectrograms = np.load(preprocessed_path, allow_pickle=True).item()
                 print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms.")
             except Exception as e:
                 print(f"Error loading file {preprocessed_path}: {e}")
                 print("Cannot proceed without preprocessed data when LOAD_PREPROCESSED_DATA is True.")
                 return 0.0 # Return 0.0 AUC for Optuna compatibility
        else:
             print(f"Error: Preprocessed data file not found at {preprocessed_path}")
             print("Please run preprocessing first or set LOAD_PREPROCESSED_DATA to False.")
             return 0.0 # Return 0.0 AUC for Optuna compatibility

    else:
        print("\nGenerating spectrograms on-the-fly.")

    # Create a working copy of the dataframe for modification
    working_df = df.copy()

    # Ensure required columns exist for on-the-fly generation if needed
    if not config.LOAD_PREPROCESSED_DATA:
        if 'filepath' not in working_df.columns:
             # Use os.path.join
            working_df['filepath'] = working_df['filename'].apply(lambda f: os.path.join(config.train_audio_dir, f))
        if 'samplename' not in working_df.columns:
             # Use split/join method consistent with preprocessing.py
             working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    # --- Cross-validation Setup (User's version) ---
    # Use primary_label for stratification
    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    oof_scores = [] # Renamed from best_scores

    # --- Fold Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
        if fold not in config.selected_folds:
            print(f"\nSkipping Fold {fold}...")
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')

        # --- Data Setup for Fold ---
        train_df_fold = working_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = working_df.iloc[val_idx].reset_index(drop=True)

        print(f'Training set: {len(train_df_fold)} samples')
        print(f'Validation set: {len(val_df_fold)} samples')

        # Create datasets for the current fold (PASS spectrograms dictionary)
        train_dataset = BirdCLEFDataset(train_df_fold, config, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDataset(val_df_fold, config, spectrograms=spectrograms, mode='valid')

        # Create dataloaders
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
        model = BirdCLEFModel(config).to(config.device)
        optimizer = get_optimizer(model, config)
        criterion = get_criterion(config)

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

        best_val_auc = 0.0
        best_epoch = 0
        # Removed patience counter / early stopping for simplicity, can be added back

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
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            # Validation step - use user's validate signature
            val_loss, val_auc = validate(
                model,
                val_loader,
                criterion,
                config.device
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
                # Use config for output path and model name
                save_path = os.path.join(config.MODEL_OUTPUT_DIR, f"{config.model_name}_fold{fold}_best.pth")
                try:
                    torch.save(checkpoint, save_path)
                    print(f"  Model saved to {save_path}")
                except Exception as e:
                    print(f"  Error saving model checkpoint: {e}")

        # --- End of Fold --- #
        print(f"\nFinished Fold {fold}. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
        oof_scores.append(best_val_auc)

        # Clean up memory
        del model, optimizer, criterion, scheduler, train_loader, val_loader, train_dataset, val_dataset
        del train_df_fold, val_df_fold # Explicitly delete fold dataframes
        torch.cuda.empty_cache()
        gc.collect()

    # --- End of Training --- #
    print("\n" + "="*60)
    print("Cross-Validation Training Summary:")
    if oof_scores:
        for i, score in enumerate(oof_scores):
            # Use config.selected_folds to report correct fold number
            fold_num = config.selected_folds[i]
            print(f"  Fold {fold_num}: Best Val AUC = {score:.4f}")
        print(f"\nMean OOF AUC across {len(oof_scores)} trained folds: {np.mean(oof_scores):.4f}")
    else:
        print("No folds were trained.")
    print("="*60)

if __name__ == "__main__":
    # Config is imported from config.py
    print("\n--- Initializing Training Script ---")

    # Load main training dataframe
    print("Loading main training metadata...")
    try:
        main_train_df = pd.read_csv(config.train_csv_path)
    except FileNotFoundError:
        print(f"Error: Main training CSV not found at {config.train_csv_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading main training CSV: {e}. Exiting.")
        sys.exit(1)

    # Start the training process
    run_training(main_train_df, config) # Pass the imported config

    print("\nTraining script finished!")