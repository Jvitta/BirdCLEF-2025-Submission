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
import cv2
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.utils import resample
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import math

from config import config
import utils as utils
from models.efficient_at.mn.model import get_model as get_efficient_at_model
from models.efficient_at.dymn.model import get_model as get_dymn_model
from losses import FocalLossBCE

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

# --- Global cache for AdaIN per-frequency stats ---
_adain_per_freq_stats_cache = None

def _load_adain_per_freq_stats(config_obj):
    global _adain_per_freq_stats_cache
    if _adain_per_freq_stats_cache is None:
        try:
            _adain_per_freq_stats_cache = np.load(config_obj.ADAIN_PER_FREQUENCY_STATS_PATH)
            expected_keys = ['mu_t_mean_per_freq', 'sigma_t_mean_per_freq', 'mu_t_std_per_freq', 'sigma_t_std_per_freq',
                               'mu_ss_mean_per_freq', 'sigma_ss_mean_per_freq', 'mu_ss_std_per_freq', 'sigma_ss_std_per_freq']
            if not all(key in _adain_per_freq_stats_cache for key in expected_keys):
                print(f"ERROR: AdaIN per-frequency stats file ({config_obj.ADAIN_PER_FREQUENCY_STATS_PATH}) is missing one or more expected keys. Per-frequency AdaIN cannot proceed. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR loading AdaIN per-frequency stats from {config_obj.ADAIN_PER_FREQUENCY_STATS_PATH}: {e}. Per-frequency AdaIN cannot proceed. Exiting.")
            sys.exit(1)
    return _adain_per_freq_stats_cache

def _apply_adain_transformation(spec_np, adain_config):
    """Applies AdaIN transformation to a numpy spectrogram.
    Supports 'global' or 'per_frequency' mode based on adain_config.ADAIN_MODE.
    """
    if adain_config.ADAIN_MODE == 'global':
        # Calculate original mean and std of the current spectrogram (overall)
        mu_s_train = np.mean(spec_np)
        sigma_s_train = np.std(spec_np)

        # Calculate the target mean for this specific spectrogram (overall)
        mu_s_train_new = ((mu_s_train - adain_config.MU_T_MEAN) / (adain_config.SIGMA_T_MEAN + adain_config.ADAIN_EPSILON)) * \
                         adain_config.SIGMA_SS_MEAN + adain_config.MU_SS_MEAN

        # Calculate the target internal standard deviation for this specific spectrogram (overall)
        sigma_s_train_new = ((sigma_s_train - adain_config.MU_T_STD) / (adain_config.SIGMA_T_STD + adain_config.ADAIN_EPSILON)) * \
                            adain_config.SIGMA_SS_STD + adain_config.MU_SS_STD
        
        # Apply the AdaIN transformation (overall)
        transformed_spec = ((spec_np - mu_s_train) / (sigma_s_train + adain_config.ADAIN_EPSILON)) * \
                           sigma_s_train_new + mu_s_train_new
        
    elif adain_config.ADAIN_MODE == 'per_frequency':
        stats = _load_adain_per_freq_stats(adain_config)
        if stats is None:
            print("Skipping per-frequency AdaIN due to stats loading issue.")
            return spec_np.astype(np.float32) # Return original if stats not loaded

        # Extract per-frequency stats from loaded NPZ
        mu_t_mean_pf = stats['mu_t_mean_per_freq']
        sigma_t_mean_pf = stats['sigma_t_mean_per_freq']
        mu_t_std_pf = stats['mu_t_std_per_freq']
        sigma_t_std_pf = stats['sigma_t_std_per_freq']
        mu_ss_mean_pf = stats['mu_ss_mean_per_freq']
        sigma_ss_mean_pf = stats['sigma_ss_mean_per_freq']
        mu_ss_std_pf = stats['mu_ss_std_per_freq']
        sigma_ss_std_pf = stats['sigma_ss_std_per_freq']

        if not (mu_t_mean_pf.shape[0] == spec_np.shape[0] == adain_config.N_MELS):
            print(f"ERROR: Mismatch in N_MELS for per-frequency AdaIN stats ({mu_t_mean_pf.shape[0]}) and spectrogram ({spec_np.shape[0]}). Expected {adain_config.N_MELS}. Skipping.")
            return spec_np.astype(np.float32)

        # Calculate original mean and std for each frequency bin of the current spectrogram
        mu_s_train_pf = np.mean(spec_np, axis=1)  # Shape: (N_MELS,)
        sigma_s_train_pf = np.std(spec_np, axis=1) # Shape: (N_MELS,)

        # Calculate the target mean for each frequency bin
        # Ensure broadcasting for per-element operations (N_MELS,)
        mu_s_train_new_pf = ((mu_s_train_pf - mu_t_mean_pf) / (sigma_t_mean_pf + adain_config.ADAIN_EPSILON)) * \
                             sigma_ss_mean_pf + mu_ss_mean_pf

        # Calculate the target std for each frequency bin
        sigma_s_train_new_pf = ((sigma_s_train_pf - mu_t_std_pf) / (sigma_t_std_pf + adain_config.ADAIN_EPSILON)) * \
                               sigma_ss_std_pf + mu_ss_std_pf
        
        # Apply the AdaIN transformation per frequency bin
        # spec_np is (N_MELS, N_TIMEBINS)
        # mu_s_train_pf, sigma_s_train_pf are (N_MELS,)
        # mu_s_train_new_pf, sigma_s_train_new_pf are (N_MELS,)
        # We need to reshape for broadcasting: (N_MELS, 1)
        transformed_spec = ((spec_np - mu_s_train_pf[:, np.newaxis]) / (sigma_s_train_pf[:, np.newaxis] + adain_config.ADAIN_EPSILON)) * \
                           sigma_s_train_new_pf[:, np.newaxis] + mu_s_train_new_pf[:, np.newaxis]
    else: # 'none' or unknown mode
        return spec_np.astype(np.float32)
    
    # Ensure spec remains float32 after transformation
    return transformed_spec.astype(np.float32)

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

        # Load taxonomy data
        taxonomy_df = pd.read_csv(self.config.taxonomy_path)
        self.species_ids = taxonomy_df['primary_label'].tolist()

        assert len(self.species_ids) == self.config.num_classes, \
            f"Number of species in taxonomy ({len(self.species_ids)}) does not match config.num_classes ({self.config.num_classes})."

        self.num_classes = self.config.num_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if self.all_spectrograms is not None:
            print(f"Dataset mode '{self.mode}': Using pre-loaded spectrogram dictionary.")
            print(f"Found {len(self.all_spectrograms)} samplenames with precomputed chunks.")
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
        data_source = row['data_source'] 
        
        spec = None # This will hold the final (H_5s, W_5s) processed chunk

        if samplename in self.all_spectrograms:
            spec_data_from_npz = self.all_spectrograms[samplename]
            raw_selected_chunk_2d = None # This will be the 2D chunk selected, potentially >5s wide

            if isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.ndim == 3:
                # Assumes data is always (N, H, W) N is number of chunks, H is height, W is width
                num_available_chunks = spec_data_from_npz.shape[0]
                
                if num_available_chunks > 0:
                    selected_idx = 0
                    if self.mode == 'train' and num_available_chunks > 1:
                        selected_idx = random.randint(0, num_available_chunks - 1)
                    raw_selected_chunk_2d = spec_data_from_npz[selected_idx]
                else:
                    print(f"WARNING: Data for '{samplename}' is a 3D array but has 0 chunks. Using zeros.")
                    
            else:
                ndim_info = spec_data_from_npz.ndim if isinstance(spec_data_from_npz, np.ndarray) else "Not an ndarray"
                print(f"WARNING: Data for '{samplename}' has unexpected ndim {ndim_info} or type. Expected 3D ndarray. Using zeros.")

            # Now, raw_selected_chunk_2d should be a single 2D spectrogram.
            if raw_selected_chunk_2d is not None:
                expected_shape = tuple(self.config.PREPROCESS_TARGET_SHAPE)
                if raw_selected_chunk_2d.shape == expected_shape:
                    spec = raw_selected_chunk_2d
                else:
                    current_samplename = self.df.iloc[idx]['samplename'] # Use self.df
                    print(f"WARNING: Samplename '{current_samplename}' - "
                          f"loaded chunk shape {raw_selected_chunk_2d.shape} "
                          f"does not match PREPROCESS_TARGET_SHAPE {expected_shape}. Attempting resize.")
                    spec = cv2.resize(raw_selected_chunk_2d,
                                      (self.config.PREPROCESS_TARGET_SHAPE[1], self.config.PREPROCESS_TARGET_SHAPE[0]),
                                      interpolation=cv2.INTER_LINEAR)
            else:
                # This implies select_version_for_training returned None, or an issue during loading from NPZ for this sample.
                current_samplename = self.df.iloc[idx]['samplename'] # Use self.df
                print(f"ERROR: Samplename '{current_samplename}' - "
                      f"no valid chunk could be selected or loaded from NPZ. Using zeros as fallback.")
                spec = np.zeros(tuple(self.config.PREPROCESS_TARGET_SHAPE), dtype=np.float32)

            # Fallback if spec is still None or issues occurred during processing
            if spec is None or spec.shape != self.config.PREPROCESS_TARGET_SHAPE:
                 original_shape_info = spec_data_from_npz.shape if isinstance(spec_data_from_npz, np.ndarray) else type(spec_data_from_npz)
                 current_spec_shape_info = spec.shape if spec is not None else "None"
                 print(f"Fallback: Using zeros for '{samplename}'. Raw NPZ shape: {original_shape_info}, Processed spec shape before fallback: {current_spec_shape_info}.")
                 spec = np.zeros(self.config.PREPROCESS_TARGET_SHAPE, dtype=np.float32)
        
        else:
            print(f"ERROR: Samplename '{samplename}' not found in pre-loaded dictionary! Using zeros.")
            spec = np.zeros(self.config.PREPROCESS_TARGET_SHAPE, dtype=np.float32)

        # --- Final Shape Guarantee --- (important for downstream code)
        if not isinstance(spec, np.ndarray) or spec.shape != tuple(self.config.PREPROCESS_TARGET_SHAPE):
             print(f"CRITICAL WARNING: Final spec for '{samplename}' has wrong shape/type ({spec.shape if isinstance(spec, np.ndarray) else type(spec)}) before unsqueeze. Forcing zeros.")
             spec = np.zeros(self.config.PREPROCESS_TARGET_SHAPE, dtype=np.float32)

        # Ensure spec is float32 before AdaIN or other augmentations
        spec = spec.astype(np.float32)

        # --- AdaIN Transformation (if enabled and in train mode) ---
        if self.config.ADAIN_MODE != 'none': # Apply if AdaIN is enabled, regardless of mode
            if self.mode == "train":
                transform_weight = np.random.uniform(0.0, 0.25)
                spec = (1-transform_weight)*spec + transform_weight*_apply_adain_transformation(spec, self.config)
            elif self.mode == "val":
                spec = 0.875*spec + 0.125*_apply_adain_transformation(spec, self.config)
        # --- End AdaIN Transformation ---

        # Apply manual SpecAugment (Time/Freq Mask, Contrast) on NumPy array
        if self.mode == "train":
            spec = self.apply_spec_augmentations(spec)

        # concatenate the spec with itself to make it 10s long
        spec = np.concatenate([spec, spec], axis=1)

        # Convert to tensor, add channel dimension
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

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
            'filename': filename_for_error,
            'samplename': samplename,
            'source': data_source 
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

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if batch is None:
            print(f"Warning: Skipping None batch at step {step}")
            continue

        try:
            inputs_orig = batch['melspec'].to(device)
            targets_orig = batch['target'].to(device)
            sources_orig = batch['source'] # List of strings, no .to(device)
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
    use_amp = config.use_amp
    species_ids = loader.dataset.species_ids 
    num_classes = len(species_ids)

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Validation")):
            #print(f"DEBUG: Validation step {step}, processing batch...") # DEBUG PRINT
            if batch is None:
                print(f"Warning: Skipping None validation batch at step {step}")
                continue

            try:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                #if 'filename' not in batch or 'samplename' not in batch:
                    #print(f"DEBUG: Validation step {step}, batch loaded. No filename/samplename key in batch.")

            except (AttributeError, TypeError) as e:
                print(f"Error: Skipping validation batch {step} due to unexpected format: {e}")
                continue

            #print(f"DEBUG: Validation step {step}, about to run model forward pass.") # DEBUG PRINT
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                model_output = model(inputs)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output # Get logits
                #print(f"DEBUG: Validation step {step}, model forward pass complete.") # DEBUG PRINT
                loss = criterion(outputs, targets)

            all_outputs_list.append(outputs.float().cpu().numpy())
            all_targets_list.append(torch.ceil(targets).cpu().numpy()) #use ceil to convert to 0/1 targets for label smoothing
            losses.append(loss.item())

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
    
    macro_auc = calculate_auc(all_targets_cat, all_outputs_cat) # Existing macro AUC calculation
    avg_loss = np.mean(losses) if losses else 0.0

    # --- Per-Class Metrics Calculation ---
    per_class_metrics_list = []
    probabilities_cat = 1 / (1 + np.exp(-all_outputs_cat)) # Sigmoid probabilities
    predictions_binary_cat = (probabilities_cat >= 0.5).astype(int)

    # Get TP, FP, FN, TN for all classes using multilabel_confusion_matrix
    try:
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
                class_auc = roc_auc_score(class_targets, class_probs)
            except ValueError:
                class_auc = 0.0 # Or handle as NaN, but 0.0 is simpler for aggregation
        
        # If class_targets are all 0, precision/recall/f1 for label 1 will be 0 due to zero_division=0
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            class_targets, class_preds_binary, average='binary', pos_label=1, zero_division=0
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

def run_training(df, config, trial=None, all_spectrograms=None, custom_run_name=None):
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
    # Use custom run name if provided, otherwise generate one
    if custom_run_name:
        run_name = custom_run_name
    else:
        run_name = f"trial_{trial.number}" if is_hpo_trial else f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    
    group_name = "hpo" if is_hpo_trial else "full_training"

    # Select relevant config parameters for logging
    relevant_config_params = config.get_wandb_config()
    
    wandb_run = wandb.init(
        project="BirdCLEF-2025", # Or your preferred project name
        config=relevant_config_params, # Use the curated dictionary
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

    if config.USE_RARE_DATA:
        print("\n--- Loading Rare Species Data (USE_RARE_DATA=True) ---")
        try:
            rare_train_df_full = pd.read_csv(config.train_rare_csv_path)
            if not rare_train_df_full.empty:
                # Ensure 'filename' and 'primary_label' exist, which they should from the generator script
                if 'filename' not in rare_train_df_full.columns or 'primary_label' not in rare_train_df_full.columns:
                    print("ERROR: train_rare.csv is missing 'filename' or 'primary_label' column. Skipping rare data.")
                else:
                    rare_train_df_full['filepath'] = rare_train_df_full['filename'].apply(lambda f: os.path.join(config.train_audio_rare_dir, f))
                    # Consistent samplename generation for rare data
                    rare_train_df_full['samplename'] = rare_train_df_full.filename.map(
                        lambda x: str(x.split('/')[0]) + '-' + str(x.split('/')[-1].split('.')[0])
                    )
                    rare_train_df_full['data_source'] = 'rare' # Add data source identifier
                    # Ensure secondary_labels exists, even if empty, for consistent concatenation
                    if 'secondary_labels' not in rare_train_df_full.columns:
                         rare_train_df_full['secondary_labels'] = [[] for _ in range(len(rare_train_df_full))]

                    required_cols_rare = ['samplename', 'primary_label', 'secondary_labels', 'filepath', 'filename', 'data_source']
                    # Select only necessary columns, ensuring all exist
                    missing_cols_rare = [col for col in required_cols_rare if col not in rare_train_df_full.columns]
                    if missing_cols_rare:
                        print(f"ERROR: Rare DataFrame is missing required columns after processing: {missing_cols_rare}. Skipping rare data.")
                    else:
                        rare_train_df = rare_train_df_full[required_cols_rare].copy()
                        print(f"Loaded and selected columns for {len(rare_train_df)} rare training samples.")
                        del rare_train_df_full; gc.collect()
                        
                        # Concatenate with main training data
                        working_df = pd.concat([working_df, rare_train_df], ignore_index=True)
                        print(f"Combined DataFrame size (main + rare): {len(working_df)} samples.")
            else:
                print("train_rare.csv found but is empty. No rare data added.")

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

    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
   
    all_folds_history = []
    best_epoch_details_all_folds = [] 
    single_fold_best_auc = 0.0 
    overall_species_ids_for_run = None # Initialize to store species_ids
    
    # New tracker for spectrograms logged in train_one_epoch
    logged_original_samplenames_for_spectrograms = [] # Using a list to preserve order of first N logged
    
    try: # Wrap the main training loop in try/finally for wandb.finish()
        for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
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

            # --- Filter Validation Set: Allow 'main' or 'rare' data sources --- 
            original_val_count = len(val_df_fold)
            if 'data_source' in val_df_fold.columns:
                allowed_sources = ['main', 'rare']
                val_df_fold = val_df_fold[val_df_fold['data_source'].isin(allowed_sources)].reset_index(drop=True)
                print(f"Filtered validation set to include 'main' or 'rare' data sources.")
                print(f"  Original val count: {original_val_count}, Filtered val count: {len(val_df_fold)}")
            else:
                print("Warning: 'data_source' column not found in validation fold. Cannot filter; proceeding with all samples in val_df_fold.")
            # --- End Validation Set Filtering ---
            
            print(f'Training set: {len(train_df_fold)} samples (includes main, rare, and potentially pseudo)')
            print(f'Validation set: {len(val_df_fold)} samples (main or rare data only)')

            # Pass the pre-loaded dictionary (or None) to the Dataset
            # NOW, pass both the hardcoded targets AND the shared tracking set (REMOVED THESE)
            train_dataset = BirdCLEFDataset(train_df_fold, config, mode='train', all_spectrograms=all_spectrograms) 
            val_dataset = BirdCLEFDataset(val_df_fold, config, mode='valid', all_spectrograms=all_spectrograms) 

            # Get species_ids from the first validation dataset instance created
            if overall_species_ids_for_run is None and hasattr(val_dataset, 'species_ids'):
                overall_species_ids_for_run = val_dataset.species_ids

            train_sampler = None
            shuffle_train_loader = True

            if config.USE_WEIGHTED_SAMPLING:
                print("Using WeightedRandomSampler for training data based on 'samplename'.")
                # Extract species identifiers from the 'samplename' column of the current fold's training data
                # Assumes train_df_fold['samplename'] exists and is in format 'species_id-other_info'
                if 'samplename' not in train_df_fold.columns:
                    print("ERROR: 'samplename' column not found in train_df_fold. Cannot apply weighted sampling. Defaulting to shuffle=True")
                    train_sampler = None
                    shuffle_train_loader = True
                else:
                    try:
                        all_species_ids_for_fold_samples = [sn.split('-')[0] for sn in train_df_fold['samplename']]
                        
                        # Calculate frequency of each species_id in this fold
                        species_id_counts_in_fold = Counter(all_species_ids_for_fold_samples)
                        
                        # Calculate weight for each sample in this fold: 1 / sqrt(count of its species_id)
                        sample_weights_for_fold = [1.0 / math.sqrt(max(1, species_id_counts_in_fold[sid])) for sid in all_species_ids_for_fold_samples]
                        
                        weights_tensor = torch.tensor(sample_weights_for_fold, dtype=torch.float)
                        train_sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
                        shuffle_train_loader = False # Sampler handles shuffling
                        print(f"  WeightedRandomSampler created for fold {fold} with {len(weights_tensor)} weights.")
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
            # model = BirdCLEFModel(config).to(config.device) # Old model instantiation
            # New model instantiation using EfficientAT's get_model
            with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
                model = get_efficient_at_model(
                    num_classes=config.num_classes,          
                    pretrained_name=config.model_name,               
                    width_mult=config.width_mult,                         
                    head_type="mlp",                         
                    input_dim_f=config.TARGET_SHAPE[0],     
                    input_dim_t=config.TARGET_SHAPE[1]      
                ).to(config.device)
            
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
                    trial.report(val_auc, epoch) # Report intermediate val_auc
                    if trial.should_prune():
                        print(f"  Pruning trial based on intermediate value at epoch {epoch+1}.")
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

                plot_dir = os.path.join(config.OUTPUT_DIR, "training_curves")
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
                if wandb.run: # wandb.run is the global accessor for the current active run status
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

    run_training(training_df, config, all_spectrograms=all_spectrograms, custom_run_name=cmd_args.run_name)

    print("\nTraining script finished!")