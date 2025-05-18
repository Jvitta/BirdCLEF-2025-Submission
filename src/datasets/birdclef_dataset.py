import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import random
import sys # For sys.exit in ADL helpers
from math import radians, sin, cos, sqrt, atan2 # Added for Haversine
# It seems 'config' is an object passed around, not directly imported here.
# If 'config' is expected to be available globally, this might need adjustment,
# but typically it would be passed to the dataset or functions.
# For now, assuming config is passed as an argument as it is in the original train_mn.py

# --- Haversine Distance Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)."""
    # Radius of earth in kilometers
    R = 6371.0

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# --- Global cache for AdaIN per-frequency stats ---
# This global variable might be an issue if multiple dataset instances are created
# and expect different stats. Consider making it an instance variable or handling it differently.
_adain_per_freq_stats_cache = None

def _load_adain_per_freq_stats(config_obj):
    global _adain_per_freq_stats_cache
    if _adain_per_freq_stats_cache is None:
        try:
            # Assuming config_obj has ADAIN_PER_FREQUENCY_STATS_PATH
            _adain_per_freq_stats_cache = np.load(config_obj.ADAIN_PER_FREQUENCY_STATS_PATH)
            expected_keys = ['mu_t_mean_per_freq', 'sigma_t_mean_per_freq', 'mu_t_std_per_freq', 'sigma_t_std_per_freq',
                               'mu_ss_mean_per_freq', 'sigma_ss_mean_per_freq', 'mu_ss_std_per_freq', 'sigma_ss_std_per_freq']
            if not all(key in _adain_per_freq_stats_cache for key in expected_keys):
                print(f"ERROR: AdaIN per-frequency stats file ({config_obj.ADAIN_PER_FREQUENCY_STATS_PATH}) is missing one or more expected keys. Per-frequency AdaIN cannot proceed. Exiting.")
                sys.exit(1) # Or raise an error
        except Exception as e:
            print(f"ERROR loading AdaIN per-frequency stats from {config_obj.ADAIN_PER_FREQUENCY_STATS_PATH}: {e}. Per-frequency AdaIN cannot proceed. Exiting.")
            sys.exit(1) # Or raise an error
    return _adain_per_freq_stats_cache

def _apply_adain_transformation(spec_np, adain_config):
    """Applies AdaIN transformation to a numpy spectrogram.
    Supports 'global' or 'per_frequency' mode based on adain_config.ADAIN_MODE.
    """
    if adain_config.ADAIN_MODE == 'global':
        # Assuming adain_config has MU_T_MEAN, SIGMA_T_MEAN, etc.
        mu_s_train = np.mean(spec_np)
        sigma_s_train = np.std(spec_np)

        mu_s_train_new = ((mu_s_train - adain_config.MU_T_MEAN) / (adain_config.SIGMA_T_MEAN + adain_config.ADAIN_EPSILON)) * \
                         adain_config.SIGMA_SS_MEAN + adain_config.MU_SS_MEAN
        sigma_s_train_new = ((sigma_s_train - adain_config.MU_T_STD) / (adain_config.SIGMA_T_STD + adain_config.ADAIN_EPSILON)) * \
                            adain_config.SIGMA_SS_STD + adain_config.MU_SS_STD
        
        transformed_spec = ((spec_np - mu_s_train) / (sigma_s_train + adain_config.ADAIN_EPSILON)) * \
                           sigma_s_train_new + mu_s_train_new
        
    elif adain_config.ADAIN_MODE == 'per_frequency':
        stats = _load_adain_per_freq_stats(adain_config)
        if stats is None:
            print("Skipping per-frequency AdaIN due to stats loading issue.")
            return spec_np.astype(np.float32)

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

        mu_s_train_pf = np.mean(spec_np, axis=1)
        sigma_s_train_pf = np.std(spec_np, axis=1)

        mu_s_train_new_pf = ((mu_s_train_pf - mu_t_mean_pf) / (sigma_t_mean_pf + adain_config.ADAIN_EPSILON)) * \
                             sigma_ss_mean_pf + mu_ss_mean_pf
        sigma_s_train_new_pf = ((sigma_s_train_pf - mu_t_std_pf) / (sigma_t_std_pf + adain_config.ADAIN_EPSILON)) * \
                               sigma_ss_std_pf + mu_ss_std_pf
        
        transformed_spec = ((spec_np - mu_s_train_pf[:, np.newaxis]) / (sigma_s_train_pf[:, np.newaxis] + adain_config.ADAIN_EPSILON)) * \
                           sigma_s_train_new_pf[:, np.newaxis] + mu_s_train_new_pf[:, np.newaxis]
    else: # 'none' or unknown mode
        return spec_np.astype(np.float32)
    
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

        # Initialize distance weighting parameters
        self.enable_distance_weighting = getattr(self.config, 'ENABLE_DISTANCE_WEIGHTING', False)
        if self.enable_distance_weighting:
            self.el_silencio_lat = self.config.EL_SILENCIO_LAT
            self.el_silencio_lon = self.config.EL_SILENCIO_LON
            self.dist_threshold_km = self.config.DISTANCE_WEIGHTING_THRESHOLD_KM
            self.min_dist_weight = self.config.MIN_DISTANCE_WEIGHT
            self.default_weight_missing_coords_config = getattr(self.config, 'DEFAULT_WEIGHT_FOR_MISSING_COORDS', 'min')
            
            if self.default_weight_missing_coords_config == 'min':
                self.default_weight_value = self.min_dist_weight
            elif isinstance(self.default_weight_missing_coords_config, (float, int)):
                self.default_weight_value = float(self.default_weight_missing_coords_config)
            else: # Fallback for unrecognized string or 'mean_of_calculated_weights' (not implemented here)
                print(f"Warning: DEFAULT_WEIGHT_FOR_MISSING_COORDS='{self.default_weight_missing_coords_config}' not fully supported or invalid. Defaulting to min_dist_weight.")
                self.default_weight_value = self.min_dist_weight

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
        secondary_labels_str = str(row.get('secondary_labels', ''))
        filename_for_error = row.get('filename', samplename) # Use filename if available
        data_source = row['data_source'] 
        
        sample_weight = 1.0 # Default
        if (self.mode == 'train' or self.mode == 'valid') and self.enable_distance_weighting:
            current_lat_val = row.get('latitude')
            current_lon_val = row.get('longitude')

            if pd.notna(current_lat_val) and pd.notna(current_lon_val):
                try:
                    distance = haversine_distance(current_lat_val, current_lon_val, self.el_silencio_lat, self.el_silencio_lon)
                    if distance <= self.dist_threshold_km:
                        sample_weight = 1.0
                    else:
                        # Define a falloff range, e.g., 4000km beyond threshold to reach min_weight
                        falloff_range_km = self.config.DISTANCE_WEIGHTING_FALLOFF_RANGE_KM
                        excess_distance = distance - self.dist_threshold_km
                        # Linear scaling
                        weight_reduction_factor = excess_distance / falloff_range_km
                        sample_weight = 1.0 - (1.0 - self.min_dist_weight) * weight_reduction_factor
                        sample_weight = max(self.min_dist_weight, min(1.0, sample_weight)) # Clamp weight
                except (ValueError, TypeError): # If lat/lon conversion to float fails
                    sample_weight = self.default_weight_value
            else: # Latitude or Longitude is NaN
                sample_weight = self.default_weight_value
        
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
        parsed_secondary_labels = []
        if pd.notna(secondary_labels_str) and secondary_labels_str not in ['nan', '', '[]']:
            try: parsed_secondary_labels = eval(secondary_labels_str)
            except: pass # Keep empty if eval fails
        if isinstance(parsed_secondary_labels, list):
            for label in parsed_secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0 - self.config.label_smoothing_factor

        output_dict = {
            'melspec': spec_tensor,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': filename_for_error,
            'samplename': samplename,
            'source': data_source 
        }
        if (self.mode == 'train' or self.mode == 'valid') and self.enable_distance_weighting:
            output_dict['sample_weight'] = torch.tensor(sample_weight, dtype=torch.float32)

        return output_dict

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