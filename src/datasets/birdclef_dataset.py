import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import random
import sys # For sys.exit in ADL helpers
from math import radians, sin, cos, sqrt, atan2 # Added for Haversine
import torchvision.transforms as transforms # For ImageNet normalization
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

class BirdCLEFDataset(Dataset):
    """Dataset class for BirdCLEF.
    Handles loading pre-computed spectrograms from a pre-loaded dictionary
    or generating them on-the-fly.
    Supports different preprocessing for EfficientNet vs. MobileNet architectures based on config.model_name.
    """
    def __init__(self, df, config, mode="train", all_spectrograms=None):
        self.df = df.copy()
        self.config = config
        self.mode = mode
        self.all_spectrograms = all_spectrograms
        # self.model_architecture = self.config.MODEL_ARCHITECTURE # Removed

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
            else: # Fallback
                print(f"Warning: DEFAULT_WEIGHT_FOR_MISSING_COORDS='{self.default_weight_missing_coords_config}' not fully supported or invalid. Defaulting to min_dist_weight.")
                self.default_weight_value = self.min_dist_weight

        # Setup based on model_name from config
        if "efficientnet" in self.config.model_name.lower():
            self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            print(f"Dataset configured for EfficientNet (model_name: {self.config.model_name}). Input shape: {self.config.TARGET_SHAPE}")
        elif "mn" in self.config.model_name.lower():
            print(f"Dataset configured for MobileNet (model_name: {self.config.model_name}). Chunk shape: {self.config.PREPROCESS_TARGET_SHAPE}, Final model input shape: {self.config.TARGET_SHAPE}")
        else:
            raise ValueError(f"Unsupported model_name for dataset configuration: {self.config.model_name}. Check if 'efficientnet' or 'mn' is in the name.")

        if self.all_spectrograms is not None:
            print(f"Dataset mode '{self.mode}': Using pre-loaded spectrogram dictionary for model: {self.config.model_name}.")
            print(f"Found {len(self.all_spectrograms)} samplenames with precomputed chunks.")
        elif self.config.LOAD_PREPROCESSED_DATA:
            print(f"Dataset mode '{self.mode}': ERROR - Configured to load preprocessed data, but none provided. Dataset will be empty.")
        else:
            print(f"Dataset mode '{self.mode}': Configured for on-the-fly generation from {len(self.df)} files (ensure preprocessing logic is present if used).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        
        primary_label = row['primary_label']
        secondary_labels_str = str(row.get('secondary_labels', ''))
        filename_for_error = row.get('filename', samplename) 
        data_source = row['data_source'] 
        
        sample_weight = 1.0 
        if (self.mode == 'train' or self.mode == 'valid') and self.enable_distance_weighting:
            current_lat_val = row.get('latitude')
            current_lon_val = row.get('longitude')

            if pd.notna(current_lat_val) and pd.notna(current_lon_val):
                try:
                    distance = haversine_distance(current_lat_val, current_lon_val, self.el_silencio_lat, self.el_silencio_lon)
                    if distance <= self.dist_threshold_km:
                        sample_weight = 1.0
                    else:
                        falloff_range_km = self.config.DISTANCE_WEIGHTING_FALLOFF_RANGE_KM
                        excess_distance = distance - self.dist_threshold_km
                        weight_reduction_factor = excess_distance / falloff_range_km
                        sample_weight = 1.0 - (1.0 - self.min_dist_weight) * weight_reduction_factor
                        sample_weight = max(self.min_dist_weight, min(1.0, sample_weight))
                except (ValueError, TypeError): 
                    sample_weight = self.default_weight_value
            else: 
                sample_weight = self.default_weight_value
        
        loaded_chunk_np = None # This will hold the initial 2D chunk from NPZ

        if samplename in self.all_spectrograms:
            spec_data_from_npz = self.all_spectrograms[samplename]
            
            if isinstance(spec_data_from_npz, np.ndarray) and spec_data_from_npz.ndim == 3:
                num_available_chunks = spec_data_from_npz.shape[0]
                if num_available_chunks > 0:
                    selected_idx = 0
                    if self.mode == 'train' and num_available_chunks > 1:
                        selected_idx = random.randint(0, num_available_chunks - 1)
                    loaded_chunk_np = spec_data_from_npz[selected_idx]
                else:
                    print(f"WARNING: Data for '{samplename}' (3D array) has 0 chunks. Using zeros.")
            else:
                print(f"WARNING: Data for '{samplename}' has unexpected format. Expected 3D ndarray. Using zeros.")

            # Ensure the loaded chunk matches PREPROCESS_TARGET_SHAPE (the shape NPZs should have)
            if loaded_chunk_np is not None:
                if loaded_chunk_np.shape != self.config.PREPROCESS_TARGET_SHAPE:
                    print(f"WARNING: Samplename '{samplename}' - loaded chunk shape {loaded_chunk_np.shape} "
                          f"does not match config.PREPROCESS_TARGET_SHAPE {self.config.PREPROCESS_TARGET_SHAPE}. Attempting resize.")
                    loaded_chunk_np = cv2.resize(loaded_chunk_np,
                                      (self.config.PREPROCESS_TARGET_SHAPE[1], self.config.PREPROCESS_TARGET_SHAPE[0]),
                                      interpolation=cv2.INTER_LINEAR)
            else: # Fallback if loaded_chunk_np is still None
                print(f"ERROR: Samplename '{samplename}' - no valid chunk could be selected. Using zeros for PREPROCESS_TARGET_SHAPE.")
                loaded_chunk_np = np.zeros(self.config.PREPROCESS_TARGET_SHAPE, dtype=np.float32)
        else:
            print(f"ERROR: Samplename '{samplename}' not found in pre-loaded dictionary! Using zeros for PREPROCESS_TARGET_SHAPE.")
            loaded_chunk_np = np.zeros(self.config.PREPROCESS_TARGET_SHAPE, dtype=np.float32)

        # Ensure loaded_chunk_np is float32
        loaded_chunk_np = loaded_chunk_np.astype(np.float32)

        # Apply manual SpecAugment (Time/Freq Mask, Contrast) on the single-channel NumPy chunk
        if self.mode == "train":
            loaded_chunk_np = self.apply_spec_augmentations(loaded_chunk_np)

        # Architecture-specific processing
        processed_spec_np = None
        if "efficientnet" in self.config.model_name.lower():
            # For EfficientNet, PREPROCESS_TARGET_SHAPE is often the same as TARGET_SHAPE
            if loaded_chunk_np.shape != self.config.TARGET_SHAPE:
                # This resize happens if PREPROCESS_TARGET_SHAPE (e.g. 5s chunk) is different from final TARGET_SHAPE
                print(f"INFO: EfficientNet - Resizing augmented chunk from {loaded_chunk_np.shape} to TARGET_SHAPE {self.config.TARGET_SHAPE} for {samplename}.")
                processed_spec_np = cv2.resize(loaded_chunk_np,
                                           (self.config.TARGET_SHAPE[1], self.config.TARGET_SHAPE[0]),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                processed_spec_np = loaded_chunk_np

            spec_tensor = torch.tensor(processed_spec_np, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
            spec_tensor = self.imagenet_normalize(spec_tensor)

        elif "mn" in self.config.model_name.lower():
            # MobileNet: loaded_chunk_np is PREPROCESS_TARGET_SHAPE (e.g., 5s chunk)
            # We concatenate to get TARGET_SHAPE (e.g., 10s)
            
            # Check if TARGET_SHAPE width is double PREPROCESS_TARGET_SHAPE width (common case for 5s -> 10s)
            if self.config.TARGET_SHAPE[1] == 2 * self.config.PREPROCESS_TARGET_SHAPE[1] and \
               self.config.TARGET_SHAPE[0] == self.config.PREPROCESS_TARGET_SHAPE[0]:
                processed_spec_np = np.concatenate([loaded_chunk_np, loaded_chunk_np], axis=1)
            else:
                # If not simple concatenation, assume direct resize or that PREPROCESS_TARGET_SHAPE is already TARGET_SHAPE
                processed_spec_np = loaded_chunk_np 

            # Ensure final processed_spec_np matches TARGET_SHAPE for MobileNet
            if processed_spec_np.shape != self.config.TARGET_SHAPE:
                print(f"INFO: MobileNet - Resizing processed chunk from {processed_spec_np.shape} to TARGET_SHAPE {self.config.TARGET_SHAPE} for {samplename}.")
                processed_spec_np = cv2.resize(processed_spec_np, 
                                           (self.config.TARGET_SHAPE[1], self.config.TARGET_SHAPE[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            
            spec_tensor = torch.tensor(processed_spec_np, dtype=torch.float32).unsqueeze(0) # (1, H, W_final)
        else:
            # This should have been caught in __init__, but as a safeguard:
            raise ValueError(f"Unsupported model_name: {self.config.model_name} during tensor creation.")

        # Encode labels
        target = self.encode_label(primary_label)
        parsed_secondary_labels = []
        if pd.notna(secondary_labels_str) and secondary_labels_str not in ['nan', '', '[]']:
            try: parsed_secondary_labels = eval(secondary_labels_str)
            except: pass 
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