import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

# Assuming 'config.py' is in the same directory or accessible via PYTHONPATH
from config import config # Import central config
# Optionally import utils if audio processing functions are moved there
# import utils

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class BirdCLEF2025Pipeline:
    """
    Pipeline for the BirdCLEF-2025 inference task.
    Organizes model loading, audio processing, prediction, and submission generation.
    """

    # Nested Model Class (uses config passed to pipeline)
    class BirdCLEFModel(nn.Module):
        """Internal model definition, mirrors training structure."""
        def __init__(self, config, num_classes):
            super().__init__()
            self.config = config # Store config if needed internally

            self.backbone = timm.create_model(
                config.model_name,
                pretrained=False, # Inference should load trained weights, not pretrained imagenet
                in_chans=config.in_channels,
                drop_rate=0.0, # Usually set drop rates to 0 for inference
                drop_path_rate=0.0
            )
            # Adjust final layers based on model type
            if 'efficientnet' in self.config.model_name:
                backbone_out = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            elif 'resnet' in self.config.model_name:
                backbone_out = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                backbone_out = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, '')
            
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.feat_dim = backbone_out
            self.classifier = nn.Linear(backbone_out, num_classes)

        def forward(self, x):
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features['features']
            # If features are 4D, apply global average pooling.
            if len(features.shape) == 4:
                features = self.pooling(features)
                features = features.view(features.size(0), -1)
            logits = self.classifier(features)
            return logits

    def __init__(self, config): # Accept central config
        """Initialize the inference pipeline with the central configuration."""
        self.config = config
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()

    def _load_taxonomy(self):
        """Load taxonomy data from CSV specified in config."""
        print("Loading taxonomy data...")
        try:
            self.taxonomy_df = pd.read_csv(self.config.taxonomy_path)
            self.species_ids = self.taxonomy_df['primary_label'].tolist()
            # Check against config num_classes
            if len(self.species_ids) != self.config.num_classes:
                print(f"Warning: Taxonomy indicates {len(self.species_ids)} species, but config has {self.config.num_classes}. Using {len(self.species_ids)}.")
            self.num_classes = len(self.species_ids)
            print(f"Number of classes: {self.num_classes}")
        except FileNotFoundError:
            print(f"Error: Taxonomy file not found at {self.config.taxonomy_path}")
            raise
        except Exception as e:
            print(f"Error loading taxonomy: {e}")
            raise

    def audio_to_melspec(self, audio_data):
        """Convert audio segment to mel spectrogram using config parameters."""
        if np.isnan(audio_data).any():
            mean_signal = np.nanmean(audio_data)
            audio_data = np.nan_to_num(audio_data, nan=mean_signal)

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.config.FS,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val > min_val:
            mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
        return mel_spec_norm

    def process_audio_segment(self, audio_data):
        """Process a 5-second audio segment for model input."""
        target_len = int(self.config.TARGET_DURATION * self.config.FS)
        if len(audio_data) < target_len:
            audio_data = np.pad(audio_data, (0, target_len - len(audio_data)), mode='constant')
        elif len(audio_data) > target_len:
            audio_data = audio_data[:target_len] # Ensure exact length

        mel_spec = self.audio_to_melspec(audio_data)

        # Resize if necessary to match TARGET_SHAPE
        if mel_spec.shape != self.config.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, self.config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)

    def find_model_files(self):
        """
        Find all .pth model files in the specified model directory.
        
        :return: List of model file paths.
        """
        model_files = []
        model_dir = Path(self.config.MODEL_INPUT_DIR)
        for path in model_dir.glob('**/*.pth'):
            model_files.append(str(path))
        return model_files

    def load_models(self):
        """
        Load all found model files and prepare them for ensemble inference.
        
        :return: List of loaded PyTorch models.
        """
        self.models = []
        model_files = self.find_model_files()
        if not model_files:
            print(f"Warning: No model files found under {self.config.MODEL_INPUT_DIR}!")
            return self.models

        print(f"Found a total of {len(model_files)} model files.")
        
        # If specific folds are required, filter the model files.
        if self.config.use_specific_folds_inference:
            filtered_files = []
            for fold in self.config.inference_folds:
                fold_pattern = f"_fold{fold}_best.pth"
                fold_files = [f for f in model_files if f.endswith(fold_pattern)]
                filtered_files.extend(fold_files)
            model_files = filtered_files
            print(f"Using {len(model_files)} model files for the specified inference folds ({self.config.inference_folds}).")
        
        # Load each model file.
        for model_path in model_files:
            try:
                print(f"Loading model: {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device(self.config.device))
                model = self.BirdCLEFModel(self.config, self.num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.config.device)
                model.eval()
                self.models.append(model)
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

        return self.models

    def apply_tta(self, spec, tta_idx):
        """Apply basic Test-Time Augmentation (flip)."""
        if tta_idx == 0:
            return spec
        elif tta_idx == 1:
            return np.flip(spec, axis=1).copy() # Horizontal flip
        elif tta_idx == 2:
            return np.flip(spec, axis=0).copy() # Vertical flip
        # Add more TTA types here if needed
        else:
            return spec

    def predict_on_audio_file(self, audio_path):
        """Predict on all 5-second segments of a single audio file."""
        all_predictions = []
        row_ids = []
        soundscape_id = Path(audio_path).name

        try:
            print(f"Processing {soundscape_id}...")
            # Load the entire audio file
            audio_data, _ = librosa.load(audio_path, sr=self.config.FS, mono=True)

            segment_length_samples = int(self.config.TARGET_DURATION * self.config.FS)
            total_duration_samples = len(audio_data)
            num_segments = total_duration_samples // segment_length_samples # Integer division for full segments

            if num_segments == 0:
                print(f"Warning: Audio file {audio_path} is shorter than target duration ({self.config.TARGET_DURATION}s). Skipping.")
                return [], []

            print(f"  Found {num_segments} segments.")
            segments_processed = 0
            
            # Process segments in batches for efficiency
            for i in range(0, num_segments, self.config.inference_batch_size):
                batch_segments = []
                batch_row_ids = []
                actual_batch_size = 0
                
                for segment_idx in range(i, min(i + self.config.inference_batch_size, num_segments)):
                    start_sample = segment_idx * segment_length_samples
                    end_sample = start_sample + segment_length_samples
                    segment_audio = audio_data[start_sample:end_sample]

                    end_time_sec = (segment_idx + 1) * self.config.TARGET_DURATION
                    row_id = f"{soundscape_id}_{int(end_time_sec)}" # Ensure integer end time
                    batch_row_ids.append(row_id)

                    # Process segment (convert to spectrogram)
                    mel_spec = self.process_audio_segment(segment_audio)
                    batch_segments.append(mel_spec)
                    actual_batch_size += 1

                if actual_batch_size == 0:
                    continue
                
                # Stack segments into a batch tensor
                batch_specs = torch.tensor(np.array(batch_segments), dtype=torch.float32).unsqueeze(1) # Add channel dim
                batch_specs = batch_specs.to(self.config.device)

                batch_final_preds = [] # Store predictions for this batch

                # --- TTA Loop (if enabled) --- #
                tta_iterations = self.config.tta_count if self.config.use_tta else 1
                all_tta_preds = [] # Store predictions across TTA iterations for this batch
                
                for tta_idx in range(tta_iterations):
                    batch_specs_tta = batch_specs.clone() # Start with original batch
                    # Apply TTA if not index 0
                    if tta_idx > 0:
                        # Apply TTA numpy function to each item in batch then restack
                        augmented_specs = [self.apply_tta(spec.squeeze().cpu().numpy(), tta_idx) for spec in batch_specs_tta]
                        batch_specs_tta = torch.tensor(np.array(augmented_specs), dtype=torch.float32).unsqueeze(1).to(self.config.device)
                
                    # --- Model Ensemble Loop --- #
                    batch_model_preds = [] # Store predictions from each model for this TTA batch
                    for model in self.models:
                        with torch.no_grad():
                            outputs = model(batch_specs_tta)
                            probs = torch.sigmoid(outputs)
                            batch_model_preds.append(probs.cpu().numpy())
                    
                    # Average predictions across models for this TTA iteration
                    avg_model_preds = np.mean(batch_model_preds, axis=0)
                    all_tta_preds.append(avg_model_preds)
                    
                # Average predictions across TTA iterations for the batch
                final_batch_preds = np.mean(all_tta_preds, axis=0)
                batch_final_preds.extend(final_batch_preds)

                # Append batch results to overall lists
                all_predictions.extend(batch_final_preds)
                row_ids.extend(batch_row_ids)
                segments_processed += actual_batch_size
                print(f"    Processed segments {segments_processed}/{num_segments}", end='\r')

            print(f"\n  Finished processing {soundscape_id}.")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Optionally return partial results or empty lists
            return [], []

        return row_ids, all_predictions

    def run_inference(self):
        """
        Run inference on all test soundscape audio files.
        
        :return: Tuple (all_row_ids, all_predictions) aggregated from all files.
        """
        test_files = list(Path(self.config.unlabeled_audio_dir).glob('*.ogg'))
        if self.config.debug:
            print(f"Debug mode enabled, using only {self.config.debug_limit_files} files")
            test_files = test_files[:self.config.debug_limit_files]
        print(f"Found {len(test_files)} test soundscapes")

        all_row_ids = []
        all_predictions = []

        for audio_path in tqdm(test_files, desc="Inferring on Test Set"):
            row_ids, predictions = self.predict_on_audio_file(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions

    def create_pseudo_labels_df(self, row_ids, predictions):
        """
        Create the pseudo-labels dataframe based on predictions.
        Applies confidence thresholding.

        :param row_ids: List of row identifiers for each segment (filename_endtime).
        :param predictions: List of prediction arrays (probabilities).
        :return: A pandas DataFrame containing high-confidence pseudo-labels.
        """
        print("Creating pseudo-labels dataframe...")
        pseudo_labels_list = []

        # Map species index to species name
        idx_to_species = {i: name for i, name in enumerate(self.species_ids)}

        for row_id, pred_probs in zip(row_ids, predictions):
            # Extract filename and end_time from row_id
            try:
                filename, end_time_str = row_id.rsplit('_', 1)
                end_time = int(end_time_str)
                start_time = end_time - self.config.TARGET_DURATION
            except ValueError:
                print(f"Warning: Could not parse row_id '{row_id}'. Skipping.")
                continue

            # Find predictions above threshold
            confident_indices = np.where(pred_probs >= self.config.threshold)[0]

            for idx in confident_indices:
                species = idx_to_species[idx]
                confidence = pred_probs[idx]
                pseudo_labels_list.append({
                    'filename': filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'primary_label': species,
                    'confidence': confidence
                })

        if not pseudo_labels_list:
            print("Warning: No predictions found above the threshold.")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['filename', 'start_time', 'end_time', 'primary_label', 'confidence'])
        
        pseudo_labels_df = pd.DataFrame(pseudo_labels_list)
        print(f"Generated {len(pseudo_labels_df)} pseudo-labels above threshold {self.config.threshold}")

        return pseudo_labels_df

    def run(self):
        """Main method to execute the full pseudo-label generation pipeline."""
        start_time = time.time()
        print("\n--- Starting BirdCLEF-2025 Pseudo-Label Generation Pipeline ---") # Updated print message
        print(f"Using Device: {self.config.device}")
        print(f"Debug Mode: {self.config.debug}")
        print(f"TTA Enabled: {self.config.use_tta} (Variations: {self.config.tta_count if self.config.use_tta else 1})")
        print(f"Model Input Directory: {self.config.MODEL_INPUT_DIR}")
        print(f"Unlabeled Audio Directory: {self.config.unlabeled_audio_dir}") # Added info
        print(f"Confidence Threshold: {self.config.threshold}") # Added info

        # 1. Load Models
        self.load_models()
        if not self.models:
            print("Error: No models loaded! Please check model paths and fold configuration. Exiting.")
            return
        print(f"Model usage: {'Single model' if len(self.models) == 1 else f'Ensemble of {len(self.models)} models'}")

        # 2. Run Inference on Unlabeled Data
        row_ids, predictions = self.run_inference()
        if not row_ids: # Check if inference produced results
             print("Inference did not produce any results. Exiting.")
             return

        # 3. Create Pseudo-Labels DataFrame (replaces create_submission)
        pseudo_labels_df = self.create_pseudo_labels_df(row_ids, predictions)
        
        # 4. Save Pseudo-Labels
        pseudo_labels_path = self.config.train_pseudo_csv_path # Save directly to the path expected by preprocessing/training
        output_dir_for_pseudo = os.path.dirname(pseudo_labels_path)
        os.makedirs(output_dir_for_pseudo, exist_ok=True) # Ensure the specific directory for the CSV exists
        pseudo_labels_df.to_csv(pseudo_labels_path, index=False)
        print(f"Pseudo-labels saved to {pseudo_labels_path}")
        
        end_time = time.time()
        print(f"\nPseudo-label generation pipeline completed in {(end_time - start_time) / 60:.2f} minutes")

# Run the BirdCLEF2025 Pipeline:
if __name__ == "__main__":
    print(f"Initializing pipeline with device: {config.device}")
    pipeline = BirdCLEF2025Pipeline(config) # Pass the imported config
    pipeline.run()