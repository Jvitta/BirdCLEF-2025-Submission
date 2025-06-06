import sys
import os

models_package_parent_dir = '/kaggle/input'
if models_package_parent_dir not in sys.path:
    sys.path.insert(0, models_package_parent_dir)
    print(f"Added {models_package_parent_dir} to sys.path to locate the 'models' package in /kaggle/input/models/")

import os
import gc
import warnings
import logging
import time
import math
import random
import cv2 
from pathlib import Path
import multiprocessing
from functools import partial
import traceback
import matplotlib.pyplot as plt # Added for visualization

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import config # Import central config
import birdclef_utils as utils # Import utils
from birdclef_utils import _preprocess_audio_file_worker # Import the worker

# EfficientAT model and preprocessing
from models.efficient_at.mn.model import get_model as get_efficient_at_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class BirdCLEF2025Pipeline:
    """
    Pipeline for the BirdCLEF-2025 inference task.
    Organizes model loading, audio processing, prediction, and submission generation.
    """

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

    def find_model_files(self):
        """
        Find all .pth model files in the specified model directory.
        
        :return: List of model file paths.
        """
        model_files = []
        print(f"Looking for Model files in {self.config.MODEL_INPUT_DIR}")
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
                
                model = get_efficient_at_model(
                    num_classes=self.config.num_classes, # Use self.config
                    pretrained_name=None,               # Pass None to prevent loading base weights from URL
                    width_mult=2.0,                     # Explicitly set for mn20_as like architecture
                    head_type="mlp",                    # Explicitly set for mn20_as like architecture
                    input_dim_f=self.config.TARGET_SHAPE[0],  # Use self.config
                    input_dim_t=self.config.TARGET_SHAPE[1]   # Use self.config
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.config.device)
                model.eval()
                self.models.append(model)
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

        return self.models

    def apply_tta(self, spec, tta_idx):
        """Apply Test-Time Augmentation.
        tta_idx=0: Original
        tta_idx > 0: Random time shift within +/- max_shift_ratio.
        """
        if tta_idx == 0:
            return spec # Return original for index 0
        else:
            # Apply random time shift for other indices
            height, width = spec.shape
            max_shift_ratio = 0.15 # Max shift percentage (e.g., 15%)
            
            # Generate random shift amount (-max_shift to +max_shift)
            shift_ratio = random.uniform(-max_shift_ratio, max_shift_ratio)
            shift_pixels = int(width * shift_ratio)

            if shift_pixels == 0:
                return spec # No shift
            
            if shift_pixels > 0: # Positive shift -> Shift content right (pad left)
                padded = np.pad(spec, ((0, 0), (shift_pixels, 0)), mode='reflect')
                return padded[:, :width] # Crop from left
            else: # Negative shift -> Shift content left (pad right)
                shift_pixels = abs(shift_pixels)
                padded = np.pad(spec, ((0, 0), (0, shift_pixels)), mode='reflect')
                return padded[:, -width:] # Crop from right

    def run_inference(self):
        """
        Run inference: Preprocess all files in parallel, then predict in batches.
        """
        if not self.config.debug:
            test_files = list(Path(self.config.test_audio_dir).glob('*.ogg'))
        else:
            test_files = list(Path(self.config.unlabeled_audio_dir).glob('*.ogg'))
            print(f"Debug mode enabled, using only {self.config.debug_limit_files} files from unlabeled_audio_dir for testing inference pipeline.")
            test_files = test_files[:self.config.debug_limit_files]
        print(f"Found {len(test_files)} test soundscapes")

        # --- Stage 1: Parallel Preprocessing --- 
        print("Starting parallel preprocessing of audio files...")
        start_preprocess = time.time()
        all_row_ids = []
        all_specs_list = []
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_workers} workers for preprocessing.")
        
        # Use functools.partial to pass the config to the *imported* worker function
        # Ensure the function name matches the one imported from utils
        worker_func = partial(_preprocess_audio_file_worker, config=self.config)
        
        pool = None
        try:
            pool = multiprocessing.Pool(processes=num_workers)
            # Process files and collect results
            results_iterator = pool.imap_unordered(worker_func, [str(p) for p in test_files])
            
            for result in tqdm(results_iterator, total=len(test_files), desc="Preprocessing Files"):
                row_ids_part, specs_part = result
                if row_ids_part: # Only extend if results are not empty (error handling in worker)
                    all_row_ids.extend(row_ids_part)
                    all_specs_list.extend(specs_part)
        except Exception as e:
            print(f"\nCRITICAL ERROR during multiprocessing: {e}")
            print(traceback.format_exc())
            # Decide how to proceed: exit or try inference on partial data?
            print("Exiting due to multiprocessing error.")
            if pool: pool.terminate()
            return [], []
        finally:
            if pool:
                pool.close()
                pool.join()
        
        specs_to_visualize = all_specs_list[:12] # Ensure we don't exceed list length

        # --- Visualize Spectrograms ---
        if specs_to_visualize:
            visualization_output_dir = "visualized_spectrograms"
            os.makedirs(visualization_output_dir, exist_ok=True)
            print(f"Saving {len(specs_to_visualize)} spectrograms to '{visualization_output_dir}'...")
            for idx, spec_to_vis in enumerate(specs_to_visualize):
                try:
                    file_path = os.path.join(visualization_output_dir, f"spec_{idx}.png")
                    # Ensure spectrogram is 2D (remove channel if it was added, though worker returns 2D)
                    if spec_to_vis.ndim == 3 and spec_to_vis.shape[0] == 1:
                        spec_to_vis_2d = spec_to_vis.squeeze(0)
                    else:
                        spec_to_vis_2d = spec_to_vis
                    
                    plt.imsave(file_path, spec_to_vis_2d, cmap='gray', origin='lower')
                except Exception as e:
                    print(f"Could not save spectrogram {idx}: {e}")
            print(f"Finished saving spectrograms.")
        # --- End Visualize Spectrograms ---

        end_preprocess = time.time()
        print(f"Preprocessing finished in {end_preprocess - start_preprocess:.2f} seconds.")
        print(f"Total spectrogram segments generated: {len(all_specs_list)}")

        if not all_specs_list:
            print("No spectrograms were generated. Exiting inference.")
            return [], []

        # Consolidate into NumPy array (potential memory bottleneck)
        try:
            print("Consolidating spectrograms into NumPy array...")
            all_specs_np = np.array(all_specs_list)
            del all_specs_list # Free up memory
            gc.collect()
            print(f"Consolidated array shape: {all_specs_np.shape}")
        except MemoryError:
            print("MemoryError: Could not create NumPy array of all spectrograms. Not enough RAM.")
            print("Consider reducing NUM_EXAMPLES_TO_VISUALIZE in debug mode or using a machine with more RAM.")
            return [], []
        except Exception as e:
            print(f"Error converting spectrogram list to NumPy array: {e}")
            return [], []

        # --- Stage 2: Batched Inference --- 
        print("Starting batched inference...")
        start_inference = time.time()
        all_predictions = []
        # Define normalization transform once -- REMOVE THIS, NOT NEEDED FOR EFFICIENTAT
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        num_total_specs = all_specs_np.shape[0]

        for i in tqdm(range(0, num_total_specs, self.config.inference_batch_size), desc="Inference Batches"):
            batch_specs_np = all_specs_np[i : i + self.config.inference_batch_size]
            
            # --- TTA Logic --- 
            tta_iterations = self.config.tta_count if self.config.use_tta else 1
            batch_tta_preds = [] # Store preds for each TTA variant for this batch
            for tta_idx in range(tta_iterations):
                # Apply TTA to the numpy batch
                tta_batch_np = np.array([self.apply_tta(spec, tta_idx) for spec in batch_specs_np])
                
                # Convert TTA'd batch to tensor for EfficientAT
                # Spectrograms from process_audio_segment are (N_MELS, 1000)
                batch_tensor = torch.tensor(tta_batch_np, dtype=torch.float32)
                # Add channel dimension: (Batch, Channels, Mels, Time) -> (B, 1, N_MELS, 1000)
                batch_tensor = batch_tensor.unsqueeze(1) 
                batch_tensor = batch_tensor.to(self.config.device)
                
                # --- Ensemble Logic --- 
                batch_model_preds = []
                for model in self.models:
                    with torch.no_grad():
                        outputs = model(batch_tensor) # model returns a tuple (logits, features)
                        logits = outputs[0] # Select the logits
                        probs = torch.sigmoid(logits) # Apply sigmoid to logits
                        batch_model_preds.append(probs.cpu().numpy())
                         
                # Average predictions across models for this TTA iteration
                avg_model_preds = np.mean(batch_model_preds, axis=0)
                batch_tta_preds.append(avg_model_preds)
                
            # Average predictions across TTA iterations for the final batch prediction
            final_batch_preds = np.mean(batch_tta_preds, axis=0)
            all_predictions.extend(final_batch_preds)
            
        end_inference = time.time()
        print(f"Inference finished in {end_inference - start_inference:.2f} seconds.")

        # Ensure prediction count matches row_id count
        if len(all_predictions) != len(all_row_ids):
            print(f"CRITICAL WARNING: Mismatch between number of predictions ({len(all_predictions)}) and row_ids ({len(all_row_ids)}). Submission will likely fail.")
            # Handle this error - maybe return empty? 
            return [], []

        return all_row_ids, all_predictions

    def create_submission(self, row_ids, predictions):
        """
        Create the submission dataframe based on predictions.
        
        :param row_ids: List of row identifiers for each segment.
        :param predictions: List of prediction arrays.
        :return: A pandas DataFrame formatted for submission.
        """
        print("Creating submission dataframe...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(self.species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        sample_sub = pd.read_csv(self.config.sample_submission_path, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} species columns in submission")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns]
        submission_df = submission_df.reset_index()
        
        return submission_df

    def smooth_submission(self, submission_path):
        """
        Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
        
        For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
        are averaged with those of its neighbors using defined weights.
        
        :param submission_path: Path to the submission CSV file.
        """
        print("Smoothing submission predictions...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        # Extract group names by splitting row_id on the last underscore
        groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            # Get indices for the current group
            idx = np.where(groups == group)[0]
            sub_group = sub.iloc[idx].copy()
            predictions = sub_group[cols].values
            new_predictions = predictions.copy()
            
            if predictions.shape[0] > 1:
                # Smooth the predictions using neighboring segments
                neighbor_weight = 0.0
                center_weight = 1.0 - (2 * neighbor_weight)
                edge_weight = 1.0 - neighbor_weight

                new_predictions[0] = (predictions[0] * edge_weight) + (predictions[1] * neighbor_weight)
                new_predictions[-1] = (predictions[-1] * edge_weight) + (predictions[-2] * neighbor_weight)
                for i in range(1, predictions.shape[0]-1):
                    new_predictions[i] = (predictions[i-1] * neighbor_weight) + (predictions[i] * center_weight) + (predictions[i+1] * neighbor_weight)
            # Replace the smoothed values in the submission dataframe
            sub.iloc[idx, 1:] = new_predictions
        
        sub.to_csv(submission_path, index=False)
        print(f"Smoothed submission saved to {submission_path}")

    def run(self):
        """Main method to execute the full inference pipeline."""
        start_time = time.time()
        print("\n--- Starting BirdCLEF-2025 Inference Pipeline (Parallel Preprocessing) ---")
        print(f"Using Device: {self.config.device}")
        print(f"Debug Mode: {self.config.debug}")
        print(f"TTA Enabled: {self.config.use_tta} (Variations: {self.config.tta_count if self.config.use_tta else 1})")
        print(f"Model Input Directory: {self.config.MODEL_INPUT_DIR}")

        # 1. Load Models
        self.load_models()
        if not self.models:
            print("Error: No models loaded! Exiting.")
            return
        print(f"Model usage: {'Single model' if len(self.models) == 1 else f'Ensemble of {len(self.models)} models'}")
        
        # 2. Run Inference (Now includes parallel preprocessing)
        row_ids, predictions = self.run_inference()
    
        # 3. Create Submission
        submission_df = self.create_submission(row_ids, predictions)
        
        # 4. Save and Smooth
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Initial submission saved to {submission_path}")
        
        # Apply smoothing on the submission predictions.
        self.smooth_submission(submission_path)
        
        end_time = time.time()
        print(f"\nInference pipeline completed in {(end_time - start_time) / 60:.2f} minutes")

# Run the BirdCLEF2025 Pipeline:
if __name__ == "__main__":
    # Force 'spawn' start method for multiprocessing in interactive environments
    # This often prevents hangs/deadlocks.
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # Might already be set or not applicable
        print("Could not set multiprocessing start method (might be already set or unsupported).")
        pass 

    print(f"Initializing pipeline with device: {config.device}")
    pipeline = BirdCLEF2025Pipeline(config) # Pass the imported config
    pipeline.run()