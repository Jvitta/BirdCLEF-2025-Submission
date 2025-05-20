import os
import sys
import pandas as pd
import numpy as np
import librosa
from tqdm.auto import tqdm
import tensorflow as tf # For TensorFlow Hub model
from pathlib import Path # Added for recursive file search

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This should be 'BirdCLEF-2025-Submission/preprocess'
project_root = os.path.dirname(project_root) # This should be 'BirdCLEF-2025-Submission'
sys.path.append(project_root)
# No longer need to add perch submodule to path

from config import config as project_config
import bioacoustics_model_zoo as bmz # Use the new library

# --- Configuration --- 
PERCH_MODEL_VERSION = 8
TARGET_SAMPLE_RATE = 32000  # Hz, as required by Perch
SEGMENT_DURATION_S = 5.0    # seconds, as required by Perch
# TODO: Decide on overlap for sliding window analysis if desired
WINDOW_OVERLAP_S = 2.0 # Example: 2s overlap for 3s step, adjust as needed
EFFECTIVE_STEP_S = SEGMENT_DURATION_S - WINDOW_OVERLAP_S

# TODO: Define output paths for detections from training audio and soundscapes
OUTPUT_TRAIN_DETECTIONS_CSV = os.path.join(project_config.RAW_DATA_DIR, f'perch_v{PERCH_MODEL_VERSION}_train_detections.csv')
OUTPUT_SOUNDSCAPE_DETECTIONS_CSV = os.path.join(project_config.RAW_DATA_DIR, f'perch_v{PERCH_MODEL_VERSION}_soundscape_detections.csv')

def load_perch_model_and_labels(model_version=PERCH_MODEL_VERSION):
    """Loads the Perch model and its species label mapping using bioacoustics-model-zoo."""
    print(f"Loading Perch model (via bioacoustics-model-zoo) version: {model_version}...")
    try:
        model = bmz.Perch(version=model_version)
        print("Perch model loaded successfully.")
    except Exception as e:
        print(f"Error loading Perch model via bioacoustics-model-zoo: {e}")
        print("Ensure tensorflow, tensorflow_hub, and opensoundscape are installed correctly.")
        # import traceback
        # traceback.print_exc()
        return None, None

    labels_list = model.taxonomic_classes["species"] # This is a numpy array of species codes/names
    labels_df = pd.DataFrame({'species_code': labels_list, 'index': range(len(labels_list))})
    print(f"Loaded {len(labels_df)} species labels from bioacoustics-model-zoo Perch wrapper.")
    
    return model, labels_df

def process_audio_file(filepath, model, labels_df, sr=TARGET_SAMPLE_RATE, 
                       segment_s=SEGMENT_DURATION_S, overlap_s=WINDOW_OVERLAP_S):
    """Processes a single audio file, extracts Perch detections."""
    detections = []
    try:
        waveform, original_sr = librosa.load(filepath, sr=None, mono=True)
        if original_sr != sr:
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=sr)
        
        effective_step_s = segment_s - overlap_s
        if effective_step_s <= 0:
            raise ValueError("Overlap cannot be greater than or equal to segment duration.")

        samples_per_segment = int(segment_s * sr)
        samples_per_step = int(effective_step_s * sr)

        for i in tqdm(range(0, len(waveform) - samples_per_segment + 1, samples_per_step), 
                      desc=f"Processing {os.path.basename(filepath)}", leave=False):
            chunk = waveform[i : i + samples_per_segment]
            
            if len(chunk) < samples_per_segment: # Pad last chunk if necessary
                padding = np.zeros(samples_per_segment - len(chunk), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, padding])
            
            # The _batch_forward method expects a batch of samples.
            # Input shape: (batch_size, num_samples)
            chunk_batch = np.array(chunk[np.newaxis, :], dtype=np.float32)
            
            # Use _batch_forward for direct waveform input
            # It returns a dictionary of outputs
            model_outputs_dict = model._batch_forward(chunk_batch, return_dict=True)
            logits = model_outputs_dict['label'] # Shape: (batch_size, num_classes)
            
            if labels_df is not None and not labels_df.empty:
                for species_idx in range(logits.shape[1]):
                    logit_value = logits[0, species_idx]
                    # TODO: Implement proper thresholding here based on future calibration
                    if logit_value > -10: # Placeholder, effectively saving many. Sigmoid for prob if needed.
                        try:
                            species_code = labels_df.loc[labels_df['index'] == species_idx, 'species_code'].iloc[0]
                        except (IndexError, KeyError):
                            species_code = f"unknown_species_idx_{species_idx}"
                        
                        start_time_s = i / sr
                        end_time_s = start_time_s + segment_s
                        detections.append({
                            'filename': os.path.basename(filepath),
                            'start_time': round(start_time_s, 3),
                            'end_time': round(end_time_s, 3),
                            'primary_label': species_code,
                            'confidence': round(float(logit_value), 4) # Storing raw logit
                        })
            else:
                # Fallback if labels_df is somehow not available (should not happen with bmz.Perch)
                for species_idx in range(logits.shape[1]):
                    logit_value = logits[0, species_idx]
                    if logit_value > -10:
                        start_time_s = i / sr
                        end_time_s = start_time_s + segment_s
                        detections.append({
                            'filename': os.path.basename(filepath),
                            'start_time': round(start_time_s, 3),
                            'end_time': round(end_time_s, 3),
                            'primary_label': f"logit_idx_{species_idx}",
                            'confidence': round(float(logit_value), 4)
                        })
                        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        # import traceback
        # traceback.print_exc()
    return detections

def run_perch_on_dataset(audio_dir, output_csv_path, model, labels_df, limit_files=None):
    """Runs Perch inference on all OGG files in a directory and its subdirectories, and saves results."""
    print(f"Starting Perch inference for audio in: {audio_dir}")
    
    audio_path = Path(audio_dir)
    audio_files = list(audio_path.glob('**/*.ogg')) # Recursively find all .ogg files
    
    if limit_files is not None:
        audio_files = audio_files[:limit_files]
        print(f"Limited to {limit_files} files.")

    if not audio_files:
        print(f"No .ogg audio files found in {audio_dir} or its subdirectories. Skipping.")
        return

    print(f"Found {len(audio_files)} .ogg files to process.")

    all_detections = []
    for audio_file in tqdm(audio_files, desc="Processing dataset"):
        file_detections = process_audio_file(audio_file, model, labels_df)
        all_detections.extend(file_detections)

    if all_detections:
        detections_df = pd.DataFrame(all_detections)
        try:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            detections_df.to_csv(output_csv_path, index=False)
            print(f"Perch detections saved to: {output_csv_path} ({len(detections_df)} rows)")
        except Exception as e:
            print(f"Error saving Perch detections CSV: {e}")
    else:
        print("No Perch detections generated.")

if __name__ == '__main__':
    print("Starting Perch Inference Script (using bioacoustics-model-zoo)")
    
    perch_model, species_labels_df = load_perch_model_and_labels()

    if perch_model is None or species_labels_df is None:
        print("Failed to load Perch model or labels. Exiting.")
        sys.exit(1)
    
    # Print a few sample labels to verify
    print("\nSample of loaded species labels:")
    print(species_labels_df.head())
    print(f"Total labels: {len(species_labels_df)}")

    # --- Example: Process Training Audio Data ---
    print("\n--- Processing Training Audio (Example) ---")
    run_perch_on_dataset(
        audio_dir=project_config.train_audio_dir, 
        output_csv_path=OUTPUT_TRAIN_DETECTIONS_CSV, 
        model=perch_model, 
        labels_df=species_labels_df,
        limit_files=2 # For quick testing
    )

    # --- Example: Process Soundscape (Unlabeled) Audio Data ---
    # print("\n--- Processing Soundscape Audio (Example) ---")
    # run_perch_on_dataset(
    #     audio_dir=project_config.unlabeled_audio_dir, 
    #     output_csv_path=OUTPUT_SOUNDSCAPE_DETECTIONS_CSV, 
    #     model=perch_model, 
    #     labels_df=species_labels_df,
    #     limit_files=2 # For quick testing
    # )

    print("\nPerch Inference Script Finished.")
    # print("Remember to uncomment and configure the dataset processing calls in main if you want to run inference.")

    # As a quick test, let's try to print model's expected input signature and output signature if available
    try:
        if hasattr(perch_model, 'model_tf') and hasattr(perch_model.model_tf, 'signatures'):
            print("\nModel Signatures:")
            print(perch_model.model_tf.signatures)
        
        # For label_infos (from TaxonomyModelTF.load_version)
        if hasattr(perch_model, 'label_infos') and perch_model.label_infos:
            print(f"\nLoaded {len(perch_model.label_infos)} label_infos from model.")
            # print a few labels
            count = 0
            for k, v in perch_model.label_infos.items():
                print(f"  Label: {k}, Common Name: {v.common_name}, Index: {v.index}")
                count +=1
                if count >= 5: break
        else:
            print("\nModel does not have a direct 'label_infos' attribute as expected for TaxonomyModelTF.")

    except Exception as e:
        print(f"Error inspecting model details: {e}") 