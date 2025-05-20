import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
import pickle
import random   
import math   
import multiprocessing
import traceback  
import cv2         
import librosa      
from tqdm.auto import tqdm 
import torch

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config as global_config # Use a distinct name to avoid conflict if config is passed as arg
from src.models.efficient_at.preprocess import AugmentMelSTFT
import src.utils.utils as utils_audio 

warnings.filterwarnings("ignore")

# --- Seed setting --- #
random.seed(global_config.seed)
np.random.seed(global_config.seed)
torch.manual_seed(global_config.seed)

# --- Output Paths (now derived from global_config) --- #
# The primary NPZ output path is directly from config
OUTPUT_NPZ_PATH = global_config.SOUNDSCAPE_VAL_NPZ_PATH 
# Derive metadata CSV path from the NPZ path to keep them associated
output_npz_basename = os.path.basename(OUTPUT_NPZ_PATH)
output_npz_dirname = os.path.dirname(OUTPUT_NPZ_PATH)
OUTPUT_METADATA_FILENAME = output_npz_basename.replace(".npz", "_metadata.csv")
OUTPUT_METADATA_PATH = os.path.join(output_npz_dirname, OUTPUT_METADATA_FILENAME)


# --- Global Spectrogram Generator (for EfficientAT models) --- #
efficient_at_spectrogram_generator = None
IS_EFFICIENT_AT_MODEL = 'mn' in global_config.model_name.lower()

if IS_EFFICIENT_AT_MODEL:
    print(f"--- Initializing Spectrogram Generator for EfficientAT model: {global_config.model_name} (fixed settings) ---")
    efficient_at_spectrogram_generator = AugmentMelSTFT(
            n_mels=global_config.N_MELS,
            sr=global_config.FS,
            win_length=global_config.WIN_LENGTH,
            hopsize=global_config.HOP_LENGTH,
            n_fft=global_config.N_FFT,
            fmin=global_config.FMIN,
            fmax=global_config.FMAX, 
            freqm=0,       
            timem=0,         
            fmin_aug_range=global_config.FMIN_AUG_RANGE,
            fmax_aug_range=global_config.FMAX_AUG_RANGE
        )
    efficient_at_spectrogram_generator.eval()
else:
    print(f"--- Using EfficientNet-style spectrogram generation via utils.audio2melspec for model: {global_config.model_name} ---")

def _ensure_fixed_length_audio(audio_data, target_samples):
    """Ensures audio_data is target_samples long. Pads with zeros if shorter, truncates if longer."""
    current_len = len(audio_data)
    if current_len == target_samples:
        return audio_data
    elif current_len < target_samples:
        padding = np.zeros(target_samples - current_len, dtype=audio_data.dtype)
        return np.concatenate((audio_data, padding))
    else: # current_len > target_samples
        return audio_data[:target_samples]


def _extract_centered_chunk_from_audio(audio_long, center_time_s, target_duration_s, sr, min_audio_len_for_extraction):
    """
    Extracts a chunk of target_duration_s centered at center_time_s.
    Shifts window to fit if out of bounds. Assumes audio_long is generally longer than target_duration_s.
    Returns a chunk of exactly target_samples, padding only if audio_long itself was exceptionally short.
    Returns None if audio_long is shorter than min_audio_len_for_extraction.
    """
    audio_len_samples = len(audio_long)
    target_samples = int(target_duration_s * sr)

    if audio_len_samples < min_audio_len_for_extraction:
        return None 

    chunk_center_samples = int(center_time_s * sr)
    start_idx = chunk_center_samples - (target_samples // 2)
    end_idx = start_idx + target_samples

    if start_idx < 0:
        start_idx = 0
        end_idx = target_samples 
    elif end_idx > audio_len_samples:
        end_idx = audio_len_samples
        start_idx = audio_len_samples - target_samples
        if start_idx < 0: 
            start_idx = 0

    extracted_audio_segment = audio_long[start_idx:end_idx]
    return _ensure_fixed_length_audio(extracted_audio_segment, target_samples)


def _generate_spectrogram_from_chunk_val(audio_chunk, passed_config_obj, efficient_at_generator_ref):
    """Generates a mel spectrogram from an audio chunk, then resizes."""
    try:
        if not isinstance(audio_chunk, np.ndarray) or audio_chunk.ndim != 1:
            return None
        
        raw_spec_chunk_numpy = None
        is_eff_at_model_local = 'mn' in passed_config_obj.model_name.lower()

        if is_eff_at_model_local:
            if efficient_at_generator_ref is None: return None # Should have been initialized globally
            audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
            with torch.no_grad():
                raw_spec_chunk_tensor = efficient_at_generator_ref(audio_tensor.unsqueeze(0))
            raw_spec_chunk_numpy = raw_spec_chunk_tensor.squeeze(0).cpu().numpy()
        elif 'efficientnet' in passed_config_obj.model_name.lower():
            raw_spec_chunk_numpy = utils_audio.audio2melspec(audio_chunk, passed_config_obj)
        else: return None

        if raw_spec_chunk_numpy is None or raw_spec_chunk_numpy.ndim != 2: return None

        current_shape = (raw_spec_chunk_numpy.shape[0], raw_spec_chunk_numpy.shape[1])
        desired_shape = passed_config_obj.PREPROCESS_TARGET_SHAPE
        if current_shape != desired_shape:
            raw_spec_chunk_numpy = cv2.resize(
                raw_spec_chunk_numpy, (desired_shape[1], desired_shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return raw_spec_chunk_numpy.astype(np.float32)
    except Exception: return None

def process_detection_worker(args):
    detection_id, filename, primary_label, start_time, end_time, worker_config_obj, soundscape_audio_dir_resolved, efficient_at_gen_glob_ref = args
    
    audio_filepath = os.path.join(soundscape_audio_dir_resolved, filename)
    min_audio_samples_for_chunking = int(0.1 * worker_config_obj.FS)

    try:
        if not os.path.exists(audio_filepath):
            return detection_id, None, f"Audio file not found: {audio_filepath}"

        audio_data, _ = librosa.load(audio_filepath, sr=worker_config_obj.FS, mono=True)
        if audio_data is None or len(audio_data) == 0:
            return detection_id, None, f"Audio loaded as None or empty: {audio_filepath}"

        center_s = (start_time + end_time) / 2.0
        
        audio_chunk = _extract_centered_chunk_from_audio(
            audio_data, center_s, worker_config_obj.TARGET_DURATION, 
            worker_config_obj.FS, min_audio_samples_for_chunking 
        )

        if audio_chunk is None:
            return detection_id, None, f"Failed to extract valid audio chunk from {filename} centered at {center_s:.2f}s."

        spectrogram = _generate_spectrogram_from_chunk_val(audio_chunk, worker_config_obj, efficient_at_gen_glob_ref)

        if spectrogram is None: return detection_id, None, f"Spectrogram generation failed for chunk from {filename}."
        
        return detection_id, spectrogram, None
    except Exception as e: return detection_id, None, f"Error processing {filename}: {str(e)}"


def main(cfg):
    overall_start_time = time.time()
    print(f"--- Starting Soundscape Validation Set Preprocessing ---")
    print(f"Using model type from config: {cfg.model_name} (Is EfficientAT type: {IS_EFFICIENT_AT_MODEL})")
    print(f"Target spectrogram shape: {cfg.PREPROCESS_TARGET_SHAPE}")
    print(f"Target audio duration for chunks: {cfg.TARGET_DURATION}s")

    soundscape_audio_dir = cfg.unlabeled_audio_dir
    if soundscape_audio_dir is None or not os.path.isdir(soundscape_audio_dir):
        print(f"Warning: cfg.SOUNDSCAPE_RAW_AUDIO_DIR not set or not a valid directory. Trying cfg.train_audio_dir...")
        soundscape_audio_dir = cfg.train_audio_dir
        if not os.path.isdir(soundscape_audio_dir):
            print(f"CRITICAL: Fallback cfg.train_audio_dir ('{soundscape_audio_dir}') also not valid. Please set cfg.SOUNDSCAPE_RAW_AUDIO_DIR.")
            return
    print(f"Using soundscape audio directory: {soundscape_audio_dir}")

    try:
        df_soundscape_all = pd.read_csv(cfg.soundscape_pseudo_calibrated_csv_path)
        required_cols = ['filename', 'primary_label', 'start_time', 'end_time']
        if not all(col in df_soundscape_all.columns for col in required_cols):
            print(f"Error: Soundscape CSV {cfg.soundscape_pseudo_calibrated_csv_path} missing one of {required_cols}.")
            return
    except FileNotFoundError:
        print(f"Error: Soundscape pseudo-calibrated CSV not found at {cfg.soundscape_pseudo_calibrated_csv_path}")
        return
    except Exception as e:
        print(f"Error loading {cfg.soundscape_pseudo_calibrated_csv_path}: {e}")
        return
    print(f"Loaded {len(df_soundscape_all)} total detections from pseudo-calibrated soundscapes.")

    val_filenames_path = os.path.join(cfg.PROCESSED_DATA_DIR, "fixed_soundscape_validation_filenames.txt")
    try:
        with open(val_filenames_path, 'r') as f:
            val_filenames = {line.strip() for line in f if line.strip()}
        if not val_filenames: print(f"Error: Validation filenames list at {val_filenames_path} is empty."); return
    except FileNotFoundError: print(f"Error: Validation filenames file not found at {val_filenames_path}"); return
    print(f"Loaded {len(val_filenames)} unique filenames for the validation set.")

    df_val_detections = df_soundscape_all[df_soundscape_all['filename'].isin(val_filenames)].copy()
    if df_val_detections.empty: print("No detections found for the filenames specified in the validation set. Exiting."); return
    
    df_val_detections['detection_id'] = df_val_detections['filename'] + "_idx_" + df_val_detections.index.astype(str)
    print(f"Filtered to {len(df_val_detections)} detections for validation set processing.")

    tasks = [
        (row['detection_id'], row['filename'], row['primary_label'], row['start_time'], row['end_time'],
         cfg, soundscape_audio_dir, efficient_at_spectrogram_generator if IS_EFFICIENT_AT_MODEL else None)
        for _, row in df_val_detections.iterrows()
    ]
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting parallel processing with {num_workers} workers for {len(tasks)} detections...")
    
    processed_spectrograms = {}
    processed_metadata = []
    error_count = 0
    error_messages = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(process_detection_worker, tasks)
        for det_id, spec_array, error_msg in tqdm(results_iterator, total=len(tasks), desc="Processing Val Detections"):
            if error_msg:
                error_count += 1; error_messages.append(f"{det_id}: {error_msg}")
            elif spec_array is not None:
                processed_spectrograms[det_id] = spec_array
                original_row = df_val_detections[df_val_detections['detection_id'] == det_id].iloc[0]
                processed_metadata.append({
                    'detection_id': det_id, 'original_filename': original_row['filename'],
                    'primary_label': original_row['primary_label'], 'original_start_time': original_row['start_time'],
                    'original_end_time': original_row['end_time'],
                    'audio_filepath_used': os.path.join(soundscape_audio_dir, original_row['filename'])
                })
            else: error_count +=1; error_messages.append(f"{det_id}: Worker returned None spec/no error (unexpected).")

    print(f"--- Processing Summary ---")
    print(f"Successfully processed {len(processed_spectrograms)} detections into spectrograms.")
    print(f"Encountered errors for {error_count} detections.")
    if error_messages: # Summarize errors
        print("\nTop 10 errors/issues:"); [print(f"  {i+1}. {msg}") for i, msg in enumerate(error_messages[:10])]
        if len(error_messages) > 10: print(f"  ... and {len(error_messages) - 10} more.")
    
    if processed_spectrograms:
        print(f"\nSaving {len(processed_spectrograms)} spectrograms to {OUTPUT_NPZ_PATH}...")
        try:
            os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
            np.savez_compressed(OUTPUT_NPZ_PATH, **processed_spectrograms)
            print("Spectrograms saved successfully.")
        except Exception as e: print(f"CRITICAL ERROR saving NPZ: {e}"); print(traceback.format_exc())

        df_meta_to_save = pd.DataFrame(processed_metadata)
        print(f"Saving metadata for {len(df_meta_to_save)} processed detections to {OUTPUT_METADATA_PATH}...")
        try:
            os.makedirs(os.path.dirname(OUTPUT_METADATA_PATH), exist_ok=True)
            df_meta_to_save.to_csv(OUTPUT_METADATA_PATH, index=False)
            print("Metadata CSV saved successfully.")
        except Exception as e: print(f"CRITICAL ERROR saving metadata CSV: {e}"); print(traceback.format_exc())
    else: print("No spectrograms were successfully processed to save.")

    overall_end_time = time.time()
    print(f"--- Soundscape Validation Set Preprocessing finished in {(overall_end_time - overall_start_time):.2f} seconds ---")

if __name__ == '__main__':
    # Pass the global_config instance to main
    if IS_EFFICIENT_AT_MODEL and efficient_at_spectrogram_generator is None:
        print("CRITICAL ERROR: EfficientAT model type detected, but spectrogram generator failed to initialize.")
    else:
        main(global_config)
