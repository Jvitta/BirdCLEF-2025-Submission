import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
import multiprocessing
import traceback
import cv2
import librosa
from tqdm.auto import tqdm
import torch 
from src.models.efficient_at.preprocess import AugmentMelSTFT
import argparse 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
project_root = os.path.dirname(project_root) 
sys.path.append(project_root)

from config import config 
import src.utils.utils as utils 

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Preprocess pseudo-labeled audio data.")
parser.add_argument("--mode", type=str, choices=["train", "val"], default="train",
                    help="Preprocessing mode: 'train' for augmented spectrograms, 'val' for fixed-setting spectrograms.")
cmd_args = parser.parse_args()

# --- Global Spectrogram Generator (for EfficientAT models) --- #
efficient_at_spectrogram_generator_global = None
IS_EFFICIENT_AT_MODEL_GLOBAL = False

def _process_pseudo_label_row(args):
    """Worker function to process a single row from the pseudo_labels dataframe."""
    index, row, worker_config_obj, efficient_at_generator_ref, is_efficient_at_model_ref = args
    
    filename = row['filename']
    start_time_orig = row['start_time'] 
    end_time_orig = row['end_time'] 

    audio_path = os.path.join(worker_config_obj.unlabeled_audio_dir, filename)
    segment_key = f"{filename}_{int(start_time_orig)}_{int(end_time_orig)}"
    
    try:
        if not os.path.exists(audio_path):
            return (segment_key, None, f"Audio file not found: {audio_path}")

        audio_data, _ = librosa.load(audio_path, sr=worker_config_obj.FS, mono=True)
        audio_len_samples = len(audio_data)
        expected_samples_5s = int(worker_config_obj.TARGET_DURATION * worker_config_obj.FS)

        desired_start_sec = start_time_orig - 1.0
        desired_end_sec = end_time_orig + 1.0

        segment_audio = None

        if desired_start_sec < 0:
            end_sample_for_first_5s = min(audio_len_samples, expected_samples_5s)
            segment_audio = audio_data[0:end_sample_for_first_5s]

        elif desired_end_sec * worker_config_obj.FS > audio_len_samples:
            start_sample_for_last_5s = max(0, audio_len_samples - expected_samples_5s)
            segment_audio = audio_data[start_sample_for_last_5s:audio_len_samples]

        else:
            start_sample = max(0, int(desired_start_sec * worker_config_obj.FS))
            end_sample = min(audio_len_samples, int(desired_end_sec * worker_config_obj.FS))
            
            if start_sample >= end_sample:
                 return (segment_key, None, f"Invalid calculated time range for {segment_key} after clamping middle case.")
            segment_audio = audio_data[start_sample:end_sample]

        # Pad or truncate the extracted segment_audio to be exactly expected_samples_5s
        current_len_samples = len(segment_audio)
        if current_len_samples == expected_samples_5s:
            pass 
        elif current_len_samples < expected_samples_5s:
            padding_needed = expected_samples_5s - current_len_samples
            segment_audio = np.pad(segment_audio, (0, padding_needed), mode='constant', constant_values=0.0)
        else: 
            segment_audio = segment_audio[:expected_samples_5s]
        
        # Final check on segment length after adjustments
        if len(segment_audio) != expected_samples_5s:
            return (segment_key, None, f"Segment for {segment_key} has incorrect final length {len(segment_audio)} after padding/truncation. Expected {expected_samples_5s}.")

        # --- Spectrogram Generation & Resizing to PREPROCESS_TARGET_SHAPE ---
        final_spec_at_preprocess_shape = None
        if is_efficient_at_model_ref:
            audio_tensor = torch.from_numpy(segment_audio.astype(np.float32))
            with torch.no_grad():
                raw_spec_tensor = efficient_at_generator_ref(audio_tensor.unsqueeze(0))

            final_spec_at_preprocess_shape = raw_spec_tensor.squeeze(0).cpu().numpy()
            # Verify shape as a safeguard
            if final_spec_at_preprocess_shape.shape[0] != worker_config_obj.PREPROCESS_TARGET_SHAPE[0] or \
               final_spec_at_preprocess_shape.shape[1] != worker_config_obj.PREPROCESS_TARGET_SHAPE[1]:
                print(f"Warning for {segment_key} (MN model): AugmentMelSTFT output shape {final_spec_at_preprocess_shape.shape} does not match PREPROCESS_TARGET_SHAPE {worker_config_obj.PREPROCESS_TARGET_SHAPE}. Resizing.")
                final_spec_at_preprocess_shape = cv2.resize(
                    final_spec_at_preprocess_shape, 
                    (worker_config_obj.PREPROCESS_TARGET_SHAPE[1], worker_config_obj.PREPROCESS_TARGET_SHAPE[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
        else: # EfficientNet model
            raw_spec_en = utils.audio2melspec(segment_audio, worker_config_obj)
            if raw_spec_en is None:
                 return (segment_key, None, f"Spectrogram generation failed (audio2melspec) for {segment_key}")
            # Resize raw_spec_en (e.g. 136x1250 for EN) to PREPROCESS_TARGET_SHAPE (e.g. 256x256 for EN)
            if raw_spec_en.shape[0] != worker_config_obj.PREPROCESS_TARGET_SHAPE[0] or \
               raw_spec_en.shape[1] != worker_config_obj.PREPROCESS_TARGET_SHAPE[1]:
                final_spec_at_preprocess_shape = cv2.resize(
                    raw_spec_en, 
                    (worker_config_obj.PREPROCESS_TARGET_SHAPE[1], worker_config_obj.PREPROCESS_TARGET_SHAPE[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                final_spec_at_preprocess_shape = raw_spec_en

        if final_spec_at_preprocess_shape is None:
            return (segment_key, None, f"Final spectrogram (at PREPROCESS_TARGET_SHAPE) is None for {segment_key}")

        final_spec_3d = np.expand_dims(final_spec_at_preprocess_shape, axis=0)
        return (segment_key, final_spec_3d.astype(np.float32), None)

    except Exception as e:
        tb_str = traceback.format_exc()
        return (segment_key, None, f"Error processing {segment_key}: {e}\\n{tb_str}")

def generate_pseudo_spectrograms(mode):
    global efficient_at_spectrogram_generator_global, IS_EFFICIENT_AT_MODEL_GLOBAL

    IS_EFFICIENT_AT_MODEL_GLOBAL = 'mn' in config.model_name.lower()
    print(f"--- Running Pseudo Spectrogram Generation in {mode.upper()} mode ---")

    if IS_EFFICIENT_AT_MODEL_GLOBAL:
        print(f"Initializing EfficientAT Spectrogram Generator ({mode} mode): Config FminAugRange={config.FMIN_AUG_RANGE}, Config FmaxAugRange={config.FMAX_AUG_RANGE}")
        
        efficient_at_spectrogram_generator_global = AugmentMelSTFT(
            n_mels=config.N_MELS, 
            sr=config.FS,
            win_length=config.WIN_LENGTH,
            hopsize=config.HOP_LENGTH, 
            n_fft=config.N_FFT,
            fmin=config.FMIN,
            fmax=config.FMAX,
            freqm=0, 
            timem=0,
            fmin_aug_range=config.FMIN_AUG_RANGE, 
            fmax_aug_range=config.FMAX_AUG_RANGE  
        )
        
        if mode == "train":
            efficient_at_spectrogram_generator_global.train()
            print("EfficientAT spectrogram generator initialized in TRAIN mode.")
        else: 
            efficient_at_spectrogram_generator_global.eval()
            print("EfficientAT spectrogram generator initialized in EVAL mode.")
    else:
        print(f"--- Using EfficientNet-style spectrogram generation via utils.audio2melspec for model: {config.model_name} ({mode} mode) ---")

    print("--- Loading Pseudo Labels --- ")
    pseudo_df = pd.read_csv(config.soundscape_pseudo_calibrated_csv_path)
    print(f"Loaded {len(pseudo_df)} pseudo labels from {config.soundscape_pseudo_calibrated_csv_path}.")

    print("\\n--- Generating Spectrograms for Pseudo Labels --- ")
    start_time = time.time()
    all_spectrograms = {}
    processed_count = 0
    error_count = 0
    errors = []

    tasks = [(index, row, config, efficient_at_spectrogram_generator_global, IS_EFFICIENT_AT_MODEL_GLOBAL) 
             for index, row in pseudo_df.iterrows()]
    print(f"Created {len(tasks)} tasks for multiprocessing.")

    num_workers = config.num_workers
    print(f"Using {num_workers} worker processes.")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(_process_pseudo_label_row, tasks)
            
            for i, result_tuple in enumerate(tqdm(results_iterator, total=len(tasks), desc=f"Generating Pseudo Specs ({mode} mode)")):
                key, data, error_msg = result_tuple 
                if data is not None and error_msg is None : 
                    all_spectrograms[key] = data
                    processed_count += 1
                else:
                    errors.append(f"{key if key else 'UnknownKey'}: {error_msg if error_msg else 'No data and no error message.'}") 
                    error_count += 1

    except Exception as e:
        print(f"\\nCRITICAL ERROR during multiprocessing: {e}")
        print(traceback.format_exc())
        sys.exit(1) 

    print() 
    end_time = time.time()
    print(f"--- Spectrogram generation finished in {end_time - start_time:.2f} seconds ---")
    print(f"Successfully generated {processed_count} spectrograms.")
    print(f"Encountered {error_count} errors during processing.")
    if errors:
        print("\\n--- Errors Encountered (sample) --- ")
        for err in errors[:20]: 
            print(err)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors. Check logs if needed.")
            
    if all_spectrograms:
        output_filename = f"pseudo_spectrograms_{config.model_name}_{mode}.npz" 
        output_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, output_filename)
        
        os.makedirs(config._PREPROCESSED_OUTPUT_DIR, exist_ok=True)
        
        print(f"\\nSaving {len(all_spectrograms)} pseudo label spectrograms (shape: {config.PREPROCESS_TARGET_SHAPE}, mode: {mode}) to: {output_path}")
        start_save = time.time()
        try:
            np.savez_compressed(output_path, **all_spectrograms)
            end_save = time.time()
            print(f"NPZ saving took {end_save - start_save:.2f} seconds.")
        except Exception as e_save:
            print(f"CRITICAL ERROR saving NPZ file: {e_save}")
            print(traceback.format_exc())
            print("Spectrogram data is likely lost. Check disk space and permissions.")
    else:
        print("No pseudo spectrograms were successfully generated to save.")

def main(mode_to_use):
    overall_start = time.time()
    print("Starting BirdCLEF Pseudo-Label Preprocessing Pipeline...")
    print(f"Mode: {mode_to_use.upper()}")
    print(f"Input CSV: {config.soundscape_pseudo_calibrated_csv_path}")
    print(f"Input Audio Dir: {config.unlabeled_audio_dir}")
    print(f"Output NPZ Dir: {config._PREPROCESSED_OUTPUT_DIR}")
    print(f"Model name from config: {config.model_name}")
    print(f"PREPROCESS_TARGET_SHAPE: {config.PREPROCESS_TARGET_SHAPE}")
    print(f"TARGET_SHAPE (model input, after dataset ops): {config.TARGET_SHAPE}")

    generate_pseudo_spectrograms(mode_to_use)

    overall_end = time.time()
    print(f"\\nTotal pseudo-label preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set or could not be forced to 'spawn'.")
        
    main(cmd_args.mode) # Pass only the mode
