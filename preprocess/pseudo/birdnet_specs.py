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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import config 
import src.utils.utils as utils 

warnings.filterwarnings("ignore")

def _process_pseudo_label_row(args):
    """Worker function to process a single row from the pseudo_labels dataframe."""
    index, row, config = args
    
    filename = row['filename']
    start_time_orig = row['start_time'] # Original BirdNET detection start
    end_time_orig = row['end_time']   # Original BirdNET detection end
    # primary_label = row['primary_label'] # Not used in spec generation directly

    audio_path = os.path.join(config.unlabeled_audio_dir, filename)
    segment_key = f"{filename}_{int(start_time_orig)}_{int(end_time_orig)}"
    
    try:
        if not os.path.exists(audio_path):
            return (None, f"Audio file not found: {audio_path}")

        audio_data, _ = librosa.load(audio_path, sr=config.FS, mono=True)
        audio_len_samples = len(audio_data)
        expected_samples_5s = int(config.TARGET_DURATION * config.FS)

        # Desired window center is roughly the center of the 3s BirdNET detection
        # Desired window is 5 seconds: start_time_orig - 1 to end_time_orig + 1
        desired_start_sec = start_time_orig - 1.0
        desired_end_sec = end_time_orig + 1.0

        segment_audio = None

        if desired_start_sec < 0:
            # Case 1: Desired start is before audio begins, take first 5s
            # print(f"DEBUG {segment_key}: Desired start < 0, taking first 5s")
            end_sample_for_first_5s = min(audio_len_samples, expected_samples_5s)
            segment_audio = audio_data[0:end_sample_for_first_5s]
        elif desired_end_sec * config.FS > audio_len_samples:
            # Case 2: Desired end is after audio ends, take last 5s
            # print(f"DEBUG {segment_key}: Desired end > len, taking last 5s")
            start_sample_for_last_5s = max(0, audio_len_samples - expected_samples_5s)
            segment_audio = audio_data[start_sample_for_last_5s:audio_len_samples]
        else:
            # Case 3: Middle case - desired window is within audio (needs clamping)
            # print(f"DEBUG {segment_key}: Middle case processing")
            start_sample = max(0, int(desired_start_sec * config.FS))
            end_sample = min(audio_len_samples, int(desired_end_sec * config.FS))
            
            if start_sample >= end_sample:
                 return (None, f"Invalid calculated time range for {segment_key} after clamping middle case.")
            segment_audio = audio_data[start_sample:end_sample]

        # Pad or truncate the extracted segment_audio to be exactly expected_samples_5s
        current_len_samples = len(segment_audio)
        if current_len_samples == expected_samples_5s:
            pass # Already correct length
        elif current_len_samples < expected_samples_5s:
            # print(f"DEBUG {segment_key}: Padding from {current_len_samples} to {expected_samples_5s}")
            padding_needed = expected_samples_5s - current_len_samples
            # Simple zero padding at the end is common
            segment_audio = np.pad(segment_audio, (0, padding_needed), mode='constant', constant_values=0.0)
        else: # current_len_samples > expected_samples_5s
            # print(f"DEBUG {segment_key}: Truncating from {current_len_samples} to {expected_samples_5s}")
            # This case should ideally not happen often if logic above is correct for 5s target
            # but as a safeguard, truncate from the start (or center crop if preferred)
            segment_audio = segment_audio[:expected_samples_5s]
        
        # Final check on segment length after adjustments
        if len(segment_audio) != expected_samples_5s:
            return (None, f"Segment for {segment_key} has incorrect final length {len(segment_audio)} after padding/truncation. Expected {expected_samples_5s}.")

        mel_spec = utils.audio2melspec(segment_audio, config)
        if mel_spec is None:
            return (None, f"Spectrogram generation failed for {segment_key}")

        if mel_spec.shape != tuple(config.PREPROCESS_TARGET_SHAPE):
             final_spec_2d = cv2.resize(mel_spec, (config.PREPROCESS_TARGET_SHAPE[1], config.PREPROCESS_TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
        else:
             final_spec_2d = mel_spec
        
        # Add a new dimension at the beginning to make it (1, H, W)
        final_spec_3d = np.expand_dims(final_spec_2d, axis=0)

        return (segment_key, final_spec_3d.astype(np.float32))

    except Exception as e:
        return (None, f"Error processing {segment_key}: {e}")


def generate_pseudo_spectrograms(config):
    """Loads pseudo labels, processes segments in parallel, and saves spectrograms to NPZ."""
    print("--- Loading Pseudo Labels --- ")
    pseudo_df = pd.read_csv(config.soundscape_pseudo_csv_path)
    pseudo_df = pseudo_df[pseudo_df['confidence'] >= 0.90]
    print(f"Loaded {len(pseudo_df)} pseudo labels from {config.soundscape_pseudo_csv_path}")

    if pseudo_df.empty:
        print("Pseudo labels dataframe is empty. No spectrograms to generate.")
        return

    # --- Debug Limiting ---
    if config.debug and config.debug_limit_files > 0:
        print(f"\nDEBUG MODE: Limiting preprocessing to the first {config.debug_limit_files} pseudo labels.")
        pseudo_df = pseudo_df.head(config.debug_limit_files).copy()
        if pseudo_df.empty:
             print("DEBUG MODE: Pseudo labels dataframe is empty after limiting. Exiting.")
             return
    
    print("\n--- Generating Spectrograms for Pseudo Labels --- ")
    start_time = time.time()
    all_spectrograms = {}
    processed_count = 0
    error_count = 0
    errors = []

    # --- Setup Multiprocessing Tasks ---
    tasks = [(index, row, config) for index, row in pseudo_df.iterrows()]
    print(f"Created {len(tasks)} tasks for multiprocessing.")

    if not tasks:
        print("No tasks to process.")
        return

    num_workers = config.num_workers
    print(f"Using {num_workers} worker processes.")

    # --- Execute Tasks in Parallel ---
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(_process_pseudo_label_row, tasks)
            
            for i, result in enumerate(tqdm(results_iterator, total=len(tasks), desc="Generating Pseudo Specs")):
                key, data = result
                if key is not None:
                    all_spectrograms[key] = data
                    processed_count += 1
                else:
                    # data contains the error message in this case
                    errors.append(data) 
                    error_count += 1

    except Exception as e:
        print(f"\nCRITICAL ERROR during multiprocessing: {e}")
        print(traceback.format_exc())
        sys.exit(1) # Exit if the pool fails critically

    print() # Newline after progress indicator
    end_time = time.time()
    print(f"--- Spectrogram generation finished in {end_time - start_time:.2f} seconds ---")
    print(f"Successfully generated {processed_count} spectrograms.")
    print(f"Encountered {error_count} errors during processing.")
    if errors:
        print("\n--- Errors Encountered (sample) --- ")
        for err in errors[:20]: # Print first 20 errors
            print(err)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors. Check logs if needed.")
            
    # --- Save Results ---
    if all_spectrograms:
        # Define specific output path for pseudo spectrograms
        output_filename = "pseudo_spectrograms.npz"
        output_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(config._PREPROCESSED_OUTPUT_DIR, exist_ok=True)
        
        print(f"\nSaving {len(all_spectrograms)} pseudo label spectrograms to: {output_path}")
        start_save = time.time()
        try:
            np.savez_compressed(output_path, **all_spectrograms)
            end_save = time.time()
            print(f"NPZ saving took {end_save - start_save:.2f} seconds.")
        except Exception as e_save:
            print(f"CRITICAL ERROR saving NPZ file: {e_save}")
            print(traceback.format_exc())
            print("Spectrogram data is likely lost. Check disk space and permissions.")
            # Don't necessarily exit, but warn user
    else:
        print("No pseudo spectrograms were successfully generated to save.")

def main(config):
    """Main function to run the pseudo-label preprocessing steps."""
    overall_start = time.time()
    print("Starting BirdCLEF Pseudo-Label Preprocessing Pipeline...")
    print(f"Input CSV: {config.soundscape_pseudo_csv_path}")
    print(f"Input Audio Dir: {config.unlabeled_audio_dir}")
    print(f"Output NPZ Dir: {config._PREPROCESSED_OUTPUT_DIR}")

    generate_pseudo_spectrograms(config)

    overall_end = time.time()
    print(f"\nTotal pseudo-label preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set or could not be forced to 'spawn'.")
        
    main(config)
