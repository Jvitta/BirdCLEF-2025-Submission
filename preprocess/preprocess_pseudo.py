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
import birdclef_utils as utils 

warnings.filterwarnings("ignore")

def _process_pseudo_label_row(args):
    """Worker function to process a single row from the pseudo_labels dataframe."""
    index, row, config = args
    
    filename = row['filename']
    start_time = row['start_time']
    end_time = row['end_time']
    primary_label = row['primary_label']

    # Construct full audio path (assuming pseudo labels refer to files in unlabeled_audio_dir)
    audio_path = os.path.join(config.unlabeled_audio_dir, filename)

    # Unique key for this specific segment
    segment_key = f"{filename}_{int(start_time)}_{int(end_time)}"
    
    try:
        if not os.path.exists(audio_path):
            return (None, f"Audio file not found: {audio_path}")

        # Load the full audio file
        # Note: Loading the full file repeatedly can be slow. Caching or pre-loading could optimize.
        audio_data, _ = librosa.load(audio_path, sr=config.FS, mono=True)
        
        # Calculate start and end samples
        start_sample = int(start_time * config.FS)
        end_sample = int(end_time * config.FS)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
             return (None, f"Invalid time range [{start_time}-{end_time}] for {filename}")

        # Extract the segment
        segment_audio = audio_data[start_sample:end_sample]
        
        # Check segment length (should be exactly TARGET_DURATION, but allow small tolerance)
        expected_samples = int(config.TARGET_DURATION * config.FS)
        if len(segment_audio) < expected_samples * 0.9: # Check if significantly shorter
             return (None, f"Extracted segment too short ({len(segment_audio)} samples) for {segment_key}")
        elif len(segment_audio) < expected_samples:
            # Pad if slightly short
             segment_audio = np.pad(segment_audio, (0, expected_samples - len(segment_audio)), mode='constant')
        elif len(segment_audio) > expected_samples:
            # Truncate if slightly long
             segment_audio = segment_audio[:expected_samples]

        # Generate Mel spectrogram
        mel_spec = utils.audio2melspec(segment_audio, config)
        if mel_spec is None:
            return (None, f"Spectrogram generation failed for {segment_key}")

        # Resize to target shape
        if mel_spec.shape != tuple(config.TARGET_SHAPE):
             final_spec = cv2.resize(mel_spec, tuple(config.TARGET_SHAPE)[::-1], interpolation=cv2.INTER_LINEAR)
        else:
             final_spec = mel_spec
             
        return (segment_key, final_spec.astype(np.float32))

    except Exception as e:
        # tb_str = traceback.format_exc() # Uncomment for more detailed errors
        return (None, f"Error processing {segment_key}: {e}")


def generate_pseudo_spectrograms(config):
    """Loads pseudo labels, processes segments in parallel, and saves spectrograms to NPZ."""
    print("--- Loading Pseudo Labels --- ")
    try:
        pseudo_df = pd.read_csv(config.train_pseudo_csv_path)
        print(f"Loaded {len(pseudo_df)} pseudo labels from {config.train_pseudo_csv_path}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Pseudo labels file not found at {config.train_pseudo_csv_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR loading pseudo labels {config.train_pseudo_csv_path}: {e}. Exiting.")
        sys.exit(1)

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
    print(f"Input CSV: {config.train_pseudo_csv_path}")
    print(f"Input Audio Dir: {config.unlabeled_audio_dir}")
    print(f"Output NPZ Dir: {config._PREPROCESSED_OUTPUT_DIR}")

    generate_pseudo_spectrograms(config)

    overall_end = time.time()
    print(f"\nTotal pseudo-label preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    # Set start method for multiprocessing (important for CUDA/GPU if workers use it)
    # 'spawn' is often safer than 'fork'
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set or could not be forced to 'spawn'.")
        
    main(config)
