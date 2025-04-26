import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import glob         # For finding background chunks
import random       # For random selection
import math         # For padding calculations
import multiprocessing # For parallel processing
from functools import partial # Useful for multiprocessing args
import traceback    # For detailed error logging
import cv2          # For resizing spectrograms
import librosa      # For audio loading
from tqdm.auto import tqdm # Import tqdm for progress bar
import soundfile as sf # For saving audio examples
from config import config
import birdclef_utils as utils

warnings.filterwarnings("ignore")

def load_and_prepare_metadata(config):
    """Loads and prepares the metadata dataframe based on configuration."""
    print("--- 1. Loading and Preparing Metadata ---")
    start_time = time.time()

    # Load main training metadata and select relevant columns
    try:
        df_train_full = pd.read_csv(config.train_csv_path)
        if 'filename' not in df_train_full.columns or 'primary_label' not in df_train_full.columns:
             print(f"CRITICAL ERROR: Main metadata {config.train_csv_path} must contain 'filename' and 'primary_label' columns. Exiting.")
             sys.exit(1)
        df_train = df_train_full[['filename', 'primary_label']].copy()
        print(f"Loaded and filtered main metadata: {df_train.shape[0]} rows")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Main metadata file not found at {config.train_csv_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR loading main metadata {config.train_csv_path}: {e}. Exiting.")
        sys.exit(1)

    # Optionally load, filter, and merge rare species data
    if config.USE_RARE_DATA:
        print("USE_RARE_DATA is True. Loading rare species metadata.")
        try:
            df_rare_full = pd.read_csv(config.train_rare_csv_path) 
            if 'filename' not in df_rare_full.columns or 'primary_label' not in df_rare_full.columns:
                 print(f"Warning: Rare metadata {config.train_rare_csv_path} missing 'filename' or 'primary_label'. Skipping rare data.")
                 df_working = df_train 
                 print(f"Working metadata size (main only after rare data issue): {df_working.shape[0]} rows")
            else:
                df_rare = df_rare_full[['filename', 'primary_label']].copy()
                print(f"Loaded and filtered rare metadata: {df_rare.shape[0]} rows")

                # Combine FILTERED dataframes and remove duplicates
                df_combined = pd.concat([df_train, df_rare], ignore_index=True)
                initial_count = len(df_combined)
                df_combined.drop_duplicates(subset=['filename'], keep='first', inplace=True)
                duplicates_removed = initial_count - len(df_combined)
                print(f"Combined filtered metadata. Removed {duplicates_removed} duplicates based on filename.")
                df_working = df_combined
                print(f"Working metadata size (main + rare): {df_working.shape[0]} rows")

        except FileNotFoundError:
            print(f"Warning: Rare species metadata file not found at {config.train_rare_csv_path}. Proceeding with main data only.") # Updated message
            df_working = df_train # Use the already filtered df_train
            print(f"Working metadata size (main only): {df_working.shape[0]} rows")
        except Exception as e:
            print(f"Error loading or processing rare species metadata: {e}. Proceeding with main data only.")
            df_working = df_train # Use the already filtered df_train
            print(f"Working metadata size (main only): {df_working.shape[0]} rows")
    else:
        print("USE_RARE_DATA is False. Using main metadata only.")
        df_working = df_train # Use the already filtered df_train
        print(f"Working metadata size: {df_working.shape[0]} rows")

    # Redundant check as filtering happens earlier, but safe to keep
    if 'filename' not in df_working.columns or 'primary_label' not in df_working.columns:
        print("CRITICAL ERROR: Working dataframe missing 'filename' or 'primary_label' after processing. Exiting.")
        sys.exit(1)

    if df_working.empty:
        print("CRITICAL ERROR: Working metadata is empty after processing. Exiting.")
        sys.exit(1)

    df_working['samplename'] = df_working['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])

    end_time = time.time()
    print(f"Metadata preparation finished in {end_time - start_time:.2f} seconds.")
    return df_working


# --- New Worker Function for Preprocessing --- #
def _process_primary_for_mixing(args):
    """Worker to generate multiple noise-mixed spectrogram chunks for one primary audio file."""
    # Unpack args including the new example_samplenames set and example audio dirs
    primary_filepath, samplename, config, background_chunk_paths, fabio_intervals, vad_intervals, example_samplenames, example_audio_dirs, primary_filename = args

    results_dict = {}
    # Store tuple: (samplename, primary_spec, bg_spec, mixed_spec)
    example_specs_to_return = None 
    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples = int(0.5 * config.FS) # Minimum 0.5s duration for primary audio relevance

    try:
        # --- 1. Load Primary Audio --- #
        if not os.path.exists(primary_filepath):
            return samplename, {}, f"Primary file not found: {primary_filepath}", None # Added None for example_spec

        primary_audio, sr = librosa.load(primary_filepath, sr=config.FS, mono=True)
        if primary_audio is None or len(primary_audio) < min_samples:
            return samplename, {}, f"Primary audio too short/empty: {primary_filepath}", None

        # --- 2. Apply Voice Removal (Optional) --- #
        relevant_audio = primary_audio
        if config.REMOVE_SPEECH_INTERVALS:
            # filename_key = os.path.basename(primary_filepath) # Don't need basename for Fabio lookup
            is_example_file = samplename in example_samplenames # Check if current file is for detailed example output

            cleaned_audio = None
            # Use primary_filename for Fabio lookup
            if primary_filename in fabio_intervals: 
                start_time, stop_time = fabio_intervals[primary_filename]
                start_idx = max(0, int(start_time * config.FS))
                end_idx = min(len(primary_audio), int(stop_time * config.FS))
                if start_idx < end_idx:
                    cleaned_audio = primary_audio[start_idx:end_idx]
            
            elif primary_filepath in vad_intervals: 
                speech_timestamps = vad_intervals[primary_filepath] 
                if speech_timestamps:
                    non_speech_segments = []
                    current_pos_sec = 0.0
                    audio_duration_sec = len(primary_audio) / config.FS
                    try: speech_timestamps.sort(key=lambda x: x['start'])
                    except: speech_timestamps = []

                    for segment in speech_timestamps:
                        if not isinstance(segment, dict) or 'start' not in segment or 'end' not in segment: continue
                        start_speech_sec = segment['start']
                        end_speech_sec = segment['end']
                        if start_speech_sec > current_pos_sec:
                            start_idx = max(0, int(current_pos_sec * config.FS))
                            end_idx = min(len(primary_audio), int(start_speech_sec * config.FS))
                            if end_idx > start_idx: non_speech_segments.append(primary_audio[start_idx:end_idx])
                        current_pos_sec = max(current_pos_sec, end_speech_sec)

                    if current_pos_sec < audio_duration_sec:
                         start_idx = max(0, int(current_pos_sec * config.FS))
                         if start_idx < len(primary_audio): non_speech_segments.append(primary_audio[start_idx:])

                    if non_speech_segments:
                         non_speech_segments = [s for s in non_speech_segments if len(s) > 0]
                         if non_speech_segments: cleaned_audio = np.concatenate(non_speech_segments)
                         else: cleaned_audio = np.array([])
                    else: cleaned_audio = np.array([])
            
            # --- Decide whether to use cleaned_audio --- #
            if cleaned_audio is not None and len(cleaned_audio) >= min_samples:
                relevant_audio = cleaned_audio
            elif cleaned_audio is not None: # Cleaned but too short
                 pass
            else: # No intervals found or processing failed
                pass

        # Check length AFTER potentially using cleaned audio
        if len(relevant_audio) < min_samples:
             return samplename, {}, f"Relevant audio too short after processing: {primary_filepath}", None

        # --- 3. Generate Mixed Chunks --- #
        relevant_duration = len(relevant_audio)
        num_versions_to_generate = config.PRECOMPUTE_MIXED_VERSIONS
        # is_example = samplename in example_samplenames # Already defined above

        for i in range(num_versions_to_generate):
            # --- 3a. Select Primary Chunk --- #
            if relevant_duration < target_samples:
                n_copy = math.ceil(target_samples / relevant_duration)
                primary_chunk = np.tile(relevant_audio, n_copy)[:target_samples]
                if len(primary_chunk) < target_samples:
                    primary_chunk = np.pad(primary_chunk, (0, target_samples - len(primary_chunk)), mode='constant')
            else:
                max_start_idx = relevant_duration - target_samples
                start_idx = random.randint(0, max_start_idx)
                primary_chunk = relevant_audio[start_idx : start_idx + target_samples]

            # --- 3b. Select & Load Background Chunk --- #
            bg_audio = None
            attempt = 0
            max_attempts = 5
            while bg_audio is None and attempt < max_attempts:
                try:
                    if not background_chunk_paths:
                        raise ValueError("Background chunk path list is empty")
                    bg_path = random.choice(background_chunk_paths)
                    bg_audio, _ = librosa.load(bg_path, sr=config.FS, mono=True)
                    if len(bg_audio) < target_samples:
                        bg_audio = np.pad(bg_audio, (0, target_samples - len(bg_audio)), mode='constant')
                    elif len(bg_audio) > target_samples:
                        bg_audio = bg_audio[:target_samples]
                except Exception as e_bg:
                    attempt += 1
                    bg_audio = None

            if bg_audio is None:
                 continue # Skip this version if background failed

            # --- 3c. Mix Audio --- #
            # Use the configurable ratio
            primary_weight = config.MIXING_RATIO_PRIMARY
            background_weight = 1.0 - primary_weight
            mixed_audio = (primary_weight * primary_chunk.astype(np.float32) + 
                           background_weight * bg_audio.astype(np.float32))

            # --- 3d. Compute Spectrograms --- #
            primary_spec, bg_spec, mixed_spec = None, None, None
            try:
                mixed_spec = utils.audio2melspec(mixed_audio, config)
                if mixed_spec is None: continue # Skip if main spec fails

                # Resize only the final mixed spec for storage/return if needed
                if mixed_spec.shape != tuple(config.TARGET_SHAPE):
                     final_mixed_spec = cv2.resize(mixed_spec, tuple(config.TARGET_SHAPE)[::-1], interpolation=cv2.INTER_LINEAR)
                else:
                     final_mixed_spec = mixed_spec

                # Store final (potentially resized) spec for NPZ
                results_dict[f"{samplename}_mixedchunk{i}"] = final_mixed_spec.astype(np.float32)

                # --- If Example, Compute/Store Constituent Parts (Audio & Specs) --- #
                if is_example_file and i == 0:
                    # Use a safe filename base
                    safe_samplename = samplename.replace('/', '_').replace('\\', '_')

                    # Save Audio Components
                    try:
                        sf.write(os.path.join(example_audio_dirs['primary'], f"{safe_samplename}_primary_chunk0.wav"), primary_chunk, config.FS)
                        sf.write(os.path.join(example_audio_dirs['background'], f"{safe_samplename}_background_chunk0.wav"), bg_audio, config.FS)
                        sf.write(os.path.join(example_audio_dirs['mixed'], f"{safe_samplename}_mixed_chunk0.wav"), mixed_audio, config.FS)
                    except Exception as e_audio_save:
                        # print(f"Worker Warning: Failed to save audio example components for {samplename}: {e_audio_save}")
                        pass # Don't stop processing for this

                    # Compute Spectrograms for constituent parts (use original shapes for plotting)
                    # We calculate them here *only if* it's an example to avoid unnecessary computation
                    primary_spec = utils.audio2melspec(primary_chunk, config)
                    bg_spec = utils.audio2melspec(bg_audio, config)
                    # mixed_spec is already computed

                    # Store the tuple of specs (original shapes) for return
                    if primary_spec is not None and bg_spec is not None and mixed_spec is not None:
                        example_specs_to_return = (samplename, 
                                                 primary_spec.astype(np.float32), 
                                                 bg_spec.astype(np.float32), 
                                                 mixed_spec.astype(np.float32))
                    # else: # Handle case where component spec fails (rare) 
                    #     print(f"Worker Warning: Failed to generate one or more component spectrograms for example {samplename}")

            except Exception as e_spec:
                 # print(f"Worker Warning: Error during spectrogram processing for {samplename} chunk {i}: {e_spec}")
                 pass # Skip storing this version if any spec step fails

    except Exception as e_main:
        tb_str = traceback.format_exc()
        # Ensure the return signature matches the success case, adding None for example_spec
        # Revert: We only need samplename, result_dict, error (example handling is done in main loop now)
        return samplename, {}, f"Error processing {primary_filepath}: {e_main}\n{tb_str}" 

    # Return results 
    # Revert: Return samplename, result_dict, None (no example tuple needed from worker)
    return samplename, results_dict, None
# --- End Worker Function --- #


def generate_and_save_spectrograms(df, config):
    """Generates multiple noise-mixed spectrogram chunks using multiprocessing and saves to NPZ."""
    if df is None or df.empty:
        print("Working dataframe is empty, skipping spectrogram generation.")
        return

    # --- Debug Limiting --- #
    if config.debug and config.debug_limit_files > 0:
        print(f"\nDEBUG MODE: Limiting preprocessing to the first {config.debug_limit_files} files.")
        df = df.head(config.debug_limit_files).copy()
        if df.empty:
             print("DEBUG MODE: Dataframe is empty after limiting. Exiting.")
             return {}
    # --- End Debug Limiting ---

    # --- Load VAD/Fabio Intervals (only if needed) --- #
    fabio_intervals = {}
    vad_intervals = {}
    if config.REMOVE_SPEECH_INTERVALS:
        print("REMOVE_SPEECH_INTERVALS is True. Loading VAD/Fabio intervals.")
        try:
            fabio_df = pd.read_csv(config.FABIO_CSV_PATH)
            # Use filename as key, assuming it's unique and matches df['filename']
            fabio_intervals = {row['filename']: (row['start'], row['stop'])
                               for _, row in fabio_df.iterrows() if 'filename' in row and 'start' in row and 'stop' in row}
            print(f"Loaded Fabio intervals for {len(fabio_intervals)} files.")
        except FileNotFoundError:
             print(f"Info: Fabio intervals file not found at {config.FABIO_CSV_PATH}. Skipping.")
        except Exception as e: print(f"Warning: Could not load Fabio intervals: {e}")

        try:
            with open(config.TRANSFORMED_VOICE_DATA_PKL_PATH, 'rb') as f:
                vad_data = pickle.load(f) 
                if isinstance(vad_data, dict):
                     vad_intervals = {k: v for k, v in vad_data.items() if isinstance(v, list)}
                     print(f"Loaded VAD intervals for {len(vad_intervals)} files.")
                else:
                     print(f"Warning: VAD pickle file at {config.TRANSFORMED_VOICE_DATA_PKL_PATH} does not contain a dictionary.")

        except FileNotFoundError:
             print(f"Info: VAD intervals pickle file not found at {config.TRANSFORMED_VOICE_DATA_PKL_PATH}. Skipping.")
        except Exception as e: print(f"Warning: Could not load VAD intervals: {e}")
    else:
         print("REMOVE_SPEECH_INTERVALS is False, skipping loading VAD/Fabio intervals.")
    # --- End Interval Loading --- #

    # --- Load Background Chunk Paths --- #
    print(f"Looking for background audio chunks in: {config.unlabeled_audio_dir_chunked}")
    background_chunk_paths = glob.glob(os.path.join(config.unlabeled_audio_dir_chunked, '*.wav'))
    if not background_chunk_paths:
        print(f"CRITICAL ERROR: No background chunk .wav files found in {config.unlabeled_audio_dir_chunked}. Please run create_background_chunks.py first. Exiting.")
        sys.exit(1)
    else:
        print(f"Found {len(background_chunk_paths)} background audio chunk files.")
    # --- End Background Path Loading --- #

    print("\n--- 2. Generating Noise-Mixed Spectrograms (Multi-Chunk, NPZ) ---")
    start_time = time.time()
    all_spectrograms = {}
    example_spectrograms = {} # Dictionary to store example spectrograms {samplename: spec}
    processed_count = 0
    error_count = 0
    skipped_files = 0 # Files processed by worker but yielded no valid spectrograms
    errors = []

    # --- Select First N Examples --- #
    num_examples = 5
    # potential_examples = df['samplename'].unique().tolist() # Old random selection
    potential_examples = df['samplename'].tolist() # Get samplenames in order
    example_samplenames = set(potential_examples[:num_examples]) # Take the first N unique ones
    print(f"Selected first {len(example_samplenames)} samplenames for detailed examples: {list(example_samplenames)}")
    # Create specific output directories for example audio components
    example_audio_base_dir = os.path.join(config.OUTPUT_DIR, "example_audio")
    example_audio_dirs = {
        'primary': os.path.join(example_audio_base_dir, 'primary'),
        'background': os.path.join(example_audio_base_dir, 'background'),
        'mixed': os.path.join(example_audio_base_dir, 'mixed')
        # 'relevant_source': os.path.join(example_audio_base_dir, 'relevant_source') # Removed debug dir
    }
    for dir_path in example_audio_dirs.values():
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create example directory {dir_path}: {e}")

    # --- Setup Multiprocessing Tasks --- #
    tasks = []
    skipped_path_count = 0
    required_cols = {'filename', 'samplename', 'primary_label'}
    if not required_cols.issubset(df.columns):
        print(f"CRITICAL ERROR: Dataframe missing required columns for processing: {required_cols - set(df.columns)}")
        sys.exit(1)

    for index, row in df.iterrows():
        primary_filename = row['filename']
        samplename = row['samplename']

        potential_main_path = os.path.join(config.train_audio_dir, primary_filename)
        potential_rare_path = os.path.join(config.train_audio_rare_dir, primary_filename) if config.USE_RARE_DATA else None

        primary_filepath = None
        if os.path.exists(potential_main_path):
            primary_filepath = potential_main_path
        elif potential_rare_path and os.path.exists(potential_rare_path):
            primary_filepath = potential_rare_path

        if primary_filepath:
            # Pass the set of example samplenames AND the original filename to the worker
            tasks.append((primary_filepath, samplename, config, background_chunk_paths, 
                          fabio_intervals, vad_intervals, example_samplenames, example_audio_dirs,
                          primary_filename)) # Added primary_filename
        else:
            skipped_path_count += 1

    if skipped_path_count > 0:
        print(f"Warning: Skipped {skipped_path_count} files because the audio path could not be found in specified directories ({config.train_audio_dir} or {config.train_audio_rare_dir if config.USE_RARE_DATA else 'N/A'}).")

    print(f"Created {len(tasks)} tasks for multiprocessing.")
    if not tasks:
        print("No tasks to process. Exiting spectrogram generation.")
        return # Return None as no examples will be generated

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} worker processes.")

    # Intermediate storage: {samplename: [list of specs]} 
    grouped_results = {} 
    processed_count = 0 # Reset counter for primary files with >=1 successful chunk
    error_count = 0
    skipped_files = 0
    errors = []

    # --- Execute Tasks in Parallel --- #
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(_process_primary_for_mixing, tasks)

            # Process results as they complete, wrapped with tqdm for progress bar
            for i, result in enumerate(tqdm(results_iterator, total=len(tasks), desc="Generating Specs")):
                # Unpack result: samplename, spec_dict (chunks for this file), error_msg
                samplename, spec_dict, error_msg = result 
                if error_msg:
                    errors.append(f"{samplename}: {error_msg}")
                    error_count += 1
                elif spec_dict: # Check if the dictionary is not empty
                    # Initialize list for this samplename if it's the first time seeing it
                    if samplename not in grouped_results:
                        grouped_results[samplename] = []
                        processed_count += 1 # Increment count only when we first add a samplename
                    # Append all generated specs (values from spec_dict) to the list
                    # Convert values to list before extending
                    grouped_results[samplename].extend(list(spec_dict.values())) 
                else:
                    # Worker finished but produced no specs
                    skipped_files += 1
                 
    except Exception as e:
        print(f"\nCRITICAL ERROR during multiprocessing: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    print() # Newline after progress indicator
    end_time = time.time()
    print(f"--- Spectrogram generation finished in {end_time - start_time:.2f} seconds ---")
    # Adjust success message slightly
    print(f"Successfully generated spectrograms for {processed_count} primary files.") 
    total_chunks = sum(len(v) for v in grouped_results.values())
    print(f"Total spectrogram chunks generated: {total_chunks}")
    print(f"Encountered errors for {error_count} primary files.")
    if skipped_files > 0:
         print(f"{skipped_files} primary files yielded no valid spectrograms (skipped).")
    if errors:
        print("\n--- Errors Encountered (sample) ---")
        for err in errors[:20]:
            print(err)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors. Check logs if needed.")

    # --- Save Results --- #
    # grouped_results now holds {samplename: [spec1, spec2, ...]}
    if grouped_results:
        print(f"Saving {len(grouped_results)} primary file entries (total {total_chunks} chunks) to: {config.PREPROCESSED_NPZ_PATH}") 
        start_save = time.time()
        try:
            # Convert lists of arrays to numpy arrays before saving for efficiency
            save_dict = {key: np.array(value) for key, value in grouped_results.items()}
            np.savez_compressed(config.PREPROCESSED_NPZ_PATH, **save_dict)
            end_save = time.time()
            print(f"NPZ saving took {end_save - start_save:.2f} seconds.")
        except Exception as e_save:
            print(f"CRITICAL ERROR saving NPZ file: {e_save}")
            print(traceback.format_exc())
            print("Spectrogram data is likely lost. Check disk space and permissions.")
            sys.exit(1)
    else:
        print("No spectrograms were successfully generated to save.")
    
    return grouped_results 

def main(config):
    """Main function to run the preprocessing steps."""
    overall_start = time.time()
    print("Starting BirdCLEF Preprocessing Pipeline...")
    print(f"Configuration: Using {'Rare Data' if config.USE_RARE_DATA else 'Main Data Only'}, {'Removing Speech Intervals' if config.REMOVE_SPEECH_INTERVALS else 'Not Removing Speech'}, Target Duration: {config.TARGET_DURATION}s, Mixed Versions: {config.PRECOMPUTE_MIXED_VERSIONS}")
    print(f"Output NPZ: {config.PREPROCESSED_NPZ_PATH}")
    print(f"Background Chunks Dir: {config.unlabeled_audio_dir_chunked}")

    # --- 1. Load Metadata --- #
    df_working = load_and_prepare_metadata(config)

    # --- 2. Generate and Save Spectrograms --- #
    if df_working is not None and not df_working.empty:
        # generate_and_save_spectrograms now returns the grouped results, but we don't use it here
        # Example plotting is removed / needs rework for the new format
        _ = generate_and_save_spectrograms(df_working, config) 
        # if example_spectrograms: # Old plotting call
        #     plot_example_spectrograms(example_spectrograms, config)
    else:
        print("Metadata loading failed or resulted in empty dataframe. Cannot proceed.")

    overall_end = time.time()
    print(f"\nTotal preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    main(config) 