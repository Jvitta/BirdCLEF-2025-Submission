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
from functools import partial 
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

def load_and_prepare_metadata(config):
    """Loads and prepares the metadata dataframe based on configuration."""
    print("--- 1. Loading and Preparing Metadata ---")
    start_time = time.time()

    try:
        df_train_full = pd.read_csv(config.train_csv_path)
        df_train = df_train_full[['filename', 'primary_label']].copy()
        print(f"Loaded and filtered main metadata: {df_train.shape[0]} rows")
    except Exception as e:
        print(f"CRITICAL ERROR loading main metadata {config.train_csv_path}: {e}. Exiting.")
        sys.exit(1)

    if config.USE_RARE_DATA:
        print("USE_RARE_DATA is True. Loading rare species metadata.")
        try:
            df_rare_full = pd.read_csv(config.train_rare_csv_path) 
            df_rare = df_rare_full[['filename', 'primary_label']].copy()
            print(f"Loaded and filtered rare metadata: {df_rare.shape[0]} rows")

            df_combined = pd.concat([df_train, df_rare], ignore_index=True)
            df_working = df_combined
            print(f"Working metadata size (main + rare): {df_working.shape[0]} rows")
        except Exception as e:
            print(f"Error loading or processing rare species metadata: {e}. Proceeding with main data only.")
            df_working = df_train 
            print(f"Working metadata size (main only): {df_working.shape[0]} rows")
    else:
        print("USE_RARE_DATA is False. Using main metadata only.")
        df_working = df_train 
        print(f"Working metadata size: {df_working.shape[0]} rows")

    df_working['samplename'] = df_working['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])

    end_time = time.time()
    print(f"Metadata preparation finished in {end_time - start_time:.2f} seconds.")
    return df_working

def _process_primary_for_chunking(args):
    """Worker to generate multiple 5-second spectrogram chunks for one primary audio file."""
    primary_filepath, samplename, config, fabio_intervals, vad_intervals, primary_filename = args

    results_dict = {}
    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples = int(0.5 * config.FS)

    try:
        primary_audio, sr = librosa.load(primary_filepath, sr=config.FS, mono=True)
        if primary_audio is None or len(primary_audio) < min_samples:
            return samplename, {}, f"Primary audio too short/empty: {primary_filepath}"

        relevant_audio = primary_audio
        if config.REMOVE_SPEECH_INTERVALS:
            cleaned_audio = None

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
            
            if cleaned_audio is not None and len(cleaned_audio) >= min_samples:
                relevant_audio = cleaned_audio
            else:
                pass

        if len(relevant_audio) < min_samples:
             return samplename, {}, f"Relevant audio too short after processing: {primary_filepath}"

        relevant_duration = len(relevant_audio)
        num_versions_to_generate = 1 if relevant_duration < target_samples else config.PRECOMPUTE_VERSIONS

        for i in range(num_versions_to_generate):
            if relevant_duration < target_samples:
                n_copy = math.ceil(target_samples / relevant_duration)
                primary_chunk = np.tile(relevant_audio, n_copy)[:target_samples]
                if len(primary_chunk) < target_samples:
                    primary_chunk = np.pad(primary_chunk, (0, target_samples - len(primary_chunk)), mode='constant')
            else:
                max_start_idx = relevant_duration - target_samples
                start_idx = random.randint(0, max_start_idx)
                primary_chunk = relevant_audio[start_idx : start_idx + target_samples]

            primary_spec_chunk = None
            try:
                primary_spec_chunk = utils.audio2melspec(primary_chunk, config)
                if primary_spec_chunk is None: continue

                if primary_spec_chunk.shape != tuple(config.TARGET_SHAPE):
                     final_spec = cv2.resize(primary_spec_chunk, tuple(config.TARGET_SHAPE)[::-1], interpolation=cv2.INTER_LINEAR)
                else:
                     final_spec = primary_spec_chunk

                results_dict[f"{samplename}_chunk{i}"] = final_spec.astype(np.float32)

            except Exception as e_spec:
                 pass

    except Exception as e_main:
        tb_str = traceback.format_exc()
        return samplename, results_dict, f"Error processing {primary_filepath}: {e_main}\n{tb_str}"

    return samplename, results_dict, None


def generate_and_save_spectrograms(df, config):
    """Generates multiple 5-second spectrogram chunks using multiprocessing and saves to NPZ."""
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
    except Exception as e:
        print(f"Warning: Could not load VAD intervals: {e}")
    else:
        print("REMOVE_SPEECH_INTERVALS is False, skipping loading VAD/Fabio intervals.")
    # --- End Interval Loading --- #

    print("\n--- 2. Generating Spectrogram Chunks (Multi-Chunk, NPZ) ---")
    start_time = time.time()
    all_spectrograms = {}
    processed_count = 0
    error_count = 0
    skipped_files = 0 # Files processed by worker but yielded no valid spectrograms
    errors = []

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
            tasks.append((primary_filepath, samplename, config, fabio_intervals, vad_intervals, primary_filename))
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
            results_iterator = pool.imap_unordered(_process_primary_for_chunking, tasks)

            # Process results as they complete, wrapped with tqdm for progress bar
            for i, result in enumerate(tqdm(results_iterator, total=len(tasks), desc="Generating Specs")):
                # Unpack result: samplename, spec_dict (chunks for this file), error_msg
                samplename, spec_dict, error_msg = result # Adjusted unpacking
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
    print(f"Configuration: Using {'Rare Data' if config.USE_RARE_DATA else 'Main Data Only'}, {'Removing Speech Intervals' if config.REMOVE_SPEECH_INTERVALS else 'Not Removing Speech'}, Target Duration: {config.TARGET_DURATION}s, Versions per File: {config.PRECOMPUTE_VERSIONS}")
    print(f"Output NPZ: {config.PREPROCESSED_NPZ_PATH}")

    df_working = load_and_prepare_metadata(config)

    if df_working is not None and not df_working.empty:
        _ = generate_and_save_spectrograms(df_working, config)
    else:
        print("Metadata loading failed or resulted in empty dataframe. Cannot proceed.")

    overall_end = time.time()
    print(f"\nTotal preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    main(config)