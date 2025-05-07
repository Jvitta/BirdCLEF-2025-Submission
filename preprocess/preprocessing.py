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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import config 
import birdclef_utils as utils 

warnings.filterwarnings("ignore")


random.seed(config.seed)
np.random.seed(config.seed)

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

def _load_and_clean_audio(filepath, filename, config, fabio_intervals, vad_intervals, 
                          class_name, scientific_name, UNCOVERED_AVES_SCIENTIFIC_NAME, min_samples):
    """Loads audio, applies VAD/Fabio if needed, checks min length."""
    try:
        primary_audio, _ = librosa.load(filepath, sr=config.FS, mono=True)
        if primary_audio is None or len(primary_audio) < min_samples:
            return None, f"Primary audio too short/empty: {filepath}"
    except Exception as e_load:
        return None, f"Error loading audio {filepath}: {e_load}"

    relevant_audio = primary_audio
    should_apply_vad_fabio = False
    if config.REMOVE_SPEECH_INTERVALS:
        if not config.REMOVE_SPEECH_ONLY_NON_AVES:
            should_apply_vad_fabio = True
        elif class_name != 'Aves' or scientific_name == UNCOVERED_AVES_SCIENTIFIC_NAME:
            should_apply_vad_fabio = True

    if should_apply_vad_fabio:
        cleaned_audio_vad_fabio = None
        if filename in fabio_intervals:
            start_time_fabio, stop_time_fabio = fabio_intervals[filename]
            start_idx_fabio = max(0, int(start_time_fabio * config.FS))
            end_idx_fabio = min(len(primary_audio), int(stop_time_fabio * config.FS))
            if start_idx_fabio < end_idx_fabio:
                cleaned_audio_vad_fabio = primary_audio[start_idx_fabio:end_idx_fabio]
        elif filepath in vad_intervals:
            speech_timestamps = vad_intervals[filepath]
            if speech_timestamps:
                non_speech_segments = []
                current_pos_sec = 0.0
                audio_duration_sec = len(primary_audio) / config.FS
                try: speech_timestamps.sort(key=lambda x: x.get('start', 0))
                except: speech_timestamps = []
                for segment in speech_timestamps:
                    if not isinstance(segment, dict) or 'start' not in segment or 'end' not in segment: continue
                    start_speech_sec = segment['start']
                    end_speech_sec = segment['end']
                    if start_speech_sec > current_pos_sec:
                        start_idx_segment = max(0, int(current_pos_sec * config.FS))
                        end_idx_segment = min(len(primary_audio), int(start_speech_sec * config.FS))
                        if end_idx_segment > start_idx_segment: non_speech_segments.append(primary_audio[start_idx_segment:end_idx_segment])
                    current_pos_sec = max(current_pos_sec, end_speech_sec)
                if current_pos_sec < audio_duration_sec:
                    start_idx_segment = max(0, int(current_pos_sec * config.FS))
                    if start_idx_segment < len(primary_audio): non_speech_segments.append(primary_audio[start_idx_segment:])
                if non_speech_segments:
                    non_speech_segments = [s for s in non_speech_segments if len(s) > 0]
                    if non_speech_segments: cleaned_audio_vad_fabio = np.concatenate(non_speech_segments)
                    else: cleaned_audio_vad_fabio = np.array([])
                else: cleaned_audio_vad_fabio = np.array([])
        
        if cleaned_audio_vad_fabio is not None and len(cleaned_audio_vad_fabio) >= min_samples:
            relevant_audio = cleaned_audio_vad_fabio

    if len(relevant_audio) < min_samples: # Check again after VAD
         return None, f"Primary audio too short after VAD/Fabio: {filepath}"

    return relevant_audio, None

def _pad_or_tile_audio(audio_data, target_samples):
    """Pads or tiles audio to target_samples length."""
    current_len = len(audio_data)
    if current_len == target_samples:
        return audio_data
    elif current_len < target_samples:
        n_copy = math.ceil(target_samples / current_len) 
        padded_audio = np.tile(audio_data, n_copy)[:target_samples]
        if len(padded_audio) < target_samples: # Final padding check
            padded_audio = np.pad(padded_audio, (0, target_samples - len(padded_audio)), mode='constant')
        return padded_audio
    else: # current_len > target_samples (shouldn't happen if called correctly, but safe)
        return audio_data[:target_samples]

def _extract_birdnet_chunk(audio_long, detection, config, min_samples, target_samples):
    """Extracts a 5s chunk centered around a BirdNET detection."""
    try:
        birdnet_start_sec = detection.get('start_time', 0)
        birdnet_end_sec = detection.get('end_time', 0)
        center_sec = (birdnet_start_sec + birdnet_end_sec) / 2.0
        target_start_sec_det = center_sec - (config.TARGET_DURATION / 2.0)
        target_end_sec_det = center_sec + (config.TARGET_DURATION / 2.0)
        final_start_idx_det = max(0, int(target_start_sec_det * config.FS))
        final_end_idx_det = min(len(audio_long), int(target_end_sec_det * config.FS))

        extracted_audio_segment = audio_long[final_start_idx_det:final_end_idx_det]

        if len(extracted_audio_segment) >= min_samples:
            return _pad_or_tile_audio(extracted_audio_segment, target_samples) # Ensure exact length
        else:
            return None # Indicate failure if segment too short after extraction
    except Exception as e:
        print(f"Warning: Error during BirdNET chunk extraction: {e}")
        return None

def _extract_random_chunk(audio_long, target_samples):
    """Extracts a random 5s chunk."""
    if len(audio_long) < target_samples: 
        # This case should ideally be handled before calling, but return padded if needed
        print(f"Warning: _extract_random_chunk called with audio shorter than target. Padding.")
        return _pad_or_tile_audio(audio_long, target_samples)
        
    max_start_idx_5s = len(audio_long) - target_samples
    start_idx_5s = random.randint(0, max_start_idx_5s)
    return audio_long[start_idx_5s : start_idx_5s + target_samples]

def _generate_spectrogram_from_chunk(audio_chunk_5s, config):
    """Generates and resizes a mel spectrogram from a 5s audio chunk."""
    try:
        raw_spec_5s_chunk = utils.audio2melspec(audio_chunk_5s, config)
        if raw_spec_5s_chunk is None:
            return None
        
        # Resize to target shape
        resized_spec_5s = cv2.resize(raw_spec_5s_chunk, (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
        if resized_spec_5s.shape == tuple(config.TARGET_SHAPE):
            return resized_spec_5s.astype(np.float32)
        else:
            print(f"Warning: Resized spec has wrong shape {resized_spec_5s.shape} (expected {config.TARGET_SHAPE})")
            return None
    except Exception as e_spec_5s:
        # Optional: Log error e_spec_5s
        # print(f"Warning: Spectrogram generation/resize failed for chunk: {e_spec_5s}")
        return None

def _process_primary_for_chunking(args):
    """Worker using helper functions to generate spectrograms for one primary audio file."""
    primary_filepath, samplename, config, fabio_intervals, vad_intervals, primary_filename, \
        class_name, scientific_name, birdnet_dets_for_file, UNCOVERED_AVES_SCIENTIFIC_NAME = args

    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples = int(0.5 * config.FS)

    try:
        # 1. Load and clean audio
        relevant_audio, error_msg = _load_and_clean_audio(
            primary_filepath, primary_filename, config, fabio_intervals, vad_intervals,
            class_name, scientific_name, UNCOVERED_AVES_SCIENTIFIC_NAME, min_samples
        )
        if error_msg:
            return samplename, None, error_msg

        # 2. Determine strategy and handle short audio
        original_relevant_len = len(relevant_audio)
        is_originally_short = original_relevant_len < target_samples
        
        audio_for_processing = relevant_audio # Will be padded if short
        if is_originally_short:
            audio_for_processing = _pad_or_tile_audio(relevant_audio, target_samples)

        use_birdnet_strategy = (
            class_name == 'Aves' and
            scientific_name != UNCOVERED_AVES_SCIENTIFIC_NAME and
            birdnet_dets_for_file is not None and
            len([d for d in birdnet_dets_for_file if isinstance(d, dict)]) > 0
        )

        # 3. Generate Spectrogram(s)
        final_specs_list = []

        if not use_birdnet_strategy:
            # Strategy 1: Non-BirdNET -> Single spectrogram from (potentially padded) full audio
            spec = _generate_spectrogram_from_chunk(audio_for_processing, config)
            if spec is not None:
                final_specs_list.append(spec)
        else:
            # Strategy 2: BirdNET -> Multiple versions
            sorted_detections = []
            try:
                sorted_detections = sorted(
                    [d for d in birdnet_dets_for_file if isinstance(d, dict) and 'confidence' in d],
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )
            except Exception as e_sort:
                print(f"Warning: Error sorting BirdNET detections for {samplename}: {e_sort}.")

            num_versions_to_generate = config.PRECOMPUTE_VERSIONS
            for i in range(num_versions_to_generate):
                chunk_5s = None
                if is_originally_short:
                    chunk_5s = audio_for_processing # Use the pre-padded 5s version
                else: # Original audio was long enough
                    if i < len(sorted_detections):
                        chunk_5s = _extract_birdnet_chunk(relevant_audio, sorted_detections[i], config, min_samples, target_samples)
                    # Fallback to random ONLY if BirdNET failed/unavailable AND original was long
                    if chunk_5s is None: 
                        chunk_5s = _extract_random_chunk(relevant_audio, target_samples)

                # Generate spec from the obtained chunk
                if chunk_5s is not None:
                    spec = _generate_spectrogram_from_chunk(chunk_5s, config)
                    if spec is not None:
                        final_specs_list.append(spec)

        # 4. Stack results and return
        if not final_specs_list:
            return samplename, None, "No valid spectrograms generated."

        try:
            final_specs_array = np.stack(final_specs_list, axis=0) # Shape (N, H, W)
            return samplename, final_specs_array, None
        except Exception as e_stack:
            return samplename, None, f"Error stacking specs for {samplename}: {e_stack}"

    except Exception as e_main:
        # Catch any other unexpected errors in the main flow
        tb_str = traceback.format_exc()
        return samplename, None, f"Outer error processing {primary_filepath}: {e_main}\n{tb_str}"

def _load_auxiliary_data(config):
    """Loads auxiliary data (VAD, Fabio, BirdNET, Taxonomy)."""
    fabio_intervals = {}
    vad_intervals = {}
    all_birdnet_detections = {}
    taxonomy_df = pd.DataFrame()

    # Load VAD/Fabio Intervals (only if REMOVE_SPEECH_INTERVALS is True)
    if config.REMOVE_SPEECH_INTERVALS:
        print("REMOVE_SPEECH_INTERVALS is True. Loading VAD/Fabio intervals.")
        try:
            fabio_df = pd.read_csv(config.FABIO_CSV_PATH)
            fabio_intervals = {row['filename']: (row['start'], row['stop'])
                               for _, row in fabio_df.iterrows() if 'filename' in row and 'start' in row and 'stop' in row}
            print(f"Loaded Fabio intervals for {len(fabio_intervals)} files.")
        except FileNotFoundError: print(f"Info: Fabio intervals file not found at {config.FABIO_CSV_PATH}. Skipping.")
        except Exception as e: print(f"Warning: Could not load Fabio intervals: {e}")

        try:
            with open(config.TRANSFORMED_VOICE_DATA_PKL_PATH, 'rb') as f:
                    vad_data = pickle.load(f)
                    if isinstance(vad_data, dict): vad_intervals = {k: v for k, v in vad_data.items() if isinstance(v, list)}
                    else: print(f"Warning: VAD pickle file does not contain a dictionary.")
            print(f"Loaded VAD intervals for {len(vad_intervals)} files.")
        except FileNotFoundError: print(f"Info: VAD intervals pickle file not found at {config.TRANSFORMED_VOICE_DATA_PKL_PATH}. Skipping.")
        except Exception as e: print(f"Warning: Could not load VAD intervals: {e}")
        
    # Load BirdNET Detections
    print(f"\nAttempting to load BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            all_birdnet_detections = {key: data[key] for key in data.files}
        print(f"Successfully loaded BirdNET detections for {len(all_birdnet_detections)} files.")
    except FileNotFoundError: print(f"Warning: BirdNET detections file not found. Proceeding without BirdNET guidance.")
    except Exception as e: print(f"Warning: Error loading BirdNET detections NPZ: {e}. Proceeding without BirdNET guidance.")

    # Load Taxonomy for Class/Scientific Name
    print("\nLoading taxonomy data...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        required_taxonomy_cols = {'primary_label', 'class_name', 'scientific_name'}
        if not required_taxonomy_cols.issubset(taxonomy_df.columns):
            print(f"Error: Taxonomy file missing required columns: {required_taxonomy_cols - set(taxonomy_df.columns)}")
            taxonomy_df = pd.DataFrame() # Reset on error
    except Exception as e: print(f"Error loading taxonomy file: {e}")
    
    return fabio_intervals, vad_intervals, all_birdnet_detections, taxonomy_df

def _prepare_dataframe_for_processing(df, taxonomy_df):
    """Merges taxonomy data and checks required columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input dataframe is invalid or empty.")
        return None

    required_input_cols = {'filename', 'samplename', 'primary_label'}
    if not required_input_cols.issubset(df.columns):
        print(f"Error: Input dataframe missing required columns: {required_input_cols - set(df.columns)}")
        return None

    df_processed = df.copy()

    if not taxonomy_df.empty and 'primary_label' in df_processed.columns:
        try:
            df_processed = pd.merge(
                df_processed,
                taxonomy_df[['primary_label', 'class_name', 'scientific_name']],
                on='primary_label',
                how='left'
            )
            print("Merged taxonomy data into working dataframe.")
            if df_processed['class_name'].isnull().any(): print("Warning: Some rows have null 'class_name' after merge.")
            if df_processed['scientific_name'].isnull().any(): print("Warning: Some rows have null 'scientific_name' after merge.")
        except Exception as e_merge:
             print(f"Error merging taxonomy data: {e_merge}. Proceeding without class/scientific names.")
             if 'class_name' not in df_processed.columns: df_processed['class_name'] = None
             if 'scientific_name' not in df_processed.columns: df_processed['scientific_name'] = None
    else:
        print("Warning: Taxonomy data empty or primary_label missing. Class/Scientific name info unavailable.")
        if 'class_name' not in df_processed.columns: df_processed['class_name'] = None
        if 'scientific_name' not in df_processed.columns: df_processed['scientific_name'] = None

    # Final check for all necessary columns before task creation
    required_final_cols = {'filename', 'samplename', 'primary_label', 'class_name', 'scientific_name'}
    missing_cols = required_final_cols - set(df_processed.columns)
    if missing_cols:
        print(f"CRITICAL ERROR: Dataframe missing required columns for processing after merge: {missing_cols}")
        return None # Indicate failure

    return df_processed

def _create_processing_tasks(df, config, fabio_intervals, vad_intervals, all_birdnet_detections):
    """Creates a list of argument tuples for the multiprocessing worker."""
    tasks = []
    skipped_path_count = 0
    UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti' # Hardcode locally

    for _, row in df.iterrows():
        primary_filename = row['filename']
        samplename = row['samplename']
        class_name = row['class_name'] 
        scientific_name = row['scientific_name'] 

        # Find audio file path
        potential_main_path = os.path.join(config.train_audio_dir, primary_filename)
        potential_rare_path = os.path.join(config.train_audio_rare_dir, primary_filename) if config.USE_RARE_DATA else None
        primary_filepath = None
        if os.path.exists(potential_main_path): primary_filepath = potential_main_path
        elif potential_rare_path and os.path.exists(potential_rare_path): primary_filepath = potential_rare_path

        if primary_filepath:
            birdnet_dets_for_file = all_birdnet_detections.get(primary_filename, None)
            tasks.append((
                primary_filepath, samplename, config, fabio_intervals, vad_intervals, 
                primary_filename, class_name, scientific_name, birdnet_dets_for_file,
                UNCOVERED_AVES_SCIENTIFIC_NAME
            ))
        else:
            skipped_path_count += 1
            
    if skipped_path_count > 0:
        print(f"Warning: Skipped {skipped_path_count} files because audio path not found.")
        
    return tasks, skipped_path_count

def _run_parallel_processing(tasks, worker_func, num_workers):
    """Runs the multiprocessing pool and collects results."""
    results_list = []
    if not tasks:
        print("No tasks to process.")
        return results_list

    print(f"Starting parallel processing with {num_workers} workers for {len(tasks)} tasks.")
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(worker_func, tasks)
            for result in tqdm(results_iterator, total=len(tasks), desc="Generating Specs"):
                results_list.append(result) # Append tuple (samplename, spec_array, error_msg)
    except Exception as e:
        print(f"\nCRITICAL ERROR during multiprocessing: {e}")
        print(traceback.format_exc())
        # Depending on desired behavior, could raise e or return partial results
        raise # Re-raise the exception to stop the script

    print() # Newline after progress indicator
    return results_list

def _aggregate_results(results_list):
    """Processes the list of results from workers into final dictionary and stats."""
    grouped_results = {}
    processed_count = 0
    error_count = 0
    skipped_files = 0
    errors_list = []

    for result_tuple in results_list:
        samplename, spec_array, error_msg = result_tuple
        if error_msg:
            errors_list.append(f"{samplename}: {error_msg}")
            error_count += 1
        elif spec_array is not None and spec_array.ndim == 3: # Expect (N, H, W)
            grouped_results[samplename] = spec_array
            processed_count += 1
        else:
            # Worker finished but produced no valid specs or returned None
            # print(f"Debug: Worker for {samplename} yielded no valid spectrogram array.")
            skipped_files += 1
            errors_list.append(f"{samplename}: Skipped (No valid spec array)")

    return grouped_results, processed_count, error_count, skipped_files, errors_list

def _report_summary(total_tasks, processed_count, error_count, skipped_files, errors_list):
    """Prints a summary of the processing results."""
    print(f"\n--- Processing Summary ---")
    print(f"Attempted to process {total_tasks} files.")
    print(f"Successfully generated spectrograms for {processed_count} primary files.") 
    if processed_count > 0:
        # This part requires grouped_results, might need adjustment if called separately
        # total_chunks = sum(arr.shape[0] for arr in grouped_results.values() if arr is not None)
        # print(f"Total spectrogram chunks generated: {total_chunks}") # Needs grouped_results
        pass # Cannot calculate total chunks without grouped_results here
    print(f"Encountered errors for {error_count} primary files.")
    if skipped_files > 0:
         print(f"{skipped_files} primary files yielded no valid spectrograms (skipped).")
    if errors_list:
        print("\n--- Errors/Skips Encountered (sample) ---")
        for err in errors_list[:20]: print(err)
        if len(errors_list) > 20: print(f"... and {len(errors_list) - 20} more.")
    print("-" * 26)

def _save_spectrogram_npz(grouped_results, config, project_root):
    """Determines path and saves the spectrogram dictionary to an NPZ file."""
    if not grouped_results:
        print("No spectrograms were successfully generated to save.")
        return

    num_saved_files = len(grouped_results)
    total_chunks = sum(arr.shape[0] for arr in grouped_results.values() if arr is not None and hasattr(arr, 'shape'))

    # Determine the output NPZ path based on debug mode
    if config.debug:
        debug_output_dir = os.path.join(project_root, "outputs", "preprocessed")
        output_npz_path = os.path.join(debug_output_dir, "debug_spectrograms2.npz")
    else:
        output_npz_path = config.PREPROCESSED_NPZ_PATH

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_npz_path)
    os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

    print(f"Saving {num_saved_files} primary file entries (total {total_chunks} chunks) to: {output_npz_path}")
    start_save = time.time()
    try:
        np.savez_compressed(output_npz_path, **grouped_results)
        end_save = time.time()
        print(f"NPZ saving took {end_save - start_save:.2f} seconds.")
    except Exception as e_save:
        print(f"CRITICAL ERROR saving NPZ file: {e_save}")
        print(traceback.format_exc())
        print("Spectrogram data is likely lost. Check disk space and permissions.")
        # Depending on desired behavior, might want to raise or exit
        sys.exit(1)

def generate_and_save_spectrograms(df, config):
    """Generates spectrogram chunks using helper functions and saves to NPZ."""
    start_time_gen = time.time()
    print("\n--- 2. Generating Spectrogram Chunks ---")

    # Handle debug file limit
    if config.debug and config.debug_limit_files > 0:
        print(f"DEBUG MODE: Limiting preprocessing to the first {config.debug_limit_files} files.")
        df = df.head(config.debug_limit_files).copy()
        if df.empty:
             print("DEBUG MODE: Dataframe is empty after limiting. Cannot proceed.")
             return None # Indicate failure or empty result

    # 1. Load auxiliary data
    fabio_intervals, vad_intervals, all_birdnet_detections, taxonomy_df = _load_auxiliary_data(config)

    # 2. Prepare main DataFrame
    df_processed = _prepare_dataframe_for_processing(df, taxonomy_df)
    if df_processed is None or df_processed.empty: # Check if preparation failed
         print("DataFrame preparation failed or resulted in empty DataFrame. Cannot proceed.")
         return None

    # 3. Create tasks for workers
    tasks, skipped_path_count = _create_processing_tasks(
        df_processed, config, fabio_intervals, vad_intervals, all_birdnet_detections
    )
    if not tasks:
        print("No valid processing tasks created. Cannot proceed.")
        return None

    # 4. Run parallel processing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    results_list = _run_parallel_processing(tasks, _process_primary_for_chunking, num_workers)

    # 5. Aggregate results
    grouped_results, processed_count, error_count, skipped_files, errors_list = _aggregate_results(results_list)

    # 6. Report summary statistics
    _report_summary(len(tasks), processed_count, error_count, skipped_files, errors_list)

    # 7. Save NPZ file (pass project_root needed for debug path construction)
    _save_spectrogram_npz(grouped_results, config, project_root)

    end_time_gen = time.time()
    print(f"--- Spectrogram generation and saving finished in {end_time_gen - start_time_gen:.2f} seconds ---")

    return grouped_results # Still return the dictionary if needed upstream

def main(config):
    """Main function to run the preprocessing steps."""
    overall_start = time.time()
    print("Starting BirdCLEF Preprocessing Pipeline...")
    print(f"Configuration: Debug={config.debug}, Seed={config.seed}, UseRare={config.USE_RARE_DATA}, RemoveSpeech={config.REMOVE_SPEECH_INTERVALS}, Versions={config.PRECOMPUTE_VERSIONS}")
    
    df_working = load_and_prepare_metadata(config)

    if df_working is not None and not df_working.empty:
        # Call the refactored main generation function
        # Removed the unnecessary assignment to '_'
        generate_and_save_spectrograms(df_working, config) 
    else:
        print("Metadata loading failed or resulted in empty dataframe. Cannot proceed.")

    overall_end = time.time()
    print(f"\nTotal preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    main(config)