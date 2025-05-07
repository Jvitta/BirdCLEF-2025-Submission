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

def _process_primary_for_chunking(args):
    """Worker to generate multiple 5-second spectrogram chunks for one primary audio file,
       using BirdNET detections if available for eligible Aves species, otherwise random chunks.
    """
    primary_filepath, samplename, config, fabio_intervals, vad_intervals, primary_filename, \
        class_name, scientific_name, birdnet_dets_for_file, UNCOVERED_AVES_SCIENTIFIC_NAME = args

    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples = int(0.5 * config.FS)
    final_specs_array = None

    try:
        primary_audio, _ = librosa.load(primary_filepath, sr=config.FS, mono=True)
        if primary_audio is None or len(primary_audio) < min_samples:
            return samplename, None, f"Primary audio too short/empty: {primary_filepath}"

        # --- Apply VAD/Fabio if configured (on the full audio first) ---
        relevant_audio = primary_audio
        should_apply_vad_fabio = False
        if config.REMOVE_SPEECH_INTERVALS:
            if not config.REMOVE_SPEECH_ONLY_NON_AVES:
                should_apply_vad_fabio = True
            elif class_name != 'Aves' or scientific_name == UNCOVERED_AVES_SCIENTIFIC_NAME:
                 should_apply_vad_fabio = True
        
        if should_apply_vad_fabio:
            # --- Start VAD/Fabio logic --- #
            cleaned_audio_vad_fabio = None 

            if primary_filename in fabio_intervals:
                start_time_fabio, stop_time_fabio = fabio_intervals[primary_filename]
                start_idx_fabio = max(0, int(start_time_fabio * config.FS))
                end_idx_fabio = min(len(primary_audio), int(stop_time_fabio * config.FS))
                if start_idx_fabio < end_idx_fabio:
                    cleaned_audio_vad_fabio = primary_audio[start_idx_fabio:end_idx_fabio]
            elif primary_filepath in vad_intervals:
                speech_timestamps = vad_intervals[primary_filepath]
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
            # --- End VAD/Fabio logic --- #
            if cleaned_audio_vad_fabio is not None and len(cleaned_audio_vad_fabio) >= min_samples:
                relevant_audio = cleaned_audio_vad_fabio
            # else relevant_audio remains primary_audio

        if len(relevant_audio) < min_samples: # Check again after VAD
             return samplename, None, f"Primary audio too short after VAD/Fabio: {primary_filepath}"

        # --- Store original length and prepare audio for strategies ---
        original_relevant_len = len(relevant_audio)
        audio_for_strategy = relevant_audio # This will be modified if original is short

        if original_relevant_len < target_samples:
            # Tile/pad relevant_audio to make audio_for_strategy exactly 5s
            n_copy = math.ceil(target_samples / original_relevant_len) # Use original_relevant_len for calculation
            audio_for_strategy = np.tile(relevant_audio, n_copy)[:target_samples]
            if len(audio_for_strategy) < target_samples: # Final padding
                audio_for_strategy = np.pad(audio_for_strategy, (0, target_samples - len(audio_for_strategy)), mode='constant')
        
        # --- Determine processing strategy --- # 
        use_birdnet_strategy = (
            class_name == 'Aves' and 
            scientific_name != UNCOVERED_AVES_SCIENTIFIC_NAME and 
            birdnet_dets_for_file is not None and 
            len([d for d in birdnet_dets_for_file if isinstance(d, dict)]) > 0
        )

        # --- Strategy 1: Full Random (Non-Aves, Uncovered Aves, or Aves with no valid BirdNET detections) --- #
        if not use_birdnet_strategy:
            try:
                raw_full_spec = utils.audio2melspec(audio_for_strategy, config)
                if raw_full_spec is not None:
                    # Resize to target shape for model
                    # cv2.resize expects (width, height) for dsize
                    resized_spec = cv2.resize(raw_full_spec, (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
                    if resized_spec.shape == tuple(config.TARGET_SHAPE): # Check final resized shape
                        final_specs_array = np.expand_dims(resized_spec.astype(np.float32), axis=0)
                    else:
                        # This error would now mean resizing failed or gave unexpected shape
                        return samplename, None, f"Resized spec has wrong shape {resized_spec.shape} for {samplename} (expected {config.TARGET_SHAPE})"
                else:
                    return samplename, None, f"Raw full spec generation failed (None) for {samplename}" # Changed error message
            except Exception as e_full_spec:
                return samplename, None, f"Error generating or resizing full spec for {samplename}: {e_full_spec}"
        # --- Strategy 2: BirdNET-Guided (with 5s random fallbacks if needed) --- #
        else:
            sorted_detections = []
            try:
                sorted_detections = sorted(
                    [d for d in birdnet_dets_for_file if isinstance(d, dict) and 'confidence' in d],
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )
            except Exception as e_sort:
                print(f"Warning: Error sorting BirdNET detections for {samplename}: {e_sort}. Assuming no valid detections.")

            generated_chunks_list = []
            num_versions_to_generate = config.PRECOMPUTE_VERSIONS

            was_originally_short = (original_relevant_len < target_samples)

            for i in range(num_versions_to_generate):
                primary_chunk_5s = None

                if was_originally_short:
                    primary_chunk_5s = audio_for_strategy
                else:
                    source_for_5s_extraction = audio_for_strategy # This is the long audio

                    is_birdnet_chunk_for_iteration = i < len(sorted_detections)
                    if is_birdnet_chunk_for_iteration:
                        try:
                            detection = sorted_detections[i]
                            birdnet_start_sec = detection.get('start_time', 0)
                            birdnet_end_sec = detection.get('end_time', 0)
                            center_sec = (birdnet_start_sec + birdnet_end_sec) / 2.0
                            target_start_sec_det = center_sec - (config.TARGET_DURATION / 2.0)
                            target_end_sec_det = center_sec + (config.TARGET_DURATION / 2.0)
                            final_start_idx_det = max(0, int(target_start_sec_det * config.FS))
                            final_end_idx_det = min(len(source_for_5s_extraction), int(target_end_sec_det * config.FS))

                            extracted_audio_segment = source_for_5s_extraction[final_start_idx_det:final_end_idx_det]

                            if len(extracted_audio_segment) >= min_samples:
                                if len(extracted_audio_segment) < target_samples:
                                    pad_width_det = target_samples - len(extracted_audio_segment)
                                    primary_chunk_5s = np.pad(extracted_audio_segment, (0, pad_width_det), mode='constant')
                                elif len(extracted_audio_segment) > target_samples:
                                    primary_chunk_5s = extracted_audio_segment[:target_samples]
                                else:
                                    primary_chunk_5s = extracted_audio_segment
                        except Exception as e_birdnet_chunk_5s:
                            print(f"Warning: Error processing BirdNET detection {i} for 5s chunk in {samplename}: {e_birdnet_chunk_5s}.")

                    if primary_chunk_5s is None:
                        max_start_idx_5s = len(source_for_5s_extraction) - target_samples
                        start_idx_5s = random.randint(0, max_start_idx_5s)
                        primary_chunk_5s = source_for_5s_extraction[start_idx_5s : start_idx_5s + target_samples]

                if primary_chunk_5s is not None and len(primary_chunk_5s) == target_samples:
                    try:
                        raw_spec_5s_chunk = utils.audio2melspec(primary_chunk_5s, config)
                        if raw_spec_5s_chunk is not None:
                            # Resize to target shape
                            resized_spec_5s = cv2.resize(raw_spec_5s_chunk, (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
                            if resized_spec_5s.shape == tuple(config.TARGET_SHAPE):
                                generated_chunks_list.append(resized_spec_5s.astype(np.float32))
                            else:
                                # Optional: Log error if resized shape is not as expected
                                print(f"Warning: Resized 5s spec for {samplename} chunk {i} has wrong shape {resized_spec_5s.shape} (expected {config.TARGET_SHAPE})")
                        # else: Optional: Log error if raw_spec_5s_chunk is None
                    except Exception as e_spec_5s:
                        # Optional: Log error e_spec_5s
                        pass

            if generated_chunks_list:
                try:
                    final_specs_array = np.stack(generated_chunks_list, axis=0)
                except Exception as e_stack_5s:
                     return samplename, None, f"Error stacking 5s specs for {samplename}: {e_stack_5s}"

    except Exception as e_main:
        tb_str = traceback.format_exc()
        return samplename, None, f"Error processing {primary_filepath}: {e_main}\n{tb_str}"

    if final_specs_array is None:
        pass

    return samplename, final_specs_array, None


def generate_and_save_spectrograms(df, config):
    """Generates spectrogram chunks using BirdNET detections or random sampling and saves to NPZ."""
    if df is None or df.empty:
        print("Working dataframe is empty, skipping spectrogram generation.")
        return

    # --- Debug Limiting ---
    if config.debug and config.debug_limit_files > 0:
        print(f"\nDEBUG MODE: Limiting preprocessing to the first {config.debug_limit_files} files.")
        df = df.head(config.debug_limit_files).copy()
        if df.empty:
             print("DEBUG MODE: Dataframe is empty after limiting. Exiting.")
             return {}
    # --- End Debug Limiting ---

    # --- Load Supporting Data ---
    # Load VAD/Fabio Intervals (only if REMOVE_SPEECH_INTERVALS is True)
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
        
    # --- Load BirdNET Detections ---
    print(f"\nAttempting to load BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    all_birdnet_detections = {}
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            all_birdnet_detections = {key: data[key] for key in data.files}
        print(f"Successfully loaded BirdNET detections for {len(all_birdnet_detections)} files.")
    except FileNotFoundError:
        print(f"Warning: BirdNET detections file not found at {config.BIRDNET_DETECTIONS_NPZ_PATH}. Proceeding without BirdNET guidance (will use random chunks).")
        # Keep all_birdnet_detections as an empty dict
    except Exception as e:
        print(f"Warning: Error loading BirdNET detections NPZ: {e}. Proceeding without BirdNET guidance.")
        all_birdnet_detections = {} # Ensure it's empty on error
    # --- End BirdNET Loading ---

    # --- Load Taxonomy for Class/Scientific Name ---
    print("\nLoading taxonomy data for class/scientific names...")
    taxonomy_df = pd.DataFrame() # Initialize empty
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        required_taxonomy_cols = {'primary_label', 'class_name', 'scientific_name'}
        if not required_taxonomy_cols.issubset(taxonomy_df.columns):
            print(f"Error: Taxonomy file missing required columns: {required_taxonomy_cols - set(taxonomy_df.columns)}")
            taxonomy_df = pd.DataFrame() # Reset on error
    except Exception as e:
        print(f"Error loading taxonomy file: {e}")

    # Merge class_name and scientific_name into the working dataframe
    if not taxonomy_df.empty and 'primary_label' in df.columns:
        df = pd.merge(
            df,
            taxonomy_df[['primary_label', 'class_name', 'scientific_name']],
            on='primary_label',
            how='left'
        )
        print("Merged taxonomy data into working dataframe.")
        # Check if merge resulted in NaNs for critical columns
        if df['class_name'].isnull().any():
            print("Warning: Some rows have null 'class_name' after merging with taxonomy.")
        if df['scientific_name'].isnull().any():
            print("Warning: Some rows have null 'scientific_name' after merging with taxonomy.")
    else:
        print("Warning: Could not merge taxonomy data. Class/Scientific name information will be unavailable for chunk selection logic.")
        # Add placeholder columns if they don't exist to prevent KeyErrors later
        if 'class_name' not in df.columns: df['class_name'] = None
        if 'scientific_name' not in df.columns: df['scientific_name'] = None

    # --- End Taxonomy Loading ---

    print("\n--- 2. Generating Spectrogram Chunks (Conditional: BirdNET or Random) ---")
    start_time = time.time()
    all_spectrograms = {}
    processed_count = 0
    error_count = 0
    skipped_files = 0
    errors = []

    # --- Setup Multiprocessing Tasks ---
    tasks = []
    skipped_path_count = 0
    # Ensure required columns exist after potential merge issues
    required_cols = {'filename', 'samplename', 'primary_label', 'class_name', 'scientific_name'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"CRITICAL ERROR: Dataframe missing required columns for processing after merge: {missing_cols}")
        sys.exit(1)

    # Hardcode the excluded species name locally for the worker
    UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'

    for index, row in df.iterrows():
        primary_filename = row['filename']
        samplename = row['samplename']
        class_name = row['class_name'] # Get class name
        scientific_name = row['scientific_name'] # Get scientific name

        # Find audio file path
        potential_main_path = os.path.join(config.train_audio_dir, primary_filename)
        potential_rare_path = os.path.join(config.train_audio_rare_dir, primary_filename) if config.USE_RARE_DATA else None
        primary_filepath = None
        if os.path.exists(potential_main_path):
            primary_filepath = potential_main_path
        elif potential_rare_path and os.path.exists(potential_rare_path):
            primary_filepath = potential_rare_path

        if primary_filepath:
            # Get BirdNET detections for this specific file (returns None if not found)
            birdnet_dets_for_file = all_birdnet_detections.get(primary_filename, None)

            tasks.append((
                primary_filepath,
                samplename,
                config,
                fabio_intervals, # Pass loaded intervals
                vad_intervals,   # Pass loaded intervals
                primary_filename,
                class_name,      # Pass class name
                scientific_name, # Pass scientific name
                birdnet_dets_for_file, # Pass detections list (or None)
                UNCOVERED_AVES_SCIENTIFIC_NAME # Pass excluded name
            ))
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
                # Unpack result: samplename, spec_array (N, H, W) or None, error_msg
                samplename, spec_array, error_msg = result # Use the new return structure
                if error_msg:
                    errors.append(f"{samplename}: {error_msg}")
                    error_count += 1
                elif spec_array is not None and spec_array.ndim == 3: # Check if we got a valid stacked array
                    grouped_results[samplename] = spec_array # Store the stacked array directly
                    processed_count += 1
                else:
                    # Worker finished but produced no valid specs or returned None
                    # print(f"Debug: Worker for {samplename} yielded no valid spectrogram array.")
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
    # grouped_results now holds {samplename: stacked_np_array (N, H, W)}
    if grouped_results:
        num_saved_files = len(grouped_results)
        total_chunks = sum(arr.shape[0] for arr in grouped_results.values() if arr is not None)
        
        if config.debug:
            debug_output_dir = os.path.join(project_root, "outputs", "preprocessed")
            output_npz_path = os.path.join(debug_output_dir, "debug_spectrograms.npz")
        else:
            output_npz_path = config.PREPROCESSED_NPZ_PATH

        output_dir = os.path.dirname(output_npz_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        print(f"Saving {num_saved_files} primary file entries (total {total_chunks} chunks) to: {output_npz_path}") 
        start_save = time.time()
        try:
            # Save the dictionary directly - values are already numpy arrays
            np.savez_compressed(output_npz_path, **grouped_results)
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