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
from src.models.efficient_at.preprocess import AugmentMelSTFT
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import config 
import src.utils.utils as utils 

warnings.filterwarnings("ignore")

# --- Seed setting --- #
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

parser = argparse.ArgumentParser(description="Preprocess audio data for BirdCLEF.")
parser.add_argument("--mode", type=str, choices=["train", "val"], default="train",
                    help="Preprocessing mode: 'train' for augmented spectrograms, 'val' for fixed-setting spectrograms.")
cmd_args = parser.parse_args()

efficient_at_spectrogram_generator = AugmentMelSTFT(
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

output_npz_path = ""

if cmd_args.mode == "train":
    print("--- Running Preprocessing in TRAIN mode (augmented) ---")
    output_npz_path = config.PREPROCESSED_NPZ_PATH
else:
    print("--- Running Preprocessing in VAL mode (fixed settings) ---")
    efficient_at_spectrogram_generator.eval()
    output_npz_path = config.PREPROCESSED_NPZ_PATH_VAL

os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)

def load_and_prepare_metadata(config):
    """Loads and prepares the metadata dataframe based on configuration."""
    print("--- 1. Loading and Preparing Metadata ---")
    start_time = time.time()

    try:
        df_train_full = pd.read_csv(config.train_csv_path)
        df_train = df_train_full[['filename', 'primary_label']].copy()
        df_train['filename'] = df_train['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
        print(f"Loaded and filtered main metadata: {df_train.shape[0]} rows")
    except Exception as e:
        print(f"CRITICAL ERROR loading main metadata {config.train_csv_path}: {e}. Exiting.")
        sys.exit(1)

    if config.USE_RARE_DATA:
        print("USE_RARE_DATA is True. Loading rare species metadata.")
        try:
            df_rare_full = pd.read_csv(config.train_rare_csv_path) 
            df_rare = df_rare_full[['filename', 'primary_label']].copy()
            df_rare['filename'] = df_rare['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
            print(f"Loaded and filtered rare metadata: {df_rare.shape[0]} rows")

            df_combined = pd.concat([df_train, df_rare], ignore_index=True)
            # Drop duplicates based on filename, keeping the first occurrence (main data preferred)
            df_working = df_combined.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)
            print(f"Working metadata size (main + rare, unique filenames): {df_working.shape[0]} rows")
        except Exception as e:
            print(f"Error loading or processing rare species metadata: {e}. Proceeding with main data only.")
            df_working = df_train 
            print(f"Working metadata size (main only): {df_working.shape[0]} rows")
    else:
        print("USE_RARE_DATA is False. Using main metadata only.")
        df_working = df_train 
        print(f"Working metadata size: {df_working.shape[0]} rows")

    # --- Filter out files where all manual annotations are marked as low quality ---
    if config.EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS:
        print("EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS is True. Applying filter.")
        if not df_working.empty: # Proceed only if df_working is not empty
            try:
                manual_ann_df = pd.read_csv(config.ANNOTATED_SEGMENTS_CSV_PATH)
                if 'filename' in manual_ann_df.columns and 'is_low_quality' in manual_ann_df.columns:
                    manual_ann_df['filename'] = manual_ann_df['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
                    
                    # Robustly convert 'is_low_quality' to boolean
                    if manual_ann_df['is_low_quality'].dtype == 'object' or pd.api.types.is_string_dtype(manual_ann_df['is_low_quality']):
                        manual_ann_df['is_low_quality'] = manual_ann_df['is_low_quality'].astype(str).str.lower().map({
                            'true': True, 'yes': True, '1': True, 't': True,
                            'false': False, 'no': False, '0': False, 'f': False,
                            'nan': False, '': False, 'none': False, '<na>': False
                        }).fillna(False) # Fill any unmapped (originally NaN or other strings) as False
                    manual_ann_df['is_low_quality'] = manual_ann_df['is_low_quality'].fillna(False).astype(bool)

                    annotated_files_present_in_df_working = manual_ann_df[manual_ann_df['filename'].isin(df_working['filename'])]
                    
                    if not annotated_files_present_in_df_working.empty:
                        # Group by filename and check if all 'is_low_quality' are True for annotations belonging to files in df_working
                        file_quality_summary = annotated_files_present_in_df_working.groupby('filename')['is_low_quality'].agg(['all', 'count'])
                        
                        # Identify files where count > 0 (i.e., has annotations) and all annotations are low quality
                        # The .all() column from agg will be True if all are True, or if the group was empty (which we filter by count > 0 implicitly with isin earlier, but explicitly here is safer)
                        low_quality_files_series = file_quality_summary[file_quality_summary['all'] == True]
                        filenames_to_exclude = low_quality_files_series.index.tolist()

                        if filenames_to_exclude:
                            initial_rows = df_working.shape[0]
                            df_working = df_working[~df_working['filename'].isin(filenames_to_exclude)]
                            excluded_count = initial_rows - df_working.shape[0]
                            if excluded_count > 0:
                                print(f"INFO: Excluded {excluded_count} files from processing because all their manual annotations were marked as low quality.")
                    else:
                        print(f"Info: No files listed in {config.ANNOTATED_SEGMENTS_CSV_PATH} matched files currently in the working dataframe. No files excluded based on low quality annotations.")
                else:
                    print(f"Info: 'filename' or 'is_low_quality' column not found in {config.ANNOTATED_SEGMENTS_CSV_PATH}. No files excluded based on low quality annotations.")
            except FileNotFoundError:
                print(f"Info: Manual annotations file '{config.ANNOTATED_SEGMENTS_CSV_PATH}' not found. No files excluded based on low quality annotations.")
            except Exception as e_lq_filter:
                print(f"Warning: Could not filter files based on low quality annotations: {e_lq_filter}")
                print(traceback.format_exc()) # Print stack trace for easier debugging
    else:
        print("EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS is False. Skipping filter.")
    # --- End filter ---

    df_working['samplename'] = df_working['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])

    # --- Calculate per-species file counts --- #
    species_file_counts = {}
    if 'primary_label' in df_working.columns and 'filename' in df_working.columns:
        try:
            species_file_counts = df_working.groupby('primary_label')['filename'].nunique().to_dict()
            print(f"Calculated file counts for {len(species_file_counts)} unique species.")
        except Exception as e_counts:
            print(f"Warning: Could not calculate species file counts: {e_counts}. Dynamic chunking might not work as expected.")
    else:
        print("Warning: 'primary_label' or 'filename' not in df_working. Cannot calculate species_file_counts for dynamic chunking.")

    end_time = time.time()
    print(f"Metadata preparation finished in {end_time - start_time:.2f} seconds.")
    return df_working, species_file_counts

def _load_and_clean_audio(filepath, filename, config, fabio_intervals, vad_intervals, min_samples):
    """
    Loads audio. If REMOVE_SPEECH_INTERVALS is True, also provides a cleaned version for random chunks.
    Returns:
        final_primary_audio (np.array): Original audio. None if loading failed or initially too short.
        audio_for_random_chunks (np.array): Cleaned audio if successful and non-empty, else final_primary_audio. None if final_primary_audio is None.
        fabio_applied (bool): True if Fabio speech removal was applied.
        vad_applied (bool): True if VAD speech removal was applied.
        error_msg (str): Error message, or None.
    """
    fabio_applied_flag = False
    vad_applied_flag = False
    try:
        primary_audio_candidate, _ = librosa.load(filepath, sr=config.FS, mono=True)
        if primary_audio_candidate is None or len(primary_audio_candidate) < min_samples:
            return None, None, fabio_applied_flag, vad_applied_flag, f"Primary audio load failed or too short: {filepath}"
    except Exception as e_load:
        return None, None, fabio_applied_flag, vad_applied_flag, f"Error loading audio {filepath}: {e_load}"

    # Initialize outputs
    final_primary_audio = primary_audio_candidate
    # Default audio_for_random_chunks to a copy of final_primary_audio.
    # It will be replaced if cleaning is successful and applicable.
    audio_for_random_chunks = final_primary_audio.copy() 

    if config.REMOVE_SPEECH_INTERVALS:
        temp_cleaned_audio = None # This will store the result of VAD/Fabio

        # Attempt Fabio speech removal first (if applicable)
        if filename in fabio_intervals:
            start_time_fabio, stop_time_fabio = fabio_intervals[filename]
            start_idx_fabio = max(0, int(start_time_fabio * config.FS))
            end_idx_fabio = min(len(final_primary_audio), int(stop_time_fabio * config.FS))

            if start_idx_fabio < end_idx_fabio:
                temp_cleaned_audio = final_primary_audio[start_idx_fabio:end_idx_fabio]
                fabio_applied_flag = True # Set flag
        # Else, attempt VAD speech removal (if applicable)
        elif filename in vad_intervals: 
            speech_timestamps = vad_intervals[filename]

            if speech_timestamps: # Ensure there are timestamps to process
                non_speech_segments = []
                current_pos_sec = 0.0
                audio_duration_sec = len(final_primary_audio) / config.FS

                try: 
                    speech_timestamps.sort(key=lambda x: x.get('start', 0))
                except: 
                    speech_timestamps = [] # Reset if sorting fails (e.g., malformed data)

                for segment in speech_timestamps:
                    if not isinstance(segment, dict) or 'start' not in segment or 'end' not in segment: 
                        continue 
                    start_speech_sec = segment['start']
                    end_speech_sec = segment['end']

                    # Add segment before current speech
                    if start_speech_sec > current_pos_sec:
                        start_idx_segment = max(0, int(current_pos_sec * config.FS))
                        end_idx_segment = min(len(final_primary_audio), int(start_speech_sec * config.FS))
                        if end_idx_segment > start_idx_segment: 
                            non_speech_segments.append(final_primary_audio[start_idx_segment:end_idx_segment])
                    
                    current_pos_sec = max(current_pos_sec, end_speech_sec) # Move past the speech segment

                # Add segment after the last speech segment until end of audio
                if current_pos_sec < audio_duration_sec:
                    start_idx_segment = max(0, int(current_pos_sec * config.FS))
                    if start_idx_segment < len(final_primary_audio): 
                        non_speech_segments.append(final_primary_audio[start_idx_segment:])

                if non_speech_segments:
                    # Filter out any potentially empty segments before concatenation
                    non_speech_segments = [s for s in non_speech_segments if len(s) > 0]
                    if non_speech_segments: 
                        temp_cleaned_audio = np.concatenate(non_speech_segments)
                        vad_applied_flag = True # Set flag
                    else: 
                        temp_cleaned_audio = np.array([]) # No valid non-speech segments found
                else: 
                    # No non-speech segments found (e.g. entire file marked as speech, or no VAD data)
                    temp_cleaned_audio = np.array([]) 
        
        # If cleaning was attempted and resulted in a non-empty audio segment, use it.
        if temp_cleaned_audio is not None and len(temp_cleaned_audio) > 0:
            audio_for_random_chunks = temp_cleaned_audio

    if final_primary_audio is None: # Should not happen if initial check passed, but as safeguard
        return None, None, fabio_applied_flag, vad_applied_flag, "Primary audio became None unexpectedly."

    return final_primary_audio, audio_for_random_chunks, fabio_applied_flag, vad_applied_flag, None

def _pad_or_tile_audio(audio_data, target_samples):
    """Pads or tiles audio to target_samples length."""
    current_len = len(audio_data)
    if current_len == target_samples:
        return audio_data
    elif current_len < target_samples:
        n_copy = math.ceil(target_samples / current_len) 
        padded_audio = np.tile(audio_data, n_copy)[:target_samples]
        if len(padded_audio) < target_samples: 
            padded_audio = np.pad(padded_audio, (0, target_samples - len(padded_audio)), mode='constant')
        return padded_audio
    else: 
        return audio_data[:target_samples]

def _extract_shifted_chunk_strategy(audio_long, center_time_s, config, min_samples, target_samples_5s):
    """
    Extracts a chunk of target_samples_5s.
    If the centered window goes out of bounds, it's shifted to fit.
    If audio is shorter than target_samples_5s, it's padded/tiled.
    """
    audio_len_samples = len(audio_long)

    if audio_len_samples < min_samples:
        return None # Too short to be usable, even with padding

    # If the entire audio is shorter than the target chunk duration, pad/tile the whole thing.
    if audio_len_samples < target_samples_5s:
        return _pad_or_tile_audio(audio_long[:], target_samples_5s) # Use a copy

    # At this point, audio_len_samples >= target_samples_5s

    # Calculate desired start based on the center point
    chunk_center_samples = int(center_time_s * config.FS)
    # Calculate start assuming the chunk is centered around chunk_center_samples
    desired_start_samples = chunk_center_samples - (target_samples_5s // 2)
    
    # Determine the actual slice based on boundaries
    if desired_start_samples < 0:
        # Shift window to start at the beginning of the file
        final_start_idx = 0
        final_end_idx = target_samples_5s
    elif (desired_start_samples + target_samples_5s) > audio_len_samples:
        # Shift window to end at the end of the file
        final_end_idx = audio_len_samples
        final_start_idx = audio_len_samples - target_samples_5s
    else:
        # Window fits perfectly by centering
        final_start_idx = desired_start_samples
        final_end_idx = desired_start_samples + target_samples_5s
    
    extracted_audio_segment = audio_long[final_start_idx:final_end_idx]
    
    # The segment should now be exactly target_samples_5s long because
    # audio_len_samples >= target_samples_5s and the shifting logic ensures this.
    # _pad_or_tile_audio here will just return the segment as is if it's the correct length.
    return _pad_or_tile_audio(extracted_audio_segment, target_samples_5s)

def _extract_manual_annotation_chunk(audio_long, center_time_s, config, min_samples, target_samples_5s):
    """Extracts a 5s audio chunk centered around a manual annotation center time, using shifting strategy."""
    try:
        return _extract_shifted_chunk_strategy(audio_long, center_time_s, config, min_samples, target_samples_5s)
    except Exception as e:
        print(f"Warning: Error during manual annotation 5s chunk extraction: {e}") # Optional
        return None

def _extract_birdnet_chunk(audio_long, detection, config, min_samples, target_samples_5s):
    """
    Extracts a 5s audio chunk centered around a BirdNET detection, using shifting strategy.
    Pads if necessary to ensure target_samples_5s length.
    Returns None on failure or if resulting chunk is too short before padding.
    """
    try:
        birdnet_start_sec = detection.get('start_time', 0)
        birdnet_end_sec = detection.get('end_time', 0)
        center_sec = (birdnet_start_sec + birdnet_end_sec) / 2.0
        return _extract_shifted_chunk_strategy(audio_long, center_sec, config, min_samples, target_samples_5s)
    except Exception as e:
        # print(f"Warning: Error during BirdNET 5s chunk extraction: {e}") # Optional
        return None

def _extract_random_chunk(audio_long, target_samples):
    """Extracts a random 5s chunk."""
    if len(audio_long) < target_samples: 
        return _pad_or_tile_audio(audio_long, target_samples)
        
    max_start_idx_5s = len(audio_long) - target_samples
    start_idx_5s = random.randint(0, max_start_idx_5s)
    return audio_long[start_idx_5s : start_idx_5s + target_samples]

def _generate_spectrogram_from_chunk(audio_chunk_5s, config_obj):
    """Generates a mel spectrogram from an audio chunk using model-specific methods, then resizes to PREPROCESS_TARGET_SHAPE if needed."""
    try:
        if not isinstance(audio_chunk_5s, np.ndarray) or audio_chunk_5s.ndim != 1:
            print(f"Warning: audio_chunk_5s is not a 1D numpy array. Shape: {audio_chunk_5s.shape if hasattr(audio_chunk_5s, 'shape') else 'N/A'}")
            return None
        
        raw_spec_chunk_numpy = None

        if 'mn' in config_obj.model_name:
            audio_tensor = torch.from_numpy(audio_chunk_5s.astype(np.float32))
            with torch.no_grad():
                # efficient_at_spectrogram_generator is initialized globally and configured by global config
                raw_spec_chunk_tensor = efficient_at_spectrogram_generator(audio_tensor.unsqueeze(0))
            raw_spec_chunk_numpy = raw_spec_chunk_tensor.squeeze(0).cpu().numpy()
        
        elif 'efficientnet' in config_obj.model_name:
            # utils.audio2melspec uses parameters from the passed config_obj
            raw_spec_chunk_numpy = utils.audio2melspec(audio_chunk_5s, config_obj)
        else:
            print(f"Warning: Unknown model_name '{config_obj.model_name}' in config. Cannot generate spectrogram.")
            return None

        if raw_spec_chunk_numpy is None or raw_spec_chunk_numpy.ndim != 2:
            print(f"Warning: Spectrogram generation failed or resulted in non-2D array for a chunk.")
            return None

        # --- Resize if necessary to config_obj.PREPROCESS_TARGET_SHAPE ---
        current_shape = (raw_spec_chunk_numpy.shape[0], raw_spec_chunk_numpy.shape[1])
        desired_shape = config_obj.PREPROCESS_TARGET_SHAPE

        if current_shape != desired_shape:
            # print(f"Info: Resizing spectrogram from {current_shape} to {desired_shape} for model {config_obj.model_name}") # Optional for debugging
            # cv2.resize expects dsize as (width, height)
            raw_spec_chunk_numpy = cv2.resize(
                raw_spec_chunk_numpy, 
                (desired_shape[1], desired_shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        return raw_spec_chunk_numpy.astype(np.float32)

    except Exception as e:
        print(f"Error in _generate_spectrogram_from_chunk: {e}")
        print(traceback.format_exc())
        return None

def _process_primary_for_chunking(args):
    """Worker using helper functions to generate spectrograms for one primary audio file."""
    primary_filepath, samplename, config, fabio_intervals, vad_intervals, primary_filename, \
        class_name, scientific_name, birdnet_dets_for_file, manual_annotations_for_file, \
        UNCOVERED_AVES_SCIENTIFIC_NAME, num_files_for_this_species = args

    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples = int(0.5 * config.FS) 

    strategy_info = {
        'used_manual': False, 'used_birdnet': False, 'used_random': False, 
        'used_last_resort_pad': False,
        'fabio_removed_speech': False, 'vad_removed_speech': False
    }

    try:
        audio_for_manual_and_birdnet_base, audio_for_random_base, fabio_applied, vad_applied, error_msg = _load_and_clean_audio(
            primary_filepath, primary_filename, config, fabio_intervals, vad_intervals, min_samples
        )
        if error_msg:
            return samplename, None, error_msg, strategy_info
        
        strategy_info['fabio_removed_speech'] = fabio_applied
        strategy_info['vad_removed_speech'] = vad_applied

        base_duration_for_versions = len(audio_for_manual_and_birdnet_base) 
        
        # --- Determine number of versions to generate --- #
        if base_duration_for_versions < target_samples or cmd_args.mode == "val":
            num_versions_to_generate_final = 1
        elif config.DYNAMIC_CHUNK_COUNTING:
            N = num_files_for_this_species
            N_common_thresh = config.COMMON_SPECIES_FILE_THRESHOLD
            N_min_files_for_max_chunks = 1 # Files at or below this count get MAX_CHUNKS_RARE

            C_max = config.MAX_CHUNKS_RARE
            C_min = config.MIN_CHUNKS_COMMON
            # Target chunks for species just below the common threshold (N_common_thresh - 1 files)
            # This is C_min + 1, ensuring a step down to C_min at N_common_thresh.
            C_interpolate_end = max(C_min + 1, C_min) # Should be at least C_min, typically C_min + 1 (e.g. 2 if C_min is 1)

            if N >= N_common_thresh:
                target_chunks = C_min
            elif N <= N_min_files_for_max_chunks:
                target_chunks = C_max
            else:
                # Linear interpolation for N between (N_min_files_for_max_chunks, N_common_thresh - 1)
                # Chunks decrease as N increases.
                x1 = float(N_min_files_for_max_chunks)
                y1 = float(C_max) # Chunks for x1 files
                x2 = float(N_common_thresh - 1) # Files just before common threshold
                y2 = float(C_interpolate_end) # Chunks for x2 files

                if x2 <= x1: # Should not happen if N_common_thresh > N_min_files_for_max_chunks + 1
                    # Fallback if interpolation range is invalid (e.g., N_common_thresh is too small)
                    target_chunks = C_max if N <= (x1 + x2) / 2 else C_interpolate_end
                else:
                    # Interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    calculated_chunks = y1 + (float(N) - x1) * (y2 - y1) / (x2 - x1)
                    target_chunks = int(round(calculated_chunks))
            
            # Ensure chunks are within the global min/max bounds defined in config
            rarity_based_chunks = max(config.MIN_CHUNKS_COMMON, min(target_chunks, config.MAX_CHUNKS_RARE))
            
            # Cap by actual audio length, allowing for more overlapping chunks for rare species
            strict_length_cap = max(1, int(base_duration_for_versions / target_samples))
            
            # Determine a floor for chunks for shorter files, especially if species is rare, 
            # but don't exceed what rarity itself dictates.
            overlap_allowance_floor = min(rarity_based_chunks, config.MAX_CHUNKS_RARE // 2) # Use integer division
            # Ensure this floor is at least 1, and not less than MIN_CHUNKS_COMMON if rarity_based_chunks was already MIN_CHUNKS_COMMON
            overlap_allowance_floor = max(1, overlap_allowance_floor)
            if rarity_based_chunks == config.MIN_CHUNKS_COMMON : # If it's a common species per rarity calc
                 overlap_allowance_floor = min(overlap_allowance_floor, config.MIN_CHUNKS_COMMON) # Don't boost common species beyond MIN_CHUNKS_COMMON via this floor

            effective_length_cap = max(strict_length_cap, overlap_allowance_floor)
            
            num_versions_to_generate_final = min(rarity_based_chunks, effective_length_cap)
            # Final check to ensure at least 1 chunk is always generated.
            num_versions_to_generate_final = max(1, num_versions_to_generate_final)
        else:
            # Original logic if dynamic chunking is off
            num_versions_to_generate_final = config.PRECOMPUTE_VERSIONS
        # --- End Determine number of versions --- #

        # Pre-evaluate conditions for use_birdnet_strategy to be more robust
        cond1_is_aves = (class_name == 'Aves')
        
        cond2_is_not_uncovered = False
        if isinstance(scientific_name, str) and isinstance(UNCOVERED_AVES_SCIENTIFIC_NAME, str):
            cond2_is_not_uncovered = (scientific_name != UNCOVERED_AVES_SCIENTIFIC_NAME)
        
        cond3_has_birdnet_dets = (birdnet_dets_for_file is not None)
        
        cond4_has_valid_birdnet_items = False
        if cond3_has_birdnet_dets: # Only try to calculate len if birdnet_dets_for_file is not None
            try:
                # Ensure it's treated as a Python iterable for this check
                processed_dets = [d for d in birdnet_dets_for_file if isinstance(d, dict)]
                cond4_has_valid_birdnet_items = (len(processed_dets) > 0)
            except TypeError:
                pass # If not iterable, cond4 remains False, which is fine.

        use_birdnet_strategy = (cond1_is_aves and cond2_is_not_uncovered and cond3_has_birdnet_dets and cond4_has_valid_birdnet_items)

        final_specs_list = []
        # Strategy 1: Manual Annotations (highest priority)
        if manual_annotations_for_file: # Process if annotations exist, regardless of mode initially
            num_manual_to_take = min(len(manual_annotations_for_file), num_versions_to_generate_final)
            # Only take one if in val mode, even if more are available and num_versions_to_generate_final was >1 (which it isn't for val)
            if cmd_args.mode == "val" and num_manual_to_take > 0: 
                num_manual_to_take = 1

            for i in range(num_manual_to_take):
                center_time_s = manual_annotations_for_file[i]
                audio_chunk = _extract_manual_annotation_chunk(
                    audio_for_manual_and_birdnet_base, center_time_s, config, min_samples, target_samples
                )
                if audio_chunk is not None and len(audio_chunk) == target_samples:
                    spec = _generate_spectrogram_from_chunk(audio_chunk, config)
                    if spec is not None: final_specs_list.append(spec); strategy_info['used_manual'] = True
            
            # For val mode, if we got a manual chunk, we are done.
            if cmd_args.mode == "val" and final_specs_list:
                pass 

        # Strategy 2: BirdNET (Aves with detections) - only if no manual annotations processed to fill the quota
        # MODIFICATION: Allow for val mode. num_versions_to_generate_final will be 1 for val.
        if use_birdnet_strategy and not final_specs_list: # If no manual spec was generated (or not enough for train)
            if audio_for_manual_and_birdnet_base is None or len(audio_for_manual_and_birdnet_base) == 0:
                return samplename, None, "Audio for BirdNet chunks is unusable.", strategy_info
            
            # MODIFIED SECTION: Filter, shuffle, and iterate through BirdNET detections
            valid_detections = []
            try:
                # Ensure birdnet_dets_for_file is iterable and filter for valid dicts
                # A detection is valid if it's a dict and has 'start_time' and 'end_time'
                if hasattr(birdnet_dets_for_file, '__iter__'):
                    valid_detections = [
                        d for d in birdnet_dets_for_file 
                        if isinstance(d, dict) and 'start_time' in d and 'end_time' in d
                    ]
                
                if valid_detections: # Only shuffle if there are detections
                    random.shuffle(valid_detections) # Shuffle in-place
            except Exception as e_proc_shuffle:
                print(f"Warning: Error processing or shuffling BirdNET detections for {samplename}: {e_proc_shuffle}.")
                valid_detections = [] # Ensure it's an empty list on error
            # END MODIFIED SECTION

            num_birdnet_chunks_generated = 0
            # For val mode, we only want 1 BirdNET chunk at most if no manual was found.
            # num_versions_to_generate_final is already 1 for val mode.
            # For train mode, it tries to fill up to num_versions_to_generate_final.
            for detection_to_use in valid_detections: # Iterate through shuffled (or empty) valid_detections
                if num_birdnet_chunks_generated >= num_versions_to_generate_final: break 
                audio_chunk_from_birdnet = _extract_birdnet_chunk(
                    audio_for_manual_and_birdnet_base, detection_to_use, config, min_samples, target_samples 
                )
                if audio_chunk_from_birdnet is not None and len(audio_chunk_from_birdnet) == target_samples:
                    spec = _generate_spectrogram_from_chunk(audio_chunk_from_birdnet, config)
                    if spec is not None: 
                        final_specs_list.append(spec)
                        num_birdnet_chunks_generated +=1
                        strategy_info['used_birdnet'] = True
            
            # Fallback to random if not enough BirdNET chunks were generated (ONLY FOR TRAIN MODE)
            # MODIFICATION: Add cmd_args.mode == "train" check here
            if cmd_args.mode == "train":
                num_random_fallbacks_needed = num_versions_to_generate_final - num_birdnet_chunks_generated
                if num_random_fallbacks_needed > 0 and audio_for_random_base is not None and len(audio_for_random_base) > 0:
                    for _ in range(num_random_fallbacks_needed):
                        audio_chunk = _extract_random_chunk(audio_for_random_base, target_samples)
                        if audio_chunk is not None and len(audio_chunk) == target_samples:
                            spec = _generate_spectrogram_from_chunk(audio_chunk, config)
                            if spec is not None: final_specs_list.append(spec)
                            strategy_info['used_random'] = True
        
        # Strategy 3: Random Chunks (Non-Aves or Aves without enough detections, and no manual annotations processed)
        if not final_specs_list: # If no specs from manual or BirdNET yet
            if cmd_args.mode == "val":
                return samplename, None, "VAL_MODE_SKIP: No manual or BirdNET chunk found.", strategy_info
            
            # This part now only runs for TRAIN mode if above conditions met
            actual_chunks_to_make_random = num_versions_to_generate_final 
            # The following if condition related to val mode is now effectively dead code due to the explicit skip above,
            # but harmless to keep for structural clarity if this block were ever reached by val mode.
            if cmd_args.mode == "val" or base_duration_for_versions < target_samples : 
                 actual_chunks_to_make_random = 1 
            
            audio_source_for_random = audio_for_random_base
            if audio_source_for_random is None or len(audio_source_for_random) == 0:
                audio_source_for_random = audio_for_manual_and_birdnet_base 
                if audio_source_for_random is None or len(audio_source_for_random) == 0:
                    return samplename, None, "All audio sources unusable for random chunks (train mode).", strategy_info
            
            for _ in range(actual_chunks_to_make_random):
                audio_chunk = _extract_random_chunk(audio_source_for_random, target_samples)
                if audio_chunk is not None and len(audio_chunk) == target_samples:
                    spec = _generate_spectrogram_from_chunk(audio_chunk, config)
                    if spec is not None: final_specs_list.append(spec)
                    strategy_info['used_random'] = True

        if not final_specs_list:
            # Last resort for train mode if absolutely nothing: Pad/tile original audio once
            if cmd_args.mode == "train" and base_duration_for_versions >= min_samples:
                audio_chunk = _pad_or_tile_audio(audio_for_manual_and_birdnet_base, target_samples)
                if audio_chunk is not None and len(audio_chunk) == target_samples:
                    spec = _generate_spectrogram_from_chunk(audio_chunk, config)
                    if spec is not None: final_specs_list.append(spec)
                    strategy_info['used_last_resort_pad'] = True
        
        if not final_specs_list:
            # This message will now correctly reflect if it's a val mode skip or a train mode complete failure
            return samplename, None, "No valid spectrograms generated after all strategies (or VAL_MODE_SKIP).", strategy_info

        # If final_specs_list is populated (e.g., with 1 spec for val mode), stack and return
        final_specs_array = np.stack([s.astype(np.float32) for s in final_specs_list], axis=0)
        return samplename, final_specs_array, None, strategy_info

    except Exception as e_main:
        tb_str = traceback.format_exc()
        return samplename, None, f"Outer error processing {primary_filepath}: {e_main}\n{tb_str}", strategy_info

def _load_auxiliary_data(config):
    """Loads auxiliary data (VAD, Fabio, BirdNET, Taxonomy)."""
    fabio_intervals = {}
    vad_intervals = {}
    all_birdnet_detections = {}
    all_manual_annotations = {} # New: For manual annotations
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
            with open(config.VOICE_DATA_PKL_PATH, 'rb') as f:
                    vad_data = pickle.load(f)
                    if isinstance(vad_data, dict): vad_intervals = {k: v for k, v in vad_data.items() if isinstance(v, list)}
                    else: print(f"Warning: VAD pickle file does not contain a dictionary.")
            print(f"Loaded VAD intervals for {len(vad_intervals)} files.")
        except FileNotFoundError: print(f"Info: VAD intervals pickle file not found at {config.VOICE_DATA_PKL_PATH}. Skipping.")
        except Exception as e: print(f"Warning: Could not load VAD intervals: {e}")
        
    # Load BirdNET Detections
    print(f"\nAttempting to load BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            all_birdnet_detections = {key: data[key] for key in data.files}
        print(f"Successfully loaded BirdNET detections for {len(all_birdnet_detections)} files.")
    except FileNotFoundError: print(f"Warning: BirdNET detections file not found. Proceeding without BirdNET guidance.")
    except Exception as e: print(f"Warning: Error loading BirdNET detections NPZ: {e}. Proceeding without BirdNET guidance.")

    # Load Manual Annotations
    print(f"\nAttempting to load Manual Annotations from: {config.ANNOTATED_SEGMENTS_CSV_PATH}")
    if config.USE_MANUAL_ANNOTATIONS:
        try:
            manual_ann_df = pd.read_csv(config.ANNOTATED_SEGMENTS_CSV_PATH)
            # Ensure filename is normalized (forward slashes)
            if 'filename' in manual_ann_df.columns: 
                manual_ann_df['filename'] = manual_ann_df['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
            # Filter out low quality and NaN center_time_s, then group
            valid_manual_anns = manual_ann_df[
                (manual_ann_df['is_low_quality'] == False) & 
                (manual_ann_df['center_time_s'].notna())
            ]
            if not valid_manual_anns.empty:
                all_manual_annotations = valid_manual_anns.groupby('filename')['center_time_s'].apply(list).to_dict()
            print(f"Successfully loaded and grouped manual annotations for {len(all_manual_annotations)} files.")
        except FileNotFoundError: print(f"Info: Manual annotations file not found at {config.ANNOTATED_SEGMENTS_CSV_PATH}. Skipping manual annotations as per config or file missing.")
        except Exception as e: print(f"Warning: Error loading manual annotations CSV: {e}. Skipping manual annotations.")
    else:
        print("USE_MANUAL_ANNOTATIONS is False. Skipping manual annotations.")

    # Load Taxonomy for Class/Scientific Name
    print("\nLoading taxonomy data...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        required_taxonomy_cols = {'primary_label', 'class_name', 'scientific_name'}
        if not required_taxonomy_cols.issubset(taxonomy_df.columns):
            print(f"Error: Taxonomy file missing required columns: {required_taxonomy_cols - set(taxonomy_df.columns)}")
            taxonomy_df = pd.DataFrame() # Reset on error
    except Exception as e: print(f"Error loading taxonomy file: {e}")
    
    return fabio_intervals, vad_intervals, all_birdnet_detections, all_manual_annotations, taxonomy_df

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

def _create_processing_tasks(df, config, fabio_intervals, vad_intervals, all_birdnet_detections, all_manual_annotations, species_file_counts):
    """Creates a list of argument tuples for the multiprocessing worker."""
    tasks = []
    skipped_path_count = 0
    UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'

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
            manual_annotations_for_file = all_manual_annotations.get(primary_filename, None)
            # Get the file count for the current species
            current_species_file_count = species_file_counts.get(row['primary_label'], 0)

            tasks.append((
                primary_filepath, samplename, config, fabio_intervals, vad_intervals, 
                primary_filename, class_name, scientific_name, birdnet_dets_for_file, manual_annotations_for_file,
                UNCOVERED_AVES_SCIENTIFIC_NAME, current_species_file_count
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
                results_list.append(result) # tuple (samplename, spec_array, error_msg)
    except Exception as e:
        print(f"\nCRITICAL ERROR during multiprocessing: {e}")
        print(traceback.format_exc())
        raise

    print()
    return results_list

def _aggregate_results(results_list):
    """Processes the list of results from workers into final dictionary and stats."""
    grouped_results = {}
    processed_count = 0
    error_count = 0
    skipped_files = 0 # Files that had an error_msg but no spec, or spec was None
    errors_list = []

    # New counters for strategies
    fabio_removed_count = 0
    vad_removed_count = 0
    manual_chunks_files_count = 0 # Files where at least one manual chunk was used
    birdnet_chunks_files_count = 0 # Files where at least one BirdNET chunk was used
    random_chunks_files_count = 0  # Files where at least one random chunk was used (could be primary or fallback)
    last_resort_pad_files_count = 0 # Files where only the last resort padding was used
    
    # Track which files used which strategy to avoid double counting for file-level stats
    files_using_manual = set()
    files_using_birdnet = set()
    files_using_random = set()
    files_using_last_resort = set()

    for result_tuple in results_list:
        samplename, spec_array, error_msg, strategy_info = result_tuple # Unpack new strategy_info
        
        if strategy_info.get('fabio_removed_speech', False):
            fabio_removed_count += 1
        if strategy_info.get('vad_removed_speech', False):
            vad_removed_count += 1

        if error_msg:
            errors_list.append(f"{samplename}: {error_msg}")
            error_count += 1 # This is error in processing the file itself
        elif spec_array is not None and spec_array.ndim == 3: # Successfully got some spectrograms
            grouped_results[samplename] = spec_array
            processed_count += 1
            # Check which strategies contributed to the successful specs for this file
            if strategy_info.get('used_manual', False):
                files_using_manual.add(samplename)
            if strategy_info.get('used_birdnet', False):
                files_using_birdnet.add(samplename)
            if strategy_info.get('used_random', False):
                 files_using_random.add(samplename)
            if strategy_info.get('used_last_resort_pad', False):
                files_using_last_resort.add(samplename)
        else:
            skipped_files += 1 # File processed by worker but no valid spec array produced
            errors_list.append(f"{samplename}: Skipped by worker (No valid spec array and no error_msg).")

    manual_chunks_files_count = len(files_using_manual)
    birdnet_chunks_files_count = len(files_using_birdnet)
    random_chunks_files_count = len(files_using_random)
    last_resort_pad_files_count = len(files_using_last_resort)

    strategy_counts = {
        'fabio_removed_speech': fabio_removed_count,
        'vad_removed_speech': vad_removed_count,
        'files_w_manual_chunks': manual_chunks_files_count,
        'files_w_birdnet_chunks': birdnet_chunks_files_count,
        'files_w_random_chunks': random_chunks_files_count,
        'files_w_last_resort_pad': last_resort_pad_files_count
    }

    return grouped_results, processed_count, error_count, skipped_files, errors_list, strategy_counts

def _report_summary(total_tasks, processed_count, error_count, skipped_files, errors_list, strategy_counts):
    """Prints a summary of the processing results."""
    print(f"\n--- Processing Summary ---")
    print(f"Attempted to process {total_tasks} primary audio files.")
    print(f"Successfully generated spectrograms for {processed_count} primary files.") 
    print(f"Encountered errors for {error_count} primary files during processing attempt.")
    if skipped_files > 0:
         print(f"{skipped_files} primary files yielded no valid spectrograms (skipped by worker without explicit error).")
    
    print("\n--- Strategy Usage Summary (counts are for files) ---")
    print(f"  Files with Fabio speech interval used for cleaning: {strategy_counts['fabio_removed_speech']}")
    print(f"  Files with VAD speech intervals used for cleaning: {strategy_counts['vad_removed_speech']}")
    print(f"  Files where at least one MANUAL annotation chunk was used: {strategy_counts['files_w_manual_chunks']}")
    print(f"  Files where at least one BIRDNET detection chunk was used: {strategy_counts['files_w_birdnet_chunks']}")
    print(f"  Files where at least one RANDOM chunk was used (incl. fallbacks): {strategy_counts['files_w_random_chunks']}")
    print(f"  Files where only LAST RESORT padding/tiling was used: {strategy_counts['files_w_last_resort_pad']}")

    if errors_list:
        print("\n--- Errors/Skips Encountered (sample) ---")
        for err in errors_list[:20]: print(err)
        if len(errors_list) > 20: print(f"... and {len(errors_list) - 20} more.")
    print("-" * 26)

def _save_spectrogram_npz(grouped_results, determined_output_path, project_root_unused):
    """Saves the spectrogram dictionary to the determined NPZ file path."""
    if not grouped_results:
        print("No spectrograms were successfully generated to save.")
        return

    num_saved_files = len(grouped_results)
    total_chunks = sum(arr.shape[0] for arr in grouped_results.values() if arr is not None and hasattr(arr, 'shape'))

    print(f"Saving {num_saved_files} primary file entries (total {total_chunks} chunks) to: {determined_output_path}")
    start_save = time.time()
    try:
        np.savez_compressed(determined_output_path, **grouped_results)
        end_save = time.time()
        print(f"NPZ saving took {end_save - start_save:.2f} seconds.")
    except Exception as e_save:
        print(f"CRITICAL ERROR saving NPZ file: {e_save}")
        print(traceback.format_exc())
        print("Spectrogram data is likely lost. Check disk space and permissions.")
        sys.exit(1)

def generate_and_save_spectrograms(df, config, species_file_counts):
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
    fabio_intervals, vad_intervals, all_birdnet_detections, all_manual_annotations, taxonomy_df = _load_auxiliary_data(config)

    # 2. Prepare main DataFrame (already includes rare if USE_RARE_DATA was true)
    df_processed = _prepare_dataframe_for_processing(df, taxonomy_df)
    if df_processed is None or df_processed.empty:
        print("DataFrame preparation failed or resulted in an empty DataFrame. Cannot proceed.")
        return None

    # 3. Create tasks for workers
    tasks, skipped_path_count = _create_processing_tasks(
        df_processed, config, fabio_intervals, vad_intervals, all_birdnet_detections, all_manual_annotations, species_file_counts
    )
    if not tasks:
        print("No valid processing tasks created. Cannot proceed.")
        return None

    # 4. Run parallel processing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    results_list = _run_parallel_processing(tasks, _process_primary_for_chunking, num_workers)

    # 5. Aggregate results
    grouped_results, processed_count, error_count, skipped_files, errors_list, strategy_counts = _aggregate_results(results_list)

    # 6. Report summary statistics
    _report_summary(len(tasks), processed_count, error_count, skipped_files, errors_list, strategy_counts)

    # 7. Save NPZ file (pass project_root needed for debug path construction)
    _save_spectrogram_npz(grouped_results, output_npz_path, project_root)

    end_time_gen = time.time()
    print(f"--- Spectrogram generation and saving finished in {end_time_gen - start_time_gen:.2f} seconds ---")

    return grouped_results # Still return the dictionary if needed upstream

def main(config):
    """Main function to run the preprocessing steps."""
    overall_start = time.time()
    print("Starting BirdCLEF Preprocessing Pipeline...")
    
    static_versions_info = f", StaticVersionsFallback={config.PRECOMPUTE_VERSIONS}" if config.DYNAMIC_CHUNK_COUNTING else f", Versions={config.PRECOMPUTE_VERSIONS}"
    print(f"Configuration: Debug={config.debug}, Seed={config.seed}, UseRare={config.USE_RARE_DATA}, UseManualAnns={config.USE_MANUAL_ANNOTATIONS}, RemoveSpeech={config.REMOVE_SPEECH_INTERVALS}{static_versions_info}")
    
    print(f"Dynamic Chunking Enabled: {config.DYNAMIC_CHUNK_COUNTING}")
    if config.DYNAMIC_CHUNK_COUNTING:
        print(f"  Dynamic Chunk Params: MaxRare={config.MAX_CHUNKS_RARE}, MinCommon={config.MIN_CHUNKS_COMMON}, CommonThresh={config.COMMON_SPECIES_FILE_THRESHOLD}")

    df_working, species_file_counts = load_and_prepare_metadata(config)

    if df_working is not None and not df_working.empty:
        generate_and_save_spectrograms(df_working, config, species_file_counts) 
    else:
        print("Metadata loading failed or resulted in empty dataframe. Cannot proceed.")

    overall_end = time.time()
    print(f"\nTotal preprocessing pipeline finished in {(overall_end - overall_start):.2f} seconds.")

if __name__ == '__main__':
    main(config)