import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython.display as ipd
import multiprocessing
import traceback
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore")

# --- Worker Function (moved from kaggle_inference.py) ---
def _preprocess_audio_file_worker(audio_path_str, config):
    """Loads one audio file, generates all 5s specs and row IDs."""
    audio_path = Path(audio_path_str)
    soundscape_id = audio_path.stem
    segment_specs = []
    segment_row_ids = []

    try:
        audio_data, sr = librosa.load(audio_path, sr=config.FS, mono=True)
        segment_length_samples = int(config.TARGET_DURATION * config.FS)
        total_duration_samples = len(audio_data)
        num_segments = total_duration_samples // segment_length_samples

        if num_segments == 0:
            return [], [] # Skip short files

        for segment_idx in range(num_segments):
            start_sample = segment_idx * segment_length_samples
            end_sample = start_sample + segment_length_samples
            segment_audio = audio_data[start_sample:end_sample]

            # --- Replicate processing logic (could be moved to utils) ---
            target_len = segment_length_samples
            if len(segment_audio) < target_len:
                segment_audio = np.pad(segment_audio, (0, target_len - len(segment_audio)), mode='constant')
            elif len(segment_audio) > target_len:
                segment_audio = segment_audio[:target_len]

            # --- Re-use audio2melspec if it exists and handles this --- 
            # For now, keep the direct implementation as before
            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio,
                sr=config.FS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
            if max_val > min_val: mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
            else: mel_spec_norm = np.zeros_like(mel_spec_db)

            if mel_spec_norm.shape != config.TARGET_SHAPE:
                mel_spec_resized = cv2.resize(mel_spec_norm, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            else:
                mel_spec_resized = mel_spec_norm
            # --- End Replicated Processing --- 

            segment_specs.append(mel_spec_resized.astype(np.float32))
            
            end_time_sec = (segment_idx + 1) * config.TARGET_DURATION
            row_id = f"{soundscape_id}_{int(end_time_sec)}"
            segment_row_ids.append(row_id)

        return segment_row_ids, segment_specs

    except Exception as e:
        # Print error specific to this file and return empty
        print(f"\nError preprocessing worker {audio_path.name}: {e}\n{traceback.format_exc()}")
        return [], []

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def _plot_debug_waveform(audio, sr, title):
    """Helper function to plot waveform and provide audio playback for debugging."""
    if audio is None or len(audio) == 0:
        print(f"Debug: Skipping plot/audio for '{title}' - audio is empty.")
        return
    try:
        plt.figure(figsize=(15, 3))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        print(f"Audio for: {title}")
        ipd.display(ipd.Audio(data=audio, rate=sr))
        
    except Exception as e:
        print(f"Debug Plot/Audio: Error processing waveform for '{title}': {e}")

def process_audio_file(filepath, filename, cfg, fabio_intervals, vad_intervals):
    """Process a single audio file to get the mel spectrogram,
       using precomputed intervals to remove speech or isolate sounds.
       Includes debug plotting of waveforms before/after processing if cfg.debug_preprocessing_mode is True.
    """
    try:
        audio_data, sr = librosa.load(filepath, sr=cfg.FS)
        if sr != cfg.FS:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=cfg.FS)

        if cfg.debug_preprocessing_mode:
            _plot_debug_waveform(audio_data, cfg.FS, f"Original Audio: {filename}")
            
        relevant_audio = None
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        
        if cfg.REMOVE_SPEECH_INTERVALS:
            if filename in fabio_intervals:
                start_time, stop_time = fabio_intervals[filename]
                start_idx = int(start_time * cfg.FS)
                end_idx = int(stop_time * cfg.FS)
                start_idx = max(0, start_idx)
                end_idx = min(len(audio_data), end_idx)
                if start_idx < end_idx:
                    relevant_audio = audio_data[start_idx:end_idx]
                else:
                    print(f"Warning: Invalid Fabio interval for {filename} ({start_time}-{stop_time}s). Using full audio.")
                    relevant_audio = audio_data

            elif filepath in vad_intervals: 
                speech_timestamps = vad_intervals[filepath]
                
                if not speech_timestamps:
                    print(f"Debug: VAD interval list is empty for {filepath}. Using full audio.")
                    relevant_audio = audio_data
                else:
                    non_speech_segments = []
                    current_pos_sec = 0.0
                    audio_duration_sec = len(audio_data) / cfg.FS
                    
                    try:
                        speech_timestamps.sort(key=lambda x: x['start']) 
                    except KeyError:
                        print(f"Warning: VAD timestamps for {filepath} lack 'start' key. Cannot sort/process. Using full audio.")
                        relevant_audio = audio_data
                        speech_timestamps = []

                    for segment in speech_timestamps:
                        if not isinstance(segment, dict) or 'start' not in segment or 'end' not in segment:
                            print(f"Warning: Invalid VAD segment format for {filepath}: {segment}. Skipping segment.")
                            continue 

                        start_speech_sec = segment['start']
                        end_speech_sec = segment['end']
                        
                        if start_speech_sec > current_pos_sec:
                            start_idx = int(current_pos_sec * cfg.FS)
                            end_idx = int(start_speech_sec * cfg.FS)
                            start_idx = max(0, start_idx)
                            end_idx = min(len(audio_data), end_idx)
                            if start_idx < end_idx:
                                non_speech_segments.append(audio_data[start_idx:end_idx])
                        
                        current_pos_sec = max(current_pos_sec, end_speech_sec)
                    
                    if current_pos_sec < audio_duration_sec:
                        start_idx = int(current_pos_sec * cfg.FS)
                        start_idx = max(0, start_idx)
                        if start_idx < len(audio_data):
                            non_speech_segments.append(audio_data[start_idx:])
                        
                    if non_speech_segments:
                        non_speech_segments = [np.asarray(seg) for seg in non_speech_segments if seg is not None and len(seg) > 0]
                        if non_speech_segments:
                            relevant_audio = np.concatenate(non_speech_segments)
                        else:
                            print(f"Warning: VAD processing resulted in only empty segments for {filename}. Using full audio.")
                            relevant_audio = audio_data
                    else:
                        print(f"Warning: VAD removed all segments for {filename}. Using full audio as fallback.")
                        relevant_audio = audio_data

        if relevant_audio is None:
            relevant_audio = audio_data
            
        # Ensure relevant_audio is not empty before proceeding
        if relevant_audio is None or len(relevant_audio) == 0:
            print(f"Warning: No valid audio segment found for {filename} after processing intervals. Using full audio as fallback.")
            relevant_audio = audio_data # Fallback to original
            if relevant_audio is None or len(relevant_audio) == 0:
                 print(f"Error: Original audio for {filename} is also empty. Skipping.")
                 if cfg.debug_preprocessing_mode:
                      _plot_debug_waveform(np.array([]), cfg.FS, f"Processed Audio (Empty!): {filename}") 
                 return None # Return None if audio is truly empty

        # --- Debug Plot 2: Processed Audio (Before 5s chunking) ---
        if cfg.debug_preprocessing_mode:
            _plot_debug_waveform(relevant_audio, cfg.FS, f"Relevant Audio (Before Target Crop): {filename}") 
        # --- End Debug Plot ---

        # --- Define Minimum Duration --- #
        min_duration_sec = 0.5
        min_samples = int(min_duration_sec * cfg.FS)
        # --- Check Minimum Duration --- #
        if len(relevant_audio) < min_samples:
            print(f"Warning: Relevant audio for {filename} ({len(relevant_audio)/cfg.FS:.2f}s) is shorter than minimum ({min_duration_sec}s). Skipping file.")
            return None
        
        # 4. Extract target duration (e.g., center 5s) from the relevant segment
        selected_audio_chunk = None
        if len(relevant_audio) < target_samples:
            n_copy = math.ceil(target_samples / len(relevant_audio))
            repeated_chunk = np.concatenate([relevant_audio] * n_copy)
            selected_audio_chunk = repeated_chunk[:target_samples]
            # Final check: If somehow still short (e.g., original was zero length, though handled above), pad
            if len(selected_audio_chunk) < target_samples:
                 selected_audio_chunk = np.pad(selected_audio_chunk,
                                              (0, target_samples - len(selected_audio_chunk)),
                                 mode='constant')
        else:
            # Extract center target_samples if relevant_audio is long enough
            start_idx = max(0, int(len(relevant_audio) / 2 - target_samples / 2))
            end_idx = start_idx + target_samples # Ensure exact length
            selected_audio_chunk = relevant_audio[start_idx:end_idx]
            
            # Ensure exact length due to potential edge cases or slicing issues
            if len(selected_audio_chunk) < target_samples:
                 selected_audio_chunk = np.pad(selected_audio_chunk,
                                              (0, target_samples - len(selected_audio_chunk)),
                                              mode='constant')
            elif len(selected_audio_chunk) > target_samples:
                 selected_audio_chunk = selected_audio_chunk[:target_samples]
        # --- End 5s chunk extraction ---

        # --- Debug Plot 3: Final 5s Chunk ---
        if cfg.debug_preprocessing_mode:
             _plot_debug_waveform(selected_audio_chunk, cfg.FS, f"Final 5s Chunk: {filename}")
        # --- End Debug Plot ---

        # 5. Generate Mel Spectrogram from the 5s chunk
        mel_spec = audio2melspec(selected_audio_chunk, cfg)
        
        # 6. Resize Spectrogram (Optional but often needed for fixed input size)
        if mel_spec.shape[1] != cfg.TARGET_SHAPE[1] or mel_spec.shape[0] != cfg.TARGET_SHAPE[0]:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# --- Worker Function for Multiprocessing (Modified) --- #
def _process_row_worker(args):
    """Worker function to process a single audio file (takes a single arg tuple).
       Returns (samplename, spectrogram_data) on success, (samplename, None) on failure/skip.
    """
    # Unpack arguments
    filepath, filename, samplename, cfg, fabio_intervals, vad_intervals = args

    try:
        mel_spec = process_audio_file(filepath, filename, cfg, fabio_intervals, vad_intervals)

        if mel_spec is not None:
            # Instead of saving, return the data
            return (samplename, mel_spec)
        else:
            # File was likely skipped due to duration or processing error in process_audio_file
            return (samplename, None)

    except Exception as e:
        # Catch errors occurring during the worker execution itself
        tb_str = traceback.format_exc()
        print(f"Error in worker for {filepath}: {e}\n{tb_str}")
        return (samplename, None) # Return None on error
# --- End Worker Function --- #

def generate_spectrograms(df, cfg, fabio_intervals, vad_intervals):
    """Generate spectrograms using multiprocessing and save to a single NPZ file."""
    print("Generating mel spectrograms using multiprocessing...")
    start_time = time.time()

    num_workers = cfg.num_workers
    print(f"Using {num_workers} worker processes.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(cfg.PREPROCESSED_NPZ_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Will save spectrograms archive to: {cfg.PREPROCESSED_NPZ_PATH}")

    # Prepare list of argument tuples for each task
    tasks = []
    for _, row in df.iterrows():
        tasks.append((row['filepath'], row['filename'], row['samplename'], cfg, fabio_intervals, vad_intervals))

    print(f"Prepared {len(tasks)} tasks for processing.")

    all_spectrograms = {}
    successful_count = 0
    skipped_or_failed_count = 0
    errors_info = [] # Store samplenames of skipped/failed files

    # --- Use multiprocessing Pool with imap_unordered --- #
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(_process_row_worker, tasks)

        print("Starting task processing...")
        for samplename, mel_spec in tqdm(results_iterator, total=len(tasks), desc="Preprocessing Audio"):
            if mel_spec is not None:
                all_spectrograms[samplename] = mel_spec
                successful_count += 1
            else:
                skipped_or_failed_count += 1
                # Log which samplename failed/was skipped
                errors_info.append(samplename)
    # --- End Pool --- #

    print(f"\nFinished collecting {successful_count} spectrograms from workers.")

    # Save all collected spectrograms to a single compressed NPZ file
    if all_spectrograms:
        print(f"Saving {len(all_spectrograms)} spectrograms to {cfg.PREPROCESSED_NPZ_PATH}...")
        try:
            np.savez_compressed(cfg.PREPROCESSED_NPZ_PATH, **all_spectrograms)
            print("Successfully saved NPZ archive.")
        except Exception as e:
            print(f"CRITICAL Error saving NPZ archive: {e}")
            # Depending on the workflow, you might want to raise the error
            # raise e
    else:
        print("Warning: No spectrograms were successfully generated to save.")

    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Attempted to process {len(tasks)} files.")
    print(f"Successfully generated {successful_count} spectrograms.")
    print(f"Skipped or failed: {skipped_or_failed_count} files.")

    # Optionally print skipped/failed sample names
    if errors_info:
        print(f"\n--- Skipped/Failed Sample Names (First {min(10, len(errors_info))}) --- ")
        for name in errors_info[:10]:
            print(f"  - {name}")
        if len(errors_info) > 10:
             print("  ... (additional names hidden)")
        print("----------------------------------")

    # No return value needed as the file is saved directly