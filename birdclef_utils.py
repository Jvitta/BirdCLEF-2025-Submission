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

import torch
import warnings
warnings.filterwarnings("ignore")

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

        else:
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

def generate_spectrograms(df, cfg, fabio_intervals, vad_intervals):
    """Generate spectrograms from audio files using precomputed intervals and save individually."""
    print("Generating mel spectrograms using precomputed intervals and saving individually...")
    start_time = time.time()

    # Ensure the output directory exists
    os.makedirs(cfg.PREPROCESSED_DATA_DIR, exist_ok=True)
    print(f"Saving individual spectrograms to: {cfg.PREPROCESSED_DATA_DIR}")

    # Remove dictionary, track counts instead
    # all_bird_data = {} 
    successful_count = 0
    errors = []
    skipped_count = 0 # Count files skipped due to short duration

    # Restore N_MAX_PREPROCESS check if needed and uncommented in config
    n_max_files = getattr(cfg, 'N_MAX_PREPROCESS', None)
    if cfg.debug_preprocessing_mode and n_max_files is not None:
         print(f"DEBUG: Will process a maximum of {n_max_files} files due to N_MAX_PREPROCESS.")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        if cfg.debug_preprocessing_mode and n_max_files is not None and i >= n_max_files:
             print(f"DEBUG: Stopping preprocessing early after {n_max_files} files.")
             break
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            filename = row['filename']
            output_filepath = os.path.join(cfg.PREPROCESSED_DATA_DIR, f"{samplename}.npy")
            
            # Skip if file already exists (optional, useful for resuming)
            # if os.path.exists(output_filepath):
            #     successful_count += 1 # Or a separate 'already_exist_count'
            #     continue

            mel_spec = process_audio_file(filepath, filename, cfg, fabio_intervals, vad_intervals)

            if mel_spec is not None:
                # Save individual file instead of adding to dict
                np.save(output_filepath, mel_spec)
                successful_count += 1
            else:
                 # process_audio_file returns None if error or skipped (e.g., too short)
                 # Check the log message from process_audio_file to differentiate
                 skipped_count += 1 # Assume None means skipped or error
                 errors.append((filepath, "Processing returned None (skipped or error)"))

        except Exception as e:
            print(f"Error processing row {i} ({row.get('filepath', 'N/A')}): {e}")
            errors.append((row.get('filepath', 'N/A'), str(e)))
            skipped_count += 1 # Count exceptions as skipped/failed

    end_time = time.time()
    total_processed = successful_count + skipped_count
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Attempted to process {df.shape[0]} files.") # Use df.shape[0] for total requested
    print(f"Successfully saved {successful_count} spectrogram files.")
    print(f"Skipped or failed: {skipped_count} files.") # Combined skipped/error count
    # Optionally print more error details if needed
    # if errors:
    #    print("\nFiles with processing errors or skips:")
    #    for err_file, reason in errors[:10]: # Print first 10 errors
    #        print(f"- {err_file}: {reason}")
    
    # No longer returns the dictionary
    # return all_bird_data