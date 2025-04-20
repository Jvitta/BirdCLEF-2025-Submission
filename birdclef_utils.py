import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

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

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Extract center 5 seconds
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')

        mel_spec = audio2melspec(center_audio, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files and return as a dictionary."""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    # Re-initialize the dictionary to store results
    all_bird_data = {} 
    errors = []

    # Revert tqdm usage if necessary (it might work either way, but revert for consistency)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Audio"):
        # Use original debug limit logic if N_MAX_PREPROCESS is defined and debug is on
        if cfg.debug and cfg.N_MAX_PREPROCESS is not None and i >= cfg.N_MAX_PREPROCESS:
             print(f"DEBUG: Stopping preprocessing early after {cfg.N_MAX_PREPROCESS} files.")
             break
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            # Remove output_filepath logic
            # output_filepath = os.path.join(cfg.PREPROCESSED_DATA_DIR, f"{samplename}.npy")

            mel_spec = process_audio_file(filepath, cfg)

            if mel_spec is not None:
                # Store result in dictionary instead of saving
                all_bird_data[samplename] = mel_spec
            else:
                 # process_audio_file should print its own errors
                 errors.append((filepath, "Processing returned None"))

        except Exception as e:
            # Use original error reporting
            print(f"Error processing {row.get('filepath', 'N/A')}: {e}")
            errors.append((row.get('filepath', 'N/A'), str(e)))

    end_time = time.time()
    # Use original success/fail counts based on dictionary and errors
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")
    
    # Return the dictionary
    return all_bird_data