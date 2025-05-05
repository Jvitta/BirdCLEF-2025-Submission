import os
import sys
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from tqdm.auto import tqdm
import math

# Ensure project root is in path to import config and utils
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import config
import birdclef_utils as utils # Assuming utils has audio2melspec or similar

# --- Configuration ---
NUM_EXAMPLES_TO_VISUALIZE = 5 # How many random chunks to analyze
MIN_BIRDNET_CONFIDENCE = 0.5 # Minimum confidence for selecting a BirdNET detection
OUTPUT_BASE_DIR = project_root / "outputs" / "eda_spectrograms"

# Define parameter sets to test (including baseline and top solutions from previous year)
PARAM_SETS = [
    {
        "name": "baseline", # Our current configuration
        "n_fft": config.N_FFT, # 1024
        "hop_length": config.HOP_LENGTH, # 128
        "n_mels": config.N_MELS, # 136
        "fmin": config.FMIN, # 20
        "fmax": config.FMAX, # 16000
    },
    {
        "name": "top_sol_1",
        "n_fft": 1024,
        "hop_length": 500,
        "n_mels": 128,
        "fmin": 40,
        "fmax": 15000,
    },
    {
        "name": "top_sol_2",
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000,
    },
    {
        "name": "top_sol_3_torch",
        "n_fft": 4096, # From 2048*2
        "hop_length": 512,
        "n_mels": 512,
        "fmin": 0,
        "fmax": 16000,
    },
    {
        "name": "top_sol_4",
        "n_fft": 1095, # Unusual n_fft
        "hop_length": 500,
        "n_mels": 128,
        "fmin": 40,
        "fmax": 15000,
    },
]

# Create base output directory
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Function for Spectrogram Generation ---
def generate_spectrogram(audio, sr, n_fft, hop_length, n_mels, fmin, fmax):
    """Generates a Mel spectrogram using specified parameters."""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            # power=2.0 # librosa default
        )
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"  Error generating spectrogram: {e}")
        return None

# --- Main Logic ---
def main():
    print("--- EDA: Spectrogram Parameter Visualization ---")

    # 1. Load BirdNET Detections
    print(f"Loading BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            all_birdnet_detections = {key: data[key] for key in data.files}
        print(f"Loaded {len(all_birdnet_detections)} files with BirdNET detections.")
        if not all_birdnet_detections:
            print("No detections found. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: BirdNET detections file not found at {config.BIRDNET_DETECTIONS_NPZ_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading BirdNET detections: {e}. Exiting.")
        return

    # 2. Select Files and Detections
    valid_filenames = list(all_birdnet_detections.keys())
    if len(valid_filenames) < NUM_EXAMPLES_TO_VISUALIZE:
        print(f"Warning: Only {len(valid_filenames)} files have detections. Visualizing all.")
        num_to_select = len(valid_filenames)
    else:
        num_to_select = NUM_EXAMPLES_TO_VISUALIZE

    selected_files = random.sample(valid_filenames, num_to_select)
    print(f"Selected {len(selected_files)} files for visualization: {selected_files}")

    # 3. Process Each Selected File/Chunk
    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples_preprocess = int(0.5 * config.FS) # From preprocessing

    for filename in tqdm(selected_files, desc="Processing Files"):
        print(f"Processing: {filename}")
        detections = all_birdnet_detections[filename]

        # Find a suitable detection (e.g., highest confidence above threshold)
        suitable_detections = [
            d for d in detections
            if isinstance(d, dict) and d.get('confidence', 0) >= MIN_BIRDNET_CONFIDENCE
        ]
        if not suitable_detections:
            print(f"  Skipping {filename}: No detections found with confidence >= {MIN_BIRDNET_CONFIDENCE}")
            continue

        # Sort by confidence and pick the best one for this example
        suitable_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        best_detection = suitable_detections[0]
        det_start = best_detection.get('start_time', 0)
        det_end = best_detection.get('end_time', 0)
        det_conf = best_detection.get('confidence', 0)
        print(f"  Using detection: Start={det_start:.2f}s, End={det_end:.2f}s, Conf={det_conf:.3f}")

        # Construct audio path (assuming it's in the main train_audio dir for simplicity)
        # TODO: Add logic for rare_audio if needed
        audio_path = Path(config.train_audio_dir) / filename
        if not audio_path.exists():
             # Try rare audio path if configured
             if config.USE_RARE_DATA:
                 audio_path = Path(config.train_audio_rare_dir) / filename
                 if not audio_path.exists():
                    print(f"  Skipping {filename}: Audio file not found in {config.train_audio_dir} or {config.train_audio_rare_dir}")
                    continue
             else:
                print(f"  Skipping {filename}: Audio file not found in {config.train_audio_dir}")
                continue

        # Create output subdir for this chunk
        chunk_id = f"{Path(filename).stem}_start{det_start:.1f}_end{det_end:.1f}"
        chunk_output_dir = OUTPUT_BASE_DIR / chunk_id
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {chunk_output_dir}")

        # Load full audio
        try:
            full_audio, sr = librosa.load(audio_path, sr=config.FS, mono=True)
            if full_audio is None or len(full_audio) < min_samples_preprocess:
                 print(f"  Skipping {filename}: Full audio too short or empty after loading.")
                 continue
            full_duration_samples = len(full_audio)

        except Exception as e:
            print(f"  Error loading audio file {audio_path}: {e}")
            continue

        # Extract 5-second chunk centered on detection (mirroring preprocessing logic)
        try:
            center_sec = (det_start + det_end) / 2.0
            target_start_sec = center_sec - (config.TARGET_DURATION / 2.0)
            target_end_sec = center_sec + (config.TARGET_DURATION / 2.0)

            # Convert to samples and clamp
            final_start_idx = max(0, int(target_start_sec * config.FS))
            final_end_idx = min(full_duration_samples, int(target_end_sec * config.FS))

            # Extract chunk
            audio_chunk = full_audio[final_start_idx:final_end_idx]

            # Pad if necessary
            current_len = len(audio_chunk)
            if current_len < target_samples:
                pad_width = target_samples - current_len
                # Simple padding at the end (adjust if different padding desired)
                audio_chunk = np.pad(audio_chunk, (0, pad_width), mode='constant')
            elif current_len > target_samples: # Should ideally not happen with correct logic, but trim just in case
                audio_chunk = audio_chunk[:target_samples]

            if len(audio_chunk) != target_samples:
                 print(f"  Skipping {filename}: Final audio chunk length ({len(audio_chunk)}) != target ({target_samples}) after processing.")
                 continue

        except Exception as e:
            print(f"  Error extracting chunk for {filename}: {e}")
            continue

        # Save audio chunk
        audio_chunk_path = chunk_output_dir / "audio_chunk.wav"
        try:
            sf.write(audio_chunk_path, audio_chunk, config.FS)
            print(f"  Saved audio chunk to: {audio_chunk_path}")
        except Exception as e:
            print(f"  Error saving audio chunk: {e}")
            # Continue to spectrogram generation even if audio saving fails

        # Generate and save spectrograms for different parameters
        print("  Generating and saving spectrograms...")
        for params in tqdm(PARAM_SETS, desc=f"Params for {chunk_id}", leave=False):
            param_name = params['name']
            # print(f"    Testing params: {param_name}") # Optional verbose logging

            mel_spec_db = generate_spectrogram(
                audio_chunk, config.FS,
                params['n_fft'], params['hop_length'], params['n_mels'],
                params['fmin'], params['fmax']
            )

            if mel_spec_db is not None:
                # Plot and save
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(
                    mel_spec_db,
                    sr=config.FS,
                    hop_length=params['hop_length'],
                    x_axis='time',
                    y_axis='mel',
                    fmin=params['fmin'],
                    fmax=params['fmax']
                )
                plt.colorbar(format='%+2.0f dB')
                title = (
                    f"Mel Spectrogram ({param_name})\n"
                    f"n_fft={params['n_fft']}, hop={params['hop_length']}, n_mels={params['n_mels']}\n"
                    f"File: {chunk_id}"
                )
                plt.title(title)
                plt.tight_layout()

                img_path = chunk_output_dir / f"spectrogram_{param_name}.png"
                try:
                    plt.savefig(img_path)
                except Exception as e:
                    print(f"    Error saving spectrogram image {img_path}: {e}")
                plt.close() # Close plot to free memory
            # else:
                # print(f"    Skipped saving for params {param_name} due to generation error.")

    print("--- EDA Script Finished ---")

if __name__ == "__main__":
    main()
