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
import cv2 # <-- Import OpenCV

# Ensure project root is in path to import config and utils
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import config
import utils as utils # Assuming utils has audio2melspec or similar

# --- Configuration ---
NUM_EXAMPLES_TO_VISUALIZE = 5 # How many random chunks to analyze
MIN_BIRDNET_CONFIDENCE = 0.5 # Minimum confidence for selecting a BirdNET detection
OUTPUT_BASE_DIR = project_root / "outputs" / "eda_spectrograms_resized" # New output dir
FINAL_RESIZE_SHAPE = (256, 256) # The final shape fed to the model (height, width)

# --- Define Parameter Combinations --- #
new_param_sets = []

# Parameter lists to combine
n_fft_list = [1024, 2048]
n_mels_list = [136, 256]
hop_lengths = [64, 128, 256]
fmin_list = [20, 50, 100]

# Add the current baseline config explicitly (which will also be resized)
new_param_sets.append({
    "name": f"BASELINE_fft{config.N_FFT}_hop{config.HOP_LENGTH}_mel{config.N_MELS}_fmin{config.FMIN}",
    "n_fft": config.N_FFT,
    "hop_length": config.HOP_LENGTH,
    "n_mels": config.N_MELS,
    "fmin": config.FMIN,
    "fmax": config.FMAX, # Keep FMAX from config
})


# Generate new combinations
for n_fft in n_fft_list:
    for hop in hop_lengths:
        for n_mels in n_mels_list:
            for fmin in fmin_list:
                # Basic validation
                if n_fft < hop: continue

                param_name = f"fft{n_fft}_hop{hop}_mel{n_mels}_fmin{fmin}"
                # Avoid adding duplicate if it matches baseline params
                is_baseline = (n_fft == config.N_FFT and
                               hop == config.HOP_LENGTH and
                               n_mels == config.N_MELS and
                               fmin == config.FMIN)
                if not is_baseline:
                    new_param_sets.append({
                        "name": param_name,
                        "n_fft": n_fft,
                        "hop_length": hop,
                        "n_mels": n_mels,
                        "fmin": fmin,
                        "fmax": config.FMAX, # Keep FMAX from config
                    })

PARAM_SETS = new_param_sets
print(f"Generated {len(PARAM_SETS)} parameter sets for visualization.")

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
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"  Error generating spectrogram: {e}")
        return None

# --- Main Logic ---
def main():
    print("--- EDA: Resized Spectrogram Parameter Visualization ---")

    # 1. Load BirdNET Detections (Optional, adjust if needed)
    print(f"Loading BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    try:
        # Load or define logic to get audio files for visualization
        # This example still uses BirdNET to find interesting segments
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            all_birdnet_detections = {key: data[key] for key in data.files}
        print(f"Loaded {len(all_birdnet_detections)} files with BirdNET detections.")
        if not all_birdnet_detections:
            print("No detections found. Exiting.")
            return
        valid_filenames = list(all_birdnet_detections.keys())
    except Exception as e:
        print(f"Could not load BirdNET detections ({e}). Falling back to random files if metadata exists.")
        try:
            df_train = pd.read_csv(config.train_csv_path)
            valid_filenames = df_train['filename'].unique().tolist()
        except Exception as e_meta:
             print(f"Could not load metadata either ({e_meta}). Exiting.")
             return

    # 2. Select Files
    if len(valid_filenames) < NUM_EXAMPLES_TO_VISUALIZE:
        print(f"Warning: Only {len(valid_filenames)} files available. Visualizing all.")
        num_to_select = len(valid_filenames)
    else:
        num_to_select = NUM_EXAMPLES_TO_VISUALIZE

    selected_files = random.sample(valid_filenames, num_to_select)
    print(f"Selected {len(selected_files)} files for visualization: {selected_files}")

    # 3. Process Each Selected File/Chunk
    target_samples = int(config.TARGET_DURATION * config.FS)
    min_samples_preprocess = int(0.5 * config.FS)

    for filename in tqdm(selected_files, desc="Processing Files"):
        print(f"Processing: {filename}")

        # Construct audio path
        audio_path = Path(config.train_audio_dir) / filename
        if not audio_path.exists():
            if config.USE_RARE_DATA:
                audio_path = Path(config.train_audio_rare_dir) / filename
                if not audio_path.exists():
                    print(f"  Skipping {filename}: Audio file not found.")
                    continue
            else:
                print(f"  Skipping {filename}: Audio file not found.")
                continue

        # Create output subdir for this file
        file_stem = Path(filename).stem
        file_output_dir = OUTPUT_BASE_DIR / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {file_output_dir}")

        # Load full audio
        try:
            full_audio, sr = librosa.load(audio_path, sr=config.FS, mono=True)
            if full_audio is None or len(full_audio) < min_samples_preprocess:
                 print(f"  Skipping {filename}: Audio too short or empty.")
                 continue
            full_duration_samples = len(full_audio)
        except Exception as e:
            print(f"  Error loading audio file {audio_path}: {e}")
            continue

        # --- Extract a SINGLE random 5-second chunk for consistent visualization ---
        try:
            if full_duration_samples <= target_samples:
                 # Pad short audio
                 pad_width = target_samples - full_duration_samples
                 audio_chunk = np.pad(full_audio, (0, pad_width), mode='constant')
            else:
                 # Take random crop
                 max_start = full_duration_samples - target_samples
                 start_idx = random.randint(0, max_start)
                 audio_chunk = full_audio[start_idx : start_idx + target_samples]

            if len(audio_chunk) != target_samples:
                 print(f"  Skipping {filename}: Problem creating 5s audio chunk.")
                 continue
        except Exception as e:
            print(f"  Error extracting chunk for {filename}: {e}")
            continue

        # Save audio chunk
        audio_chunk_path = file_output_dir / "audio_chunk.wav"
        try:
            sf.write(audio_chunk_path, audio_chunk, config.FS)
            # print(f"  Saved audio chunk to: {audio_chunk_path}") # Less verbose
        except Exception as e:
            print(f"  Warning: Error saving audio chunk: {e}")

        # Generate and save spectrograms for different parameters
        print("  Generating and resizing spectrograms...")
        for params in tqdm(PARAM_SETS, desc=f"Params for {file_stem}", leave=False):
            param_name = params['name']

            # --- Generate original spectrogram ---
            mel_spec_db = generate_spectrogram(
                audio_chunk, config.FS,
                params['n_fft'], params['hop_length'], params['n_mels'],
                params['fmin'], params['fmax']
            )

            if mel_spec_db is not None:
                orig_h, orig_w = mel_spec_db.shape

                # --- ALWAYS resize to FINAL_RESIZE_SHAPE ---
                target_h, target_w = FINAL_RESIZE_SHAPE
                target_shape_cv2 = (target_w, target_h) # OpenCV uses (width, height)
                spec_to_plot = cv2.resize(mel_spec_db, target_shape_cv2, interpolation=cv2.INTER_LINEAR)
                final_shape_h, final_shape_w = target_h, target_w # Should always be 256x256

                # --- Plotting ---
                # Set DPI and calculate figsize for exact 256x256 pixel output
                dpi = 100
                figsize = (final_shape_w / dpi, final_shape_h / dpi)
                plt.figure(figsize=figsize, dpi=dpi)

                # Use imshow with aspect='equal'
                img = plt.imshow(spec_to_plot, aspect='equal', origin='lower', cmap='magma', interpolation='nearest')

                # Remove axes, labels, colorbar, title for pure image view
                plt.axis('off')
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                # Construct filename including original params
                img_path = file_output_dir / f"resized_256x256_orig_{param_name}.png"
                try:
                    # Save without padding/whitespace
                    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
                except Exception as e:
                    print(f"    Error saving spectrogram image {img_path}: {e}")
                plt.close() # Close plot to free memory
            else:
                 print(f"    Skipped {param_name} due to generation error.")

    print("--- EDA Script Finished ---")

if __name__ == "__main__":
    main()
