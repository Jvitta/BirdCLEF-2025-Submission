import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import random
from tqdm.auto import tqdm

# Assuming config.py is two levels up (e.g., from eda/ -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- EDA: Visual Inspection of Random Spectrogram Chunks ---")

# --- Configuration ---
spectrogram_path = config.PREPROCESSED_NPZ_PATH
num_samples_to_plot = 10
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "sample_spectrograms")
os.makedirs(plot_dir, exist_ok=True)

# --- Load Spectrogram Data ---
print(f"Loading processed spectrograms from: {spectrogram_path}")

all_chunks_info = [] # List to store (samplename, chunk_idx, chunk_data)

try:
    with np.load(spectrogram_path) as data:
        samplenames = list(data.files)
        if not samplenames:
            print("Loaded NPZ file is empty. Exiting.")
            sys.exit(0)
        print(f"Found {len(samplenames)} samplenames in the NPZ.")

        print("Collecting all individual chunks...")
        for name in tqdm(samplenames, desc="Extracting Chunks"):
            try:
                array = data[name]
                if isinstance(array, np.ndarray) and array.ndim == 3:
                    num_chunks_in_file = array.shape[0]
                    for i in range(num_chunks_in_file):
                        all_chunks_info.append({
                            "samplename": name,
                            "chunk_idx_in_file": i,
                            "spectrogram": array[i, :, :]
                        })
                else:
                    print(f"Warning: Unexpected data format for samplename '{name}'. Expected 3D ndarray, got {type(array)} with ndim {getattr(array, 'ndim', 'N/A')}. Skipping.")
            except Exception as e_inner:
                print(f"Warning: Error accessing data for samplename '{name}': {e_inner}. Skipping.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Spectrogram file not found at {spectrogram_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR loading spectrogram NPZ: {e}. Exiting.")
    sys.exit(1)

if not all_chunks_info:
    print("No valid spectrogram chunks found in the NPZ file. Exiting.")
    sys.exit(0)

print(f"Total individual spectrogram chunks collected: {len(all_chunks_info)}")

# --- Randomly Select Chunks for Plotting ---
if len(all_chunks_info) > num_samples_to_plot:
    selected_chunks_info = random.sample(all_chunks_info, num_samples_to_plot)
    print(f"Randomly selected {num_samples_to_plot} chunks for plotting.")
else:
    selected_chunks_info = all_chunks_info
    print(f"Selected all available {len(all_chunks_info)} chunks for plotting (less than or equal to requested {num_samples_to_plot}).")

# --- Plot and Save Selected Spectrograms ---
if not selected_chunks_info:
    print("No chunks selected for plotting. Exiting.")
    sys.exit(0)

print(f"\nPlotting and saving {len(selected_chunks_info)} spectrogram samples to: {plot_dir}")
plt.style.use('seaborn-v0_8-whitegrid')

for i, chunk_info in enumerate(tqdm(selected_chunks_info, desc="Plotting Samples")):
    samplename = chunk_info["samplename"]
    chunk_idx = chunk_info["chunk_idx_in_file"]
    spectrogram = chunk_info["spectrogram"]

    plt.figure(figsize=(10, 5))
    # Assuming mel spectrograms where frequency bins are typically from bottom to top
    # And time is along the x-axis. If your spectrograms are oriented differently, adjust origin and aspect.
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis') 
    plt.colorbar(label='Amplitude (Float32 or Quantized Value)')
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.title(f"Spectrogram: {samplename} (Chunk {chunk_idx + 1})")
    
    # Sanitize filename slightly in case samplename has slashes
    safe_samplename = samplename.replace("/", "_").replace("\\", "_")
    plot_filename = f"sample_chunk_{safe_samplename}_idx{chunk_idx}.png"
    plot_save_path = os.path.join(plot_dir, plot_filename)

    try:
        plt.tight_layout()
        plt.savefig(plot_save_path)
        if i < 5: # Print path for first few
            print(f"  Saved: {plot_save_path}")
        elif i == 5:
            print("  ... (further save paths suppressed)")
    except Exception as e_save:
        print(f"Error saving plot for {samplename}, chunk {chunk_idx}: {e_save}")
    plt.close() # Close the figure to free memory

print(f"\nFinished plotting sample spectrograms.")
print("Visual inspection EDA script finished.")
