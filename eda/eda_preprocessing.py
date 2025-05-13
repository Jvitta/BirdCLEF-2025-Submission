import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import random
from tqdm.auto import tqdm

# Add project root to sys.path to allow importing config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- EDA: Analysis of Preprocessed Spectrogram NPZ File ---")

# --- Configuration ---
npz_file_path = config.PREPROCESSED_NPZ_PATH
target_shape = tuple(config.PREPROCESS_TARGET_SHAPE) # Expected (H, W)
num_samples_to_plot = 10 # How many spectrograms to visually inspect
num_samples_for_value_stats = 50 # How many chunks to check for value range, shape, dtype
plot_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "preprocessed_spectrograms")

os.makedirs(plot_output_dir, exist_ok=True)

# --- 1. Load Data ---
print(f"\\nLoading NPZ file from: {npz_file_path}")
if not os.path.exists(npz_file_path):
    print(f"ERROR: NPZ file not found at {npz_file_path}. Exiting.")
    sys.exit(1)

try:
    with np.load(npz_file_path) as data_archive:
        all_samplenames = list(data_archive.keys())
        # Load data into a dictionary for easier access
        # This might be memory intensive for very large NPZs, but good for EDA.
        # For extremely large files, consider iterating keys and loading one by one.
        spectrogram_data = {name: data_archive[name] for name in tqdm(all_samplenames, desc="Loading samplenames from NPZ")}
    print(f"Successfully loaded data for {len(all_samplenames)} samplenames.")
except Exception as e:
    print(f"ERROR: Could not load NPZ file: {e}. Exiting.")
    sys.exit(1)

if not spectrogram_data:
    print("No data found in the NPZ file. Exiting.")
    sys.exit(1)

# --- 2. Basic Statistics ---
print(f"\\n--- Basic Statistics ---")
print(f"Total unique samplenames in NPZ: {len(spectrogram_data)}")

# --- 3. Chunk Analysis ---
print(f"\\n--- Chunk Analysis ---")
chunk_counts = []
for samplename, chunks_array in spectrogram_data.items():
    if isinstance(chunks_array, np.ndarray) and chunks_array.ndim == 3: # Expected (N_chunks, H, W)
        chunk_counts.append(chunks_array.shape[0])
    else:
        print(f"Warning: Data for samplename '{samplename}' is not a 3D ndarray as expected. Shape: {getattr(chunks_array, 'shape', 'N/A')}")
        chunk_counts.append(0) # Or handle as an error/skip

if chunk_counts:
    chunk_counts_series = pd.Series(chunk_counts)
    print("Distribution of precomputed chunks per samplename:")
    print(f"  Min chunks:    {chunk_counts_series.min()}")
    print(f"  Max chunks:    {chunk_counts_series.max()}")
    print(f"  Mean chunks:   {chunk_counts_series.mean():.2f}")
    print(f"  Median chunks: {chunk_counts_series.median()}")
    print(f"  Expected per config (PRECOMPUTE_VERSIONS): {config.PRECOMPUTE_VERSIONS}")

    # Plot histogram of chunk counts
    plt.figure(figsize=(10, 6))
    sns.histplot(chunk_counts_series, discrete=True) # discrete=True for integer counts
    plt.title("Histogram of Number of Chunks per Samplename")
    plt.xlabel("Number of Chunks")
    plt.ylabel("Frequency")
    plt.grid(True)
    plot_path = os.path.join(plot_output_dir, "chunk_counts_histogram.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved chunk counts histogram to: {plot_path}")
    except Exception as e_save:
        print(f"Error saving chunk counts histogram: {e_save}")
    plt.close()
else:
    print("No valid chunk counts found to analyze.")

# --- 4. Spectrogram Properties (from a sample) ---
print(f"\\n--- Spectrogram Properties (Sampled) ---")
print(f"Target shape from config: {target_shape}")
shapes_ok = 0
shapes_mismatch = 0
dtypes_ok = 0
dtypes_mismatch = 0
all_pixel_values_sample = []

# Collect all individual chunks for sampling
all_individual_chunks_with_names = []
for samplename, chunks_array in spectrogram_data.items():
    if isinstance(chunks_array, np.ndarray) and chunks_array.ndim == 3:
        for i in range(chunks_array.shape[0]):
            all_individual_chunks_with_names.append((samplename, i, chunks_array[i]))

if not all_individual_chunks_with_names:
    print("No individual chunks found to sample for properties.")
else:
    num_to_sample_props = min(num_samples_for_value_stats, len(all_individual_chunks_with_names))
    sampled_chunks_for_props = random.sample(all_individual_chunks_with_names, num_to_sample_props)

    print(f"Analyzing {num_to_sample_props} random chunks for shape, dtype, and value range...")
    for samplename, chunk_idx, chunk_data in tqdm(sampled_chunks_for_props, desc="Analyzing chunk properties"):
        # Shape check
        if chunk_data.shape == target_shape:
            shapes_ok += 1
        else:
            shapes_mismatch += 1
            print(f"  Mismatch: Samplename '{samplename}', chunk {chunk_idx} has shape {chunk_data.shape}")
        
        # Dtype check
        if chunk_data.dtype == np.float32:
            dtypes_ok += 1
        else:
            dtypes_mismatch += 1
            print(f"  Mismatch: Samplename '{samplename}', chunk {chunk_idx} has dtype {chunk_data.dtype}")
        
        all_pixel_values_sample.extend(chunk_data.flatten())

    print(f"\\nShape Check Summary (Target: {target_shape}):")
    print(f"  Chunks with correct shape: {shapes_ok}")
    print(f"  Chunks with incorrect shape: {shapes_mismatch}")

    print(f"\\nDtype Check Summary (Target: np.float32):")
    print(f"  Chunks with correct dtype: {dtypes_ok}")
    print(f"  Chunks with incorrect dtype: {dtypes_mismatch}")

    if all_pixel_values_sample:
        values_series = pd.Series(all_pixel_values_sample)
        print(f"\\nValue Range Summary (from {num_to_sample_props} sampled chunks):")
        print(f"  Min pixel value:  {values_series.min():.4f}")
        print(f"  Max pixel value:  {values_series.max():.4f}")
        print(f"  Mean pixel value: {values_series.mean():.4f}")
        print(f"  Median pixel value: {values_series.median():.4f}")
    else:
        print("\\nNo pixel values collected from samples.")

# --- 5. Visual Inspection (Plot Random Samples) ---
print(f"\\n--- Visual Inspection: Plotting Sample Spectrograms ---")
if not all_samplenames:
    print("No samplenames available to plot.")
else:
    num_to_plot = min(num_samples_to_plot, len(all_samplenames))
    samplenames_to_plot = random.sample(all_samplenames, num_to_plot)
    
    print(f"Plotting one random chunk from {num_to_plot} random samplenames...")
    for samplename in tqdm(samplenames_to_plot, desc="Plotting samples"):
        chunks_array = spectrogram_data.get(samplename)
        if chunks_array is None or not isinstance(chunks_array, np.ndarray) or chunks_array.ndim != 3 or chunks_array.shape[0] == 0:
            print(f"Skipping plotting for '{samplename}': No valid chunks found.")
            continue
            
        # Select one random chunk from this samplename
        random_chunk_idx = random.randint(0, chunks_array.shape[0] - 1)
        spec_to_plot = chunks_array[random_chunk_idx]

        if spec_to_plot.shape != target_shape:
            print(f"Warning: Samplename '{samplename}', chunk {random_chunk_idx} has unexpected shape {spec_to_plot.shape} for plotting. Expected {target_shape}. Skipping plot.")
            continue
            
        plt.figure(figsize=(8, 5))
        plt.imshow(spec_to_plot, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Magnitude (Log Mel?)')
        plt.title(f"Spectrogram: {samplename} (Chunk {random_chunk_idx+1}/{chunks_array.shape[0]})\nShape: {spec_to_plot.shape}, Dtype: {spec_to_plot.dtype}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Bins")
        
        fig_filename = f"spec_{samplename}_chunk{random_chunk_idx}.png"
        plot_path = os.path.join(plot_output_dir, fig_filename)
        try:
            plt.tight_layout()
            plt.savefig(plot_path)
        except Exception as e_save_fig:
            print(f"Error saving figure {fig_filename}: {e_save_fig}")
        plt.close()
        
    print(f"Saved sample spectrogram plots to: {plot_output_dir}")

print("\\n--- EDA Finished ---")
