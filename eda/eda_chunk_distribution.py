import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm.auto import tqdm

# Assuming config.py is two levels up (e.g., from eda/ -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- EDA: Distribution of Spectrogram Chunks per File ---")

# --- Load Spectrogram Data ---
spectrogram_path = config.PREPROCESSED_NPZ_PATH
print(f"Loading processed spectrograms from: {spectrogram_path}")

try:
    # Load the npz file
    with np.load(spectrogram_path) as data:
        # Get keys (samplenames)
        samplenames = list(data.files)
        print(f"Loaded data for {len(samplenames)} samplenames.")
        if not samplenames:
            print("Loaded file is empty. Exiting.")
            sys.exit(0)
        
        # --- Analyze Chunk Counts --- 
        chunk_counts = []
        print("Analyzing chunk counts...")
        for name in tqdm(samplenames, desc="Checking Chunks"):
            try:
                array = data[name]
                # Expecting shape (N, H, W)
                if isinstance(array, np.ndarray) and array.ndim == 3:
                    chunk_counts.append(array.shape[0]) # N is the number of chunks
                else:
                    print(f"Warning: Unexpected data format for {name}. Expected 3D ndarray, got {type(array)} with ndim {getattr(array, 'ndim', 'N/A')}. Skipping.")
            except Exception as e_inner:
                print(f"Warning: Error accessing data for {name}: {e_inner}. Skipping.")

except FileNotFoundError:
    print(f"Error: Spectrogram file not found at {spectrogram_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading spectrogram NPZ: {e}. Exiting.")
    sys.exit(1)

if not chunk_counts:
    print("No valid chunk data found to analyze. Exiting.")
    sys.exit(0)

# --- Summary Statistics ---
print("\n--- Summary Statistics: Chunks per File ---")
chunk_series = pd.Series(chunk_counts)
print(chunk_series.describe())
print("\nValue Counts (Top 10):")
print(chunk_series.value_counts().head(10))

# --- Create Plot --- 
print("\nGenerating chunk count histogram...")
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plot_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Determine bins - use discrete bins if max count is small
max_chunks = chunk_series.max()
if max_chunks <= 15: # Use discrete bins for small numbers
    bins = np.arange(chunk_series.min(), max_chunks + 2) - 0.5
else: # Use a reasonable number of bins otherwise
    bins = min(50, int(max_chunks) + 1)

sns.histplot(chunk_series, bins=bins, kde=False, discrete=(max_chunks <= 15))
plt.title('Distribution of Spectrogram Chunks per File')
plt.xlabel('Number of Chunks (N)')
plt.ylabel('Number of Files (Samplenames)')
plt.xticks(np.arange(0, max_chunks + 1, step=max(1, int(max_chunks/10)))) # Adjust x-ticks
plt.grid(True)

plot_save_path = os.path.join(plot_dir, "chunk_distribution_histogram.png")
try:
    plt.savefig(plot_save_path)
    print(f"Saved chunk distribution histogram to: {plot_save_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.close()

print("\nChunk Distribution EDA script finished.")
