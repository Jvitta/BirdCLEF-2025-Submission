import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm.auto import tqdm

# Add project root to sys.path to allow importing config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import config

print("--- EDA: Signal Statistics Comparison (Soundscape Pseudo vs. Train Audio BN-Guided) ---")

# --- Configuration & Paths ---
SOUNDSCAPE_CONF_THRESHOLD = 0.9 # Use from central config
PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "signal_stats_comparison")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions for Statistics ---
def calculate_signal_stats(spec_2d):
    """Calculates various signal statistics from a 2D spectrogram."""
    if spec_2d is None or spec_2d.size == 0:
        return {"overall_mean": np.nan, "mean_of_max_per_time_frame": np.nan, "mean_of_top_5_percent_values": np.nan}
    
    overall_mean = np.mean(spec_2d)
    
    if spec_2d.shape[1] > 0: # Check if there are time frames
        mean_of_max_per_time_frame = np.mean(np.max(spec_2d, axis=0))
    else: # Handle empty spectrograms or ones with no time frames
        mean_of_max_per_time_frame = np.nan

    # Top 5% of values
    if spec_2d.size > 0:
        flat_spec = spec_2d.flatten()
        top_5_percent_threshold = np.percentile(flat_spec, 95)
        mean_of_top_5_percent_values = np.mean(flat_spec[flat_spec >= top_5_percent_threshold])
    else:
        mean_of_top_5_percent_values = np.nan
        
    return {
        "overall_mean": overall_mean,
        "mean_of_max_per_time_frame": mean_of_max_per_time_frame,
        "mean_of_top_5_percent_values": mean_of_top_5_percent_values
    }

all_stats = []

# --- 1. Process Soundscape Pseudo-Label Spectrograms ---
print("\nProcessing Soundscape Pseudo-Label Spectrograms...")
try:
    soundscape_pseudo_df = pd.read_csv(config.train_pseudo_csv_path)
    soundscape_pseudo_df_filtered = soundscape_pseudo_df[soundscape_pseudo_df['confidence'] >= SOUNDSCAPE_CONF_THRESHOLD].copy()
    print(f"Loaded and filtered {len(soundscape_pseudo_df_filtered)} high-confidence soundscape pseudo-labels (>= {SOUNDSCAPE_CONF_THRESHOLD})")

    with np.load(os.path.join(config._PREPROCESSED_OUTPUT_DIR, 'pseudo_spectrograms.npz')) as data:
        soundscape_specs_data = {k: data[k] for k in data.files}
    print(f"Loaded {len(soundscape_specs_data)} soundscape spectrogram entries from 'pseudo_spectrograms.npz'.")

    for _, row in tqdm(soundscape_pseudo_df_filtered.iterrows(), total=len(soundscape_pseudo_df_filtered), desc="Soundscape Specs"):
        segment_key = f"{row['filename']}_{int(row['start_time'])}_{int(row['end_time'])}"
        spec_array = soundscape_specs_data.get(segment_key)
        if spec_array is not None and spec_array.ndim == 3 and spec_array.shape[0] == 1:
            spec_2d = spec_array[0] # Shape (1, H, W) -> (H, W)
            stats = calculate_signal_stats(spec_2d)
            stats['source'] = 'soundscape_pseudo'
            stats['id'] = segment_key
            all_stats.append(stats)
        # else:
            # print(f"Warning: Spectrogram for soundscape segment {segment_key} not found or invalid shape.")

except FileNotFoundError as fnf_e:
    print(f"ERROR: A required data file for soundscapes was not found: {fnf_e}. Skipping soundscape processing.")
except Exception as e:
    print(f"ERROR loading or processing soundscape data: {e}. Skipping soundscape processing.")


# --- 2. Process Train Audio BirdNET-Guided Spectrograms ---
print("\nProcessing Train Audio BirdNET-Guided Spectrograms...")
try:
    with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
        train_audio_bn_detections_files = list(data.files) # These are original filenames
    print(f"Found {len(train_audio_bn_detections_files)} files with BirdNET detections in '{config.BIRDNET_DETECTIONS_NPZ_PATH}'.")

    with np.load(config.PREPROCESSED_NPZ_PATH) as data:
        train_audio_specs_data = {k: data[k] for k in data.files} # Keys are base_samplenames
    print(f"Loaded {len(train_audio_specs_data)} train_audio spectrogram entries (samplenames) from '{config.PREPROCESSED_NPZ_PATH}'.")
    
    processed_train_chunks = 0
    for bn_filename in tqdm(train_audio_bn_detections_files, desc="Train Audio BN Files"):
        base_samplename_for_file = os.path.splitext(bn_filename.replace('/', '-'))[0]
        
        if base_samplename_for_file in train_audio_specs_data:
            spec_chunks_array = train_audio_specs_data[base_samplename_for_file] # Should be (num_chunks, H, W)
            
            if spec_chunks_array is not None and spec_chunks_array.ndim == 3 and spec_chunks_array.shape[0] > 0:
                for i in range(spec_chunks_array.shape[0]):
                    spec_2d = spec_chunks_array[i] # Individual (H, W) spectrogram
                    chunk_id = f"{base_samplename_for_file}_bnchunk{i}"
                    stats = calculate_signal_stats(spec_2d)
                    stats['source'] = 'train_audio_bn_chunk'
                    stats['id'] = chunk_id
                    all_stats.append(stats)
                    processed_train_chunks +=1
            # else:
                # print(f"Warning: Spectrogram stack for train audio {base_samplename_for_file} not found or invalid shape in preprocessed data.")
        # else:
            # print(f"Debug: Base samplename {base_samplename_for_file} (from BN detections) not found as key in train_audio_specs_data.")
    print(f"Processed {processed_train_chunks} individual BirdNET-guided chunks from train audio.")

except FileNotFoundError as fnf_e:
    print(f"ERROR: A required data file for train audio was not found: {fnf_e}. Skipping train audio processing.")
except Exception as e:
    print(f"ERROR loading or processing train audio data: {e}. Skipping train audio processing.")


# --- 3. Aggregation and Plotting ---
if not all_stats:
    print("\nNo statistics collected. Exiting.")
    sys.exit(0)

stats_df = pd.DataFrame(all_stats)
stats_df.dropna(inplace=True) # Remove rows where stats might be NaN (e.g., empty specs)

if stats_df.empty:
    print("\nStatistics DataFrame is empty after dropping NaNs. Exiting.")
    sys.exit(0)

print(f"\nCollected a total of {len(stats_df)} valid spectrogram statistics entries.")
print("\nSummary Statistics by Source:")
stat_metrics_to_plot = ["overall_mean", "mean_of_max_per_time_frame", "mean_of_top_5_percent_values"]
summary = stats_df.groupby('source')[stat_metrics_to_plot].agg(['mean', 'median', 'std'])
print(summary)


print("\nGenerating comparison plots...")
plt.style.use('seaborn-v0_8-whitegrid')

for metric in stat_metrics_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stats_df, x=metric, hue='source', kde=True, bins=50, common_norm=False, stat="density")
    plt.title(f"Distribution of {metric.replace('_', ' ').title()}\n(Soundscape Pseudo vs. Train Audio BN-Guided)")
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel("Density")
    plot_filename = f"{metric}_comparison.png"
    plot_path = os.path.join(PLOT_OUTPUT_DIR, plot_filename)
    try:
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
    except Exception as e_save:
        print(f"Error saving plot {plot_filename}: {e_save}")
    plt.close()

print(f"\nSignal statistics comparison plots saved to: {PLOT_OUTPUT_DIR}")
print("--- EDA Finished ---")
