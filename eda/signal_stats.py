import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm.auto import tqdm
import librosa
import torch
import multiprocessing
from pathlib import Path
import traceback
from functools import partial
import warnings

# Add project root to sys.path to allow importing config and models
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config
from models.efficient_at.preprocess import AugmentMelSTFT # Import spectrogram generator
from birdclef_training import _apply_adain_transformation # Import AdaIN function

warnings.filterwarnings("ignore")

print("--- EDA: Signal Statistics Comparison (All Soundscape Chunks vs. All Train Audio Chunks) ---")

# --- Configuration & Paths ---
PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "signal_stats_comparison_all_chunks")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
N_MELS = config.N_MELS # Get N_MELS from config for convenience

# --- Helper Functions for Statistics ---
def calculate_signal_stats(spec_2d):
    """Calculates overall and per-frequency mean and std deviation from a 2D spectrogram."""
    if spec_2d is None or spec_2d.size == 0 or spec_2d.shape[0] != N_MELS:
        # Return NaNs for all if spec is invalid or doesn't match expected N_MELS
        nan_array = np.full(N_MELS, np.nan)
        return {
            "overall_mean": np.nan, 
            "overall_std": np.nan,
            "mean_per_freq_bin": nan_array.copy(),
            "std_per_freq_bin": nan_array.copy()
        }

    overall_mean = np.mean(spec_2d)
    overall_std = np.std(spec_2d)
    mean_per_freq_bin = np.mean(spec_2d, axis=1) # Mean across time for each mel bin
    std_per_freq_bin = np.std(spec_2d, axis=1)   # Std across time for each mel bin

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "mean_per_freq_bin": mean_per_freq_bin,
        "std_per_freq_bin": std_per_freq_bin
    }

# --- Worker Function for Soundscape Processing ---
def _process_soundscape_file(audio_path_str, config_obj):
    """Loads one soundscape file, generates all 5s specs, and calculates stats."""
    audio_path = Path(audio_path_str)
    file_id = audio_path.stem
    file_stats = []

    try:
        spectrogram_generator_worker = AugmentMelSTFT(
            n_mels=config_obj.N_MELS,
            sr=config_obj.FS,
            win_length=config_obj.WIN_LENGTH,
            hopsize=config_obj.HOP_LENGTH,
            n_fft=config_obj.N_FFT,
            fmin=config_obj.FMIN,
            fmax=config_obj.FMAX,
            freqm=0, timem=0, fmin_aug_range=1, fmax_aug_range=1
        )
        spectrogram_generator_worker.eval()

        audio_data, _ = librosa.load(audio_path, sr=config_obj.FS, mono=True)
        target_length_samples = int(config_obj.TARGET_DURATION * config_obj.FS)
        total_duration_samples = len(audio_data)
        num_segments = total_duration_samples // target_length_samples

        if num_segments == 0:
            # print(f"Warning: Soundscape {audio_path.name} too short, skipping.")
            return []

        for segment_idx in range(num_segments):
            start_sample = segment_idx * target_length_samples
            end_sample = start_sample + target_length_samples
            segment_audio = audio_data[start_sample:end_sample]
            if len(segment_audio) < target_length_samples:
                segment_audio = np.pad(segment_audio, (0, target_length_samples - len(segment_audio)), mode='constant')

            segment_audio_tensor = torch.from_numpy(segment_audio.astype(np.float32))
            with torch.no_grad():
                mel_spec_tensor = spectrogram_generator_worker(segment_audio_tensor.unsqueeze(0))
            mel_spec_numpy = mel_spec_tensor.squeeze(0).cpu().numpy()

            stats = calculate_signal_stats(mel_spec_numpy)
            stats['id'] = f"{file_id}_chunk{segment_idx}"
            file_stats.append(stats)
        return file_stats
    except Exception as e:
        # print(f"Error processing soundscape worker {audio_path.name}: {e}")
        # print(traceback.format_exc())
        # Return the exception instead of an empty list
        return e # Modified return

# --- Main Processing Logic ---
def main():
    all_stats = []

    # --- 1. Process ALL Soundscape Chunks ---
    print("Processing ALL Soundscape Chunks (Generating Spectrograms)...")
    soundscape_files = list(Path(config.unlabeled_audio_dir).glob('*.ogg'))
    if config.debug and config.debug_limit_files > 0:
         print(f"DEBUG MODE: Limiting soundscape processing to {config.debug_limit_files} files.")
         soundscape_files = soundscape_files[:config.debug_limit_files]

    if not soundscape_files:
        print(f"Warning: No soundscape audio files found in {config.unlabeled_audio_dir}. Skipping.")
    else:
        print(f"Found {len(soundscape_files)} soundscape files to process.")
        worker_func = partial(_process_soundscape_file, config_obj=config)
        soundscape_results = []
        pool = None
        try:
            try: multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError: pass
            print(f"Starting soundscape processing pool with {NUM_WORKERS} workers...")
            pool = multiprocessing.Pool(processes=NUM_WORKERS)
            results_iterator = pool.imap_unordered(worker_func, [str(p) for p in soundscape_files])
            for result in tqdm(results_iterator, total=len(soundscape_files), desc="Processing Soundscapes"):
                # Check if the result is an exception
                if isinstance(result, Exception):
                     print(f"Worker Error: {result}") # Print error if exception returned
                     # Optionally print traceback: print(traceback.format_exception(result))
                elif result: # Check if result is not empty list (original check)
                     soundscape_results.extend(result)
        except Exception as e_pool:
            print(f"CRITICAL ERROR during multiprocessing: {e_pool}")
            print(traceback.format_exc())
        finally:
            if pool: pool.close(); pool.join()
        print(f"Collected stats for {len(soundscape_results)} soundscape chunks.")
        for stats_item in soundscape_results: stats_item['source'] = 'soundscape_all_chunks' # Corrected variable name
        all_stats.extend(soundscape_results)

    # --- 2. Process ALL Train Audio Chunks (from Precomputed) ---
    print("Processing ALL Train Audio Chunks (from Precomputed NPZ)...")
    processed_train_chunks = 0
    # Store original train stats separately for comparison
    original_train_stats = [] 
    transformed_train_stats = []

    try:
        # Open the NPZ file but don't load everything into memory
        with np.load(config.PREPROCESSED_NPZ_PATH) as data:
            # Get the list of keys (samplenames) without loading the data yet
            samplenames = list(data.keys())
            print(f"Found {len(samplenames)} train_audio spectrogram entries in '{config.PREPROCESSED_NPZ_PATH}'.")

            # Iterate through the keys and load/process one sample at a time WITH TQDM
            for samplename in tqdm(samplenames, desc="Train Audio Specs"): # Added tqdm here
                spec_chunks_array = data[samplename] # Load data for this key only
                if spec_chunks_array is not None and spec_chunks_array.ndim == 3 and spec_chunks_array.shape[0] > 0:
                    for i in range(spec_chunks_array.shape[0]):
                        spec_2d_original = spec_chunks_array[i]
                        
                        # Calculate stats for the original spectrogram
                        original_stats = calculate_signal_stats(spec_2d_original)
                        original_stats['source'] = 'train_audio_original' # New source name
                        original_stats['id'] = f"{samplename}_prechunk{i}_orig"
                        original_train_stats.append(original_stats)

                        # Apply AdaIN transformation (if enabled in config, which it should be for this test)
                        if config.APPLY_ADAIN:
                            spec_2d_transformed = _apply_adain_transformation(spec_2d_original.astype(np.float32), config)
                            # Calculate stats for the transformed spectrogram
                            transformed_stats = calculate_signal_stats(spec_2d_transformed)
                            transformed_stats['source'] = 'train_audio_adain' # New source name
                            transformed_stats['id'] = f"{samplename}_prechunk{i}_adain"
                            transformed_train_stats.append(transformed_stats)
                        
                        processed_train_chunks += 1
        print(f"Processed {processed_train_chunks} individual precomputed chunks from train audio.")
        all_stats.extend(original_train_stats) # Add original train stats
        if config.APPLY_ADAIN:
            all_stats.extend(transformed_train_stats) # Add transformed train stats if AdaIN was applied

    except FileNotFoundError as fnf_e:
        print(f"ERROR: Precomputed train audio NPZ file not found: {fnf_e}. Skipping.")
    except Exception as e:
        print(f"ERROR loading/processing precomputed train audio data: {e}. Skipping.")

    # --- 3. Aggregation and Plotting ---
    if not all_stats:
        print("No statistics collected. Exiting.")
        sys.exit(0)

    stats_df = pd.DataFrame(all_stats)
    stats_df.dropna(subset=['overall_mean', 'overall_std'], inplace=True) # Drop if overall stats are NaN

    # Further drop rows where per-frequency arrays might be all NaNs (if any slipped through)
    def check_valid_array(arr):
        return isinstance(arr, np.ndarray) and not np.all(np.isnan(arr))
    stats_df = stats_df[stats_df['mean_per_freq_bin'].apply(check_valid_array)]
    stats_df = stats_df[stats_df['std_per_freq_bin'].apply(check_valid_array)]

    if stats_df.empty:
        print("Statistics DataFrame is empty after dropping NaNs. Exiting.")
        sys.exit(0)

    print(f"Collected a total of {len(stats_df)} valid spectrogram statistics entries.")
    print("Summary Statistics by Source (Overall Metrics):")
    stat_metrics_to_plot_overall = ["overall_mean", "overall_std"]
    summary_overall = stats_df.groupby('source')[stat_metrics_to_plot_overall].agg(['mean', 'median', 'std'])
    print(summary_overall)

    print("Generating comparison plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    freq_bin_indices = np.arange(N_MELS)

    # Histograms for overall stats
    for metric in stat_metrics_to_plot_overall:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=stats_df, x=metric, hue='source', kde=True, bins=50, common_norm=False, stat="density")
        plt.title(f"Distribution of Spectrogram {metric.replace('_', ' ').title()}")
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Density")
        plot_filename = f"{metric}_comparison_all_chunks.png"
        plot_path = os.path.join(PLOT_OUTPUT_DIR, plot_filename)
        try:
            plt.savefig(plot_path)
            print(f"Saved plot: {plot_path}")
        except Exception as e_save:
            print(f"Error saving plot {plot_filename}: {e_save}")
        plt.close()

    # Per-frequency line plots
    print("Aggregating and plotting per-frequency statistics...")
    per_freq_plot_data = {}
    for source_name, group_df in stats_df.groupby('source'):
        if group_df.empty: continue
        # Stack the arrays for mean_per_freq_bin and std_per_freq_bin
        # Ensure all arrays in the group are valid before stacking
        valid_mean_arrays = [arr for arr in group_df['mean_per_freq_bin'] if isinstance(arr, np.ndarray) and arr.shape == (N_MELS,)]
        valid_std_arrays = [arr for arr in group_df['std_per_freq_bin'] if isinstance(arr, np.ndarray) and arr.shape == (N_MELS,)]

        if not valid_mean_arrays or not valid_std_arrays:
            print(f"Warning: Not enough valid per-frequency arrays for source '{source_name}'. Skipping per-frequency plots for this source.")
            continue

        stacked_means = np.stack(valid_mean_arrays) # (num_samples, N_MELS)
        stacked_stds = np.stack(valid_std_arrays)   # (num_samples, N_MELS)

        avg_of_means_per_bin = np.mean(stacked_means, axis=0)
        std_of_means_per_bin = np.std(stacked_means, axis=0)
        avg_of_stds_per_bin = np.mean(stacked_stds, axis=0)
        std_of_stds_per_bin = np.std(stacked_stds, axis=0)
        per_freq_plot_data[source_name] = {
            'avg_mean': avg_of_means_per_bin,
            'std_mean': std_of_means_per_bin,
            'avg_std': avg_of_stds_per_bin,
            'std_std': std_of_stds_per_bin
        }

    # Plotting per-frequency mean
    plt.figure(figsize=(12, 7))
    title_mean = "Average Mean Value per Frequency Bin"
    for source_name, data in per_freq_plot_data.items():
        plt.plot(freq_bin_indices, data['avg_mean'], label=source_name)
        plt.fill_between(freq_bin_indices, data['avg_mean'] - data['std_mean'], data['avg_mean'] + data['std_mean'], alpha=0.2)
    plt.xlabel("Frequency Bin Index")
    plt.ylabel("Average Mean Spectrogram Value")
    plt.title(title_mean)
    plt.legend()
    plt.grid(True)
    plot_filename_mean = "per_freq_avg_mean_comparison.png"
    plot_path_mean = os.path.join(PLOT_OUTPUT_DIR, plot_filename_mean)
    try:
        plt.savefig(plot_path_mean)
        print(f"Saved plot: {plot_path_mean}")
    except Exception as e_save:
        print(f"Error saving plot {plot_filename_mean}: {e_save}")
    plt.close()

    # Plotting per-frequency std dev
    plt.figure(figsize=(12, 7))
    title_std = "Average Standard Deviation per Frequency Bin"
    for source_name, data in per_freq_plot_data.items():
        plt.plot(freq_bin_indices, data['avg_std'], label=source_name)
        plt.fill_between(freq_bin_indices, data['avg_std'] - data['std_std'], data['avg_std'] + data['std_std'], alpha=0.2)
    plt.xlabel("Frequency Bin Index")
    plt.ylabel("Average Std Dev of Spectrogram Values")
    plt.title(title_std)
    plt.legend()
    plt.grid(True)
    plot_filename_std = "per_freq_avg_std_comparison.png"
    plot_path_std = os.path.join(PLOT_OUTPUT_DIR, plot_filename_std)
    try:
        plt.savefig(plot_path_std)
        print(f"Saved plot: {plot_path_std}")
    except Exception as e_save:
        print(f"Error saving plot {plot_filename_std}: {e_save}")
    plt.close()

    print(f"Signal statistics comparison plots saved to: {PLOT_OUTPUT_DIR}")
    print("--- EDA Finished ---")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
