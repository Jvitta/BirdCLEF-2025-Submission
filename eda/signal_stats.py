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
from birdclef_training import _apply_adain_transformation, _load_adain_per_freq_stats

warnings.filterwarnings("ignore")

print("--- EDA: Signal Statistics Comparison (Soundscape vs. Train Original vs. Train AdaIN Transformed) ---")

# --- Configuration & Paths ---
PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "signal_stats_full_comparison")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
N_MELS = config.N_MELS

# --- Helper Functions for Statistics ---
def calculate_signal_stats(spec_2d):
    """Calculates overall, per-frequency, and per-time mean and std deviation from a 2D spectrogram."""
    if spec_2d is None or spec_2d.size == 0 or spec_2d.shape[0] != N_MELS:
        nan_array_freq = np.full(N_MELS, np.nan)
        # Assuming PREPROCESS_TARGET_SHAPE[1] gives the number of time bins for the original 5s spec
        # If spec_2d is invalid, we don't know its width, so we use a sensible default or make it dynamic later if needed.
        # For now, let's assume config.PREPROCESS_TARGET_SHAPE[1] is accessible or a fixed value like 500 for time_bins.
        # This part might need adjustment if spec_2d can have variable width AND be invalid.
        # However, the check spec_2d.shape[0] != N_MELS implies we expect a certain structure.
        # Let's get N_TIMEBINS from config as it should be fixed for valid inputs.
        time_bins = config.PREPROCESS_TARGET_SHAPE[1] 
        nan_array_time = np.full(time_bins, np.nan)
        return {
            "overall_mean": np.nan, 
            "overall_std": np.nan,
            "mean_per_freq_bin": nan_array_freq.copy(),
            "std_per_freq_bin": nan_array_freq.copy(),
            "mean_per_time_bin": nan_array_time.copy(),
            "std_per_time_bin": nan_array_time.copy()
        }

    overall_mean = np.mean(spec_2d)
    overall_std = np.std(spec_2d)
    mean_per_freq_bin = np.mean(spec_2d, axis=1) # Across time
    std_per_freq_bin = np.std(spec_2d, axis=1)   # Across time
    mean_per_time_bin = np.mean(spec_2d, axis=0) # Across frequency
    std_per_time_bin = np.std(spec_2d, axis=0)   # Across frequency

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "mean_per_freq_bin": mean_per_freq_bin,
        "std_per_freq_bin": std_per_freq_bin,
        "mean_per_time_bin": mean_per_time_bin,
        "std_per_time_bin": std_per_time_bin
    }

# --- Worker Function for Soundscape Processing ---
def _process_soundscape_file(audio_path_str, config_obj):
    audio_path = Path(audio_path_str)
    file_id = audio_path.stem
    file_stats = []
    try:
        spectrogram_generator_worker = AugmentMelSTFT(
            n_mels=config_obj.N_MELS, sr=config_obj.FS, win_length=config_obj.WIN_LENGTH,
            hopsize=config_obj.HOP_LENGTH, n_fft=config_obj.N_FFT, fmin=config_obj.FMIN,
            fmax=config_obj.FMAX, freqm=0, timem=0,
            fmin_aug_range=config_obj.FMIN_AUG_RANGE, fmax_aug_range=config_obj.FMAX_AUG_RANGE
        )
        spectrogram_generator_worker.eval()
        audio_data, _ = librosa.load(audio_path, sr=config_obj.FS, mono=True)
        target_length_samples = int(config_obj.TARGET_DURATION * config_obj.FS)
        num_segments = len(audio_data) // target_length_samples
        if num_segments == 0: return []
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
        return e

# --- Main Processing Logic ---
def main():
    all_stats_collection = [] # To collect stats dictionaries before DataFrame

    # --- Define save path for AdaIN stats & ensure directory exists ---
    adain_stats_filename = "adain_per_frequency_stats.npz"
    os.makedirs(config._PREPROCESSED_OUTPUT_DIR, exist_ok=True)
    adain_stats_save_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, adain_stats_filename)

# --- 1. Process ALL Soundscape Chunks ---
print("Processing ALL Soundscape Chunks (Generating Spectrograms)...")
soundscape_files = list(Path(config.unlabeled_audio_dir).glob('*.ogg'))
if config.debug and config.debug_limit_files > 0:
     print(f"DEBUG MODE: Limiting soundscape processing to {config.debug_limit_files} files.")
     soundscape_files = soundscape_files[:config.debug_limit_files]

if not soundscape_files:
        print(f"Warning: No soundscape audio files found in {config.unlabeled_audio_dir}. Skipping soundscape processing.")
else:
    print(f"Found {len(soundscape_files)} soundscape files to process.")
    worker_func = partial(_process_soundscape_file, config_obj=config)
        soundscape_results_raw = []
    pool = None
    try:
        try: multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError: pass
        print(f"Starting soundscape processing pool with {NUM_WORKERS} workers...")
        pool = multiprocessing.Pool(processes=NUM_WORKERS)
        results_iterator = pool.imap_unordered(worker_func, [str(p) for p in soundscape_files])
            for result in tqdm(results_iterator, total=len(soundscape_files), desc="Processing Soundscapes"):
                if isinstance(result, Exception): print(f"Worker Error: {result}")
                elif result: soundscape_results_raw.extend(result)
    except Exception as e_pool:
            print(f"CRITICAL ERROR during multiprocessing: {e_pool}\\n{traceback.format_exc()}")
    finally:
        if pool: pool.close(); pool.join()
        
        for stats_item in soundscape_results_raw: stats_item['source'] = 'soundscape_all_chunks'
        all_stats_collection.extend(soundscape_results_raw)
        print(f"Collected stats for {len(soundscape_results_raw)} soundscape chunks.")

    # --- 2. Process ALL Original Train Audio Chunks (from Precomputed) ---
    print("Processing ALL Original Train Audio Chunks (from Precomputed NPZ)...")
    original_train_stats_list = []
try:
    with np.load(config.PREPROCESSED_NPZ_PATH) as data:
            samplenames = list(data.keys())
            print(f"Found {len(samplenames)} train_audio spectrogram entries in '{config.PREPROCESSED_NPZ_PATH}'.")
            for samplename in tqdm(samplenames, desc="Original Train Specs"):
                spec_chunks_array = data[samplename]
        if spec_chunks_array is not None and spec_chunks_array.ndim == 3 and spec_chunks_array.shape[0] > 0:
            for i in range(spec_chunks_array.shape[0]):
                        spec_2d_original = spec_chunks_array[i]
                        stats = calculate_signal_stats(spec_2d_original)
                        stats['source'] = 'train_audio_original'
                        stats['id'] = f"{samplename}_prechunk{i}_orig"
                        original_train_stats_list.append(stats)
        all_stats_collection.extend(original_train_stats_list)
        print(f"Collected stats for {len(original_train_stats_list)} original train chunks.")
except FileNotFoundError as fnf_e:
        print(f"ERROR: Precomputed train audio NPZ file not found: {fnf_e}. Cannot process original train stats.")
except Exception as e:
        print(f"ERROR loading/processing precomputed train audio data for original stats: {e}.")

    # --- 3. Create and Save AdaIN Statistics NPZ (from soundscape and original train) ---
    print("\\n--- Preparing and Saving Per-Frequency AdaIN Statistics NPZ ---")
    
    # Create a temporary DataFrame from the collected stats so far
    if not all_stats_collection:
        print("ERROR: No stats collected for soundscapes or original train. Cannot create AdaIN NPZ. Exiting.")
        sys.exit(1)
        
    temp_df_for_npz = pd.DataFrame([s for s in all_stats_collection if s['source'] in ['soundscape_all_chunks', 'train_audio_original']])
    
    # Clean this temporary DataFrame
    temp_df_for_npz.dropna(subset=['overall_mean', 'overall_std'], inplace=True)
    def check_valid_array_for_npz(arr): return isinstance(arr, np.ndarray) and not np.all(np.isnan(arr)) and arr.shape == (N_MELS,)
    temp_df_for_npz = temp_df_for_npz[temp_df_for_npz['mean_per_freq_bin'].apply(check_valid_array_for_npz)]
    temp_df_for_npz = temp_df_for_npz[temp_df_for_npz['std_per_freq_bin'].apply(check_valid_array_for_npz)]

    per_freq_data_to_save = {}
    source_train_key_npz = 'train_audio_original'
    source_soundscape_key_npz = 'soundscape_all_chunks'

    for source_name, group_df in temp_df_for_npz.groupby('source'):
        if source_name not in [source_train_key_npz, source_soundscape_key_npz]: continue
        if group_df.empty:
            print(f"Warning: Empty group for source '{source_name}' when preparing NPZ. This source will be missing from NPZ.")
            continue
        
        stacked_means = np.stack(group_df['mean_per_freq_bin'].tolist())
        stacked_stds = np.stack(group_df['std_per_freq_bin'].tolist())
        
        per_freq_data_to_save[source_name] = {
            'avg_mean': np.mean(stacked_means, axis=0), 'std_mean': np.std(stacked_means, axis=0),
            'avg_std': np.mean(stacked_stds, axis=0), 'std_std': np.std(stacked_stds, axis=0)
        }

    if source_train_key_npz not in per_freq_data_to_save or source_soundscape_key_npz not in per_freq_data_to_save:
        print(f"ERROR: Missing critical data for '{source_train_key_npz}' or '{source_soundscape_key_npz}' for AdaIN NPZ. Exiting.")
        sys.exit(1)
    
    try:
        np.savez_compressed(
            adain_stats_save_path,
            mu_t_mean_per_freq=per_freq_data_to_save[source_train_key_npz]['avg_mean'],
            sigma_t_mean_per_freq=per_freq_data_to_save[source_train_key_npz]['std_mean'],
            mu_t_std_per_freq=per_freq_data_to_save[source_train_key_npz]['avg_std'],
            sigma_t_std_per_freq=per_freq_data_to_save[source_train_key_npz]['std_std'],
            mu_ss_mean_per_freq=per_freq_data_to_save[source_soundscape_key_npz]['avg_mean'],
            sigma_ss_mean_per_freq=per_freq_data_to_save[source_soundscape_key_npz]['std_mean'],
            mu_ss_std_per_freq=per_freq_data_to_save[source_soundscape_key_npz]['avg_std'],
            sigma_ss_std_per_freq=per_freq_data_to_save[source_soundscape_key_npz]['std_std']
        )
        print(f"Successfully saved per-frequency AdaIN statistics to: {adain_stats_save_path}")
        # Crucially, clear the global cache in birdclef_training if it was loaded by another part of the code.
        # This forces _apply_adain_transformation to reload the new NPZ.
        if 'birdclef_training' in sys.modules:
            sys.modules['birdclef_training']._adain_per_freq_stats_cache = None 
            print("Cleared _adain_per_freq_stats_cache in birdclef_training module.")

    except Exception as e_save_npz:
        print(f"ERROR saving per-frequency AdaIN statistics NPZ file: {e_save_npz}. Exiting.")
        sys.exit(1)
    
    del temp_df_for_npz, per_freq_data_to_save # Free memory

    # --- 4. Process Transformed Train Audio Chunks (using the new NPZ) ---
    print("\\nProcessing Transformed Train Audio Chunks (using newly saved AdaIN NPZ)...")
    transformed_train_stats_list = []
    if config.ADAIN_MODE == 'none':
        print("AdaIN mode is 'none'. Skipping transformed train stats calculation.")
    else:
        try:
            with np.load(config.PREPROCESSED_NPZ_PATH) as data: # Re-load original train specs
                samplenames = list(data.keys())
                print(f"Re-iterating {len(samplenames)} train_audio entries for AdaIN transformation.")
                for samplename in tqdm(samplenames, desc="Transformed Train Specs"):
                    spec_chunks_array = data[samplename]
                    if spec_chunks_array is not None and spec_chunks_array.ndim == 3 and spec_chunks_array.shape[0] > 0:
                        for i in range(spec_chunks_array.shape[0]):
                            spec_2d_original = spec_chunks_array[i]
                            # Apply AdaIN (will now load the NPZ we just saved)
                            spec_2d_transformed = _apply_adain_transformation(spec_2d_original.astype(np.float32), config)
                            stats = calculate_signal_stats(spec_2d_transformed)
                            stats['source'] = 'train_audio_adain'
                            stats['id'] = f"{samplename}_prechunk{i}_adain"
                            transformed_train_stats_list.append(stats)
            all_stats_collection.extend(transformed_train_stats_list)
            print(f"Collected stats for {len(transformed_train_stats_list)} AdaIN transformed train chunks.")
        except FileNotFoundError: # Should not happen if original train stats were processed
             print(f"ERROR: Precomputed train audio NPZ file not found for transformation. This should not happen.")
        except Exception as e:
            print(f"ERROR processing transformed train audio data: {e}.")


    # --- 5. Final Aggregation and Plotting (all three sources) ---
    if not all_stats_collection:
        print("No statistics collected at all. Exiting.")
    sys.exit(0)

    stats_df = pd.DataFrame(all_stats_collection)
    stats_df.dropna(subset=['overall_mean', 'overall_std'], inplace=True)
    stats_df = stats_df[stats_df['mean_per_freq_bin'].apply(check_valid_array_for_npz)] # Use the stricter check
    stats_df = stats_df[stats_df['std_per_freq_bin'].apply(check_valid_array_for_npz)]   # Use the stricter check
    
    # Add similar check for per-timeframe arrays
    def check_valid_time_array(arr): 
        return isinstance(arr, np.ndarray) and not np.all(np.isnan(arr)) and arr.shape == (config.PREPROCESS_TARGET_SHAPE[1],)
    stats_df = stats_df[stats_df['mean_per_time_bin'].apply(check_valid_time_array)]
    stats_df = stats_df[stats_df['std_per_time_bin'].apply(check_valid_time_array)]

if stats_df.empty:
        print("Final statistics DataFrame is empty after dropping NaNs. Exiting.")
    sys.exit(0)

    print(f"\\nCollected a total of {len(stats_df)} valid spectrogram statistics entries for plotting.")
print("Summary Statistics by Source (Overall Metrics):")
stat_metrics_to_plot_overall = ["overall_mean", "overall_std"]
summary_overall = stats_df.groupby('source')[stat_metrics_to_plot_overall].agg(['mean', 'median', 'std'])
print(summary_overall)

plt.style.use('seaborn-v0_8-whitegrid')

    # Overall Histograms
    print("\\nGenerating overall distribution plots...")
for metric in stat_metrics_to_plot_overall:
        plt.figure(figsize=(12, 7))
    sns.histplot(data=stats_df, x=metric, hue='source', kde=True, bins=50, common_norm=False, stat="density")
    plt.title(f"Distribution of Spectrogram {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel("Density")
        plot_filename = f"{metric}_comparison_all_sources.png"
    plot_path = os.path.join(PLOT_OUTPUT_DIR, plot_filename)
        try: plt.savefig(plot_path); print(f"Saved plot: {plot_path}")
        except Exception as e_save: print(f"Error saving plot {plot_filename}: {e_save}")
    plt.close()

# Per-frequency line plots
    print("\\nAggregating and plotting per-frequency statistics for all sources...")
    final_per_freq_plot_data = {}
for source_name, group_df in stats_df.groupby('source'):
    if group_df.empty: continue
        
        # Ensure arrays are correctly stacked (they should be valid due to earlier check_valid_array_for_npz)
        stacked_means = np.stack(group_df['mean_per_freq_bin'].tolist())
        stacked_stds = np.stack(group_df['std_per_freq_bin'].tolist())

        final_per_freq_plot_data[source_name] = {
            'avg_mean': np.mean(stacked_means, axis=0), 'std_mean': np.std(stacked_means, axis=0),
            'avg_std': np.mean(stacked_stds, axis=0), 'std_std': np.std(stacked_stds, axis=0)
        }
    
    if not final_per_freq_plot_data:
        print("Warning: No per-frequency data aggregated for final plotting. Skipping these plots.")
    else:
        # --- Add printing of the aggregated per-frequency data ---
        print("\n--- Aggregated Per-Frequency Statistics (for plotting) ---")
        for source, data_vals in final_per_freq_plot_data.items():
            print(f"Source: {source}")
            for stat_name, stat_array in data_vals.items():
                # Print first few values to give an idea, without flooding the console
                print(f"  {stat_name} (first 5 bins): {stat_array[:5]}")
            print("---")
        # --- End of printing section ---

        freq_bin_indices = np.arange(N_MELS)
        
        # Define plot styles for better distinction
        plot_styles = {
            'soundscape_all_chunks': {'linestyle': '-', 'marker': 'None', 'color': 'blue'}, # Solid blue
            'train_audio_original': {'linestyle': '--', 'marker': 'None', 'color': 'green'}, # Dashed green
            'train_audio_adain': {'linestyle': ':', 'marker': 'None', 'color': 'red'}     # Dotted red
        }
        default_style = {'linestyle': '-', 'marker': 'None', 'color': 'black'} # Fallback

# Plotting per-frequency mean
        plt.figure(figsize=(14, 8)) # Slightly larger figure
        title_mean = "Average Mean Value per Frequency Bin (All Sources)"
        for source_name, data_vals in final_per_freq_plot_data.items():
            style = plot_styles.get(source_name, default_style)
            plt.plot(freq_bin_indices, data_vals['avg_mean'], 
                     label=source_name, 
                     linestyle=style['linestyle'], 
                     marker=style['marker'],
                     color=style['color'])
            plt.fill_between(freq_bin_indices, data_vals['avg_mean'] - data_vals['std_mean'], 
                             data_vals['avg_mean'] + data_vals['std_mean'], 
                             alpha=0.15, color=style['color']) # Reduced alpha for fill
        plt.xlabel("Frequency Bin Index"); plt.ylabel("Average Mean Spectrogram Value"); plt.title(title_mean); plt.legend(); plt.grid(True)
        plot_filename_mean = "per_freq_avg_mean_comparison_all_sources.png"
plot_path_mean = os.path.join(PLOT_OUTPUT_DIR, plot_filename_mean)
        try: plt.savefig(plot_path_mean); print(f"Saved plot: {plot_path_mean}")
        except Exception as e_save: print(f"Error saving plot {plot_filename_mean}: {e_save}")
plt.close()

# Plotting per-frequency std dev
        plt.figure(figsize=(14, 8)) # Slightly larger figure
        title_std = "Average Standard Deviation per Frequency Bin (All Sources)"
        for source_name, data_vals in final_per_freq_plot_data.items():
            style = plot_styles.get(source_name, default_style)
            plt.plot(freq_bin_indices, data_vals['avg_std'], 
                     label=source_name, 
                     linestyle=style['linestyle'], 
                     marker=style['marker'],
                     color=style['color'])
            plt.fill_between(freq_bin_indices, data_vals['avg_std'] - data_vals['std_std'], 
                             data_vals['avg_std'] + data_vals['std_std'], 
                             alpha=0.15, color=style['color']) # Reduced alpha for fill
        plt.xlabel("Frequency Bin Index"); plt.ylabel("Average Std Dev of Spectrogram Values"); plt.title(title_std); plt.legend(); plt.grid(True)
        plot_filename_std = "per_freq_avg_std_comparison_all_sources.png"
plot_path_std = os.path.join(PLOT_OUTPUT_DIR, plot_filename_std)
        try: plt.savefig(plot_path_std); print(f"Saved plot: {plot_path_std}")
        except Exception as e_save: print(f"Error saving plot {plot_filename_std}: {e_save}")
        plt.close()

    # --- New: Per-Timeframe Line Plots ---
    print("\\nAggregating and plotting per-timeframe statistics...")
    final_per_time_plot_data = {}
    # We are interested in 'soundscape_all_chunks' and 'train_audio_original' for this specific request
    # and 'train_audio_adain' if available and desired for comparison.
    sources_for_time_plots = ['soundscape_all_chunks', 'train_audio_original']
    if 'train_audio_adain' in stats_df['source'].unique():
        sources_for_time_plots.append('train_audio_adain')

    for source_name in sources_for_time_plots:
        group_df = stats_df[stats_df['source'] == source_name]
        if group_df.empty: 
            print(f"Warning: No data for source '{source_name}' for per-timeframe plots.")
            continue
        
        # Ensure arrays are correctly stacked
        stacked_time_means = np.stack(group_df['mean_per_time_bin'].tolist())
        stacked_time_stds = np.stack(group_df['std_per_time_bin'].tolist())

        final_per_time_plot_data[source_name] = {
            'avg_mean_time': np.mean(stacked_time_means, axis=0),
            'std_mean_time': np.std(stacked_time_means, axis=0),
            'avg_std_time': np.mean(stacked_time_stds, axis=0),
            'std_std_time': np.std(stacked_time_stds, axis=0)
        }

    if not final_per_time_plot_data:
        print("Warning: No per-timeframe data aggregated for plotting. Skipping these plots.")
    else:
        time_bin_indices = np.arange(config.PREPROCESS_TARGET_SHAPE[1])
        # Use the same plot_styles as defined for per-frequency plots for consistency

        # Plotting per-timeframe mean
        plt.figure(figsize=(14, 8))
        title_mean_time = "Average Mean Value per Time Bin"
        for source_name, data_vals in final_per_time_plot_data.items():
            style = plot_styles.get(source_name, default_style) # plot_styles and default_style defined earlier
            plt.plot(time_bin_indices, data_vals['avg_mean_time'], 
                     label=source_name, 
                     linestyle=style['linestyle'], 
                     marker=style['marker'],
                     color=style['color'])
            plt.fill_between(time_bin_indices, data_vals['avg_mean_time'] - data_vals['std_mean_time'], 
                             data_vals['avg_mean_time'] + data_vals['std_mean_time'], 
                             alpha=0.15, color=style['color'])
        plt.xlabel("Time Bin Index"); plt.ylabel("Average Mean Spectrogram Value (across freqs)"); plt.title(title_mean_time); plt.legend(); plt.grid(True)
        plot_filename_mean_time = "per_time_avg_mean_comparison.png"
        plot_path_mean_time = os.path.join(PLOT_OUTPUT_DIR, plot_filename_mean_time)
        try: plt.savefig(plot_path_mean_time); print(f"Saved plot: {plot_path_mean_time}")
        except Exception as e_save: print(f"Error saving plot {plot_filename_mean_time}: {e_save}")
        plt.close()

        # Plotting per-timeframe std dev
        plt.figure(figsize=(14, 8))
        title_std_time = "Average Standard Deviation per Time Bin"
        for source_name, data_vals in final_per_time_plot_data.items():
            style = plot_styles.get(source_name, default_style)
            plt.plot(time_bin_indices, data_vals['avg_std_time'], 
                     label=source_name, 
                     linestyle=style['linestyle'], 
                     marker=style['marker'],
                     color=style['color'])
            plt.fill_between(time_bin_indices, data_vals['avg_std_time'] - data_vals['std_std_time'], 
                             data_vals['avg_std_time'] + data_vals['std_std_time'], 
                             alpha=0.15, color=style['color'])
        plt.xlabel("Time Bin Index"); plt.ylabel("Average Std Dev of Spectrogram Values (across freqs)"); plt.title(title_std_time); plt.legend(); plt.grid(True)
        plot_filename_std_time = "per_time_avg_std_comparison.png"
        plot_path_std_time = os.path.join(PLOT_OUTPUT_DIR, plot_filename_std_time)
        try: plt.savefig(plot_path_std_time); print(f"Saved plot: {plot_path_std_time}")
        except Exception as e_save: print(f"Error saving plot {plot_filename_std_time}: {e_save}")
plt.close()

    print(f"\\nSignal statistics comparison plots saved to: {PLOT_OUTPUT_DIR}")
print("--- EDA Finished ---")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
