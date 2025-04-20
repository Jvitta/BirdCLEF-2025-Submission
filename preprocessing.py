import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from config import config 
import birdclef_utils as utils 

warnings.filterwarnings("ignore")

def load_and_prepare_metadata(config):
    """Loads taxonomy and training metadata, prepares the working dataframe."""
    print("--- 1. Loading and Preparing Metadata ---")
    print("Loading taxonomy data...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
    except FileNotFoundError:
        print(f"Error: Taxonomy file not found at {config.taxonomy_path}")
        return None
    except Exception as e:
        print(f"Error loading taxonomy data: {e}")
        return None

    print("Loading training metadata...")
    try:
        train_df = pd.read_csv(config.train_csv_path)
    except FileNotFoundError:
        print(f"Error: Training CSV not found at {config.train_csv_path}")
        return None
    except Exception as e:
        print(f"Error loading training metadata: {e}")
        return None

    print(f'Found {len(train_df["primary_label"].unique())} unique species initially.')

    working_df = train_df[['primary_label', 'rating', 'filename']].copy()
    working_df['filepath'] = config.train_audio_dir + '/' + working_df.filename
    working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))

    if config.debug and config.N_MAX_PREPROCESS is not None:
        print(f"DEBUG: Limiting processing to {config.N_MAX_PREPROCESS} samples.")
        working_df = working_df.head(config.N_MAX_PREPROCESS).copy() 

    print(f'Total samples to process: {len(working_df)}')
    print(f'Samples by class:')
    print(working_df['class'].value_counts())
    return working_df

def generate_and_save_spectrograms(df, config):
    """Generates spectrograms dictionary using utils function (with intervals) and saves it."""
    if df is None or df.empty:
        print("Working dataframe is empty, skipping spectrogram generation.")
        return {} 
        
    # --- Define Paths for VAD/Fabio data on GCP --- #
    # TODO: Move these path definitions to config.py later
    voice_separation_dir = os.path.join(config.GCS_MOUNT_POINT, "BC25 voice separation")
    fabio_csv_path = os.path.join(voice_separation_dir, "fabio.csv")
    voice_data_pkl_path = os.path.join(voice_separation_dir, "voice_data.pkl")
    # --- End Path Definitions --- #

    # --- Load Precomputed Interval Data (adapted from previous version) --- #
    fabio_intervals = {}
    try:
        fabio_df = pd.read_csv(fabio_csv_path)
        # Use the relative filename as the key (assuming CSV has 'filename', 'start', 'stop')
        fabio_intervals = {row['filename']: (row['start'], row['stop']) for _, row in fabio_df.iterrows()}
        print(f"Loaded Fabio intervals for {len(fabio_intervals)} files from {fabio_csv_path}")
    except FileNotFoundError:
        print(f"Warning: Fabio CSV file not found at {fabio_csv_path}. Fabio-specific cropping will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading Fabio CSV from {fabio_csv_path}: {e}. Fabio-specific cropping will not be applied.")

    vad_intervals = {}
    try:
        with open(voice_data_pkl_path, 'rb') as f:
            # Assuming the keys in the pkl file are full paths relative to DATA_ROOT or similar
            # We need to ensure they match df['filepath'] keys
            raw_vad_data = pickle.load(f)
            # Store with full filepath as key, matching df['filepath']
            vad_intervals = raw_vad_data
            print(f"Loaded VAD intervals for {len(vad_intervals)} files from {voice_data_pkl_path}")
            # --- VAD Key Check (from previous version) --- #
            if vad_intervals and df is not None and not df.empty:
                print("--- Checking VAD Key Format ---")
                vad_keys_sample = list(vad_intervals.keys())[:3]
                df_paths_sample = df['filepath'].iloc[:3].tolist()
                print(f"First 3 VAD keys: {vad_keys_sample}")
                print(f"First 3 df['filepath'] values: {df_paths_sample}")
                # Basic check for one key
                example_df_path = df_paths_sample[0]
                if example_df_path in vad_intervals:
                     print(f"Format Check: First df path ('{example_df_path}') found in VAD keys. Good.")
                else:
                     # Try checking with basename if full path fails
                     example_df_basename = os.path.basename(example_df_path)
                     # Be careful with list slicing if vad_keys_sample is short
                     matching_keys = [k for k in vad_keys_sample if example_df_basename in k] 
                     if matching_keys:
                         print(f"Format Check WARNING: First df path ('{example_df_path}') NOT found directly, but basename matched key(s) like '{matching_keys[0]}'. Check key format consistency!")
                     else:
                         print(f"Format Check CRITICAL WARNING: First df path ('{example_df_path}') or its basename ('{example_df_basename}') NOT found in VAD keys sample: {vad_keys_sample}. Check paths in config.py and pkl file source.")
                print("-------------------------------")
            # --- End Key Check --- 

    except FileNotFoundError:
        print(f"Warning: VAD pickle file not found at {voice_data_pkl_path}. VAD-based cropping will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading VAD pickle from {voice_data_pkl_path}: {e}. VAD-based cropping will not be applied.")
    # --- End Load Precomputed Interval Data --- #
        
    print("\n--- 2. Generating Spectrograms ---")
    start_time = time.time()
    all_bird_data = utils.generate_spectrograms(df, config, fabio_intervals, vad_intervals)
    end_time = time.time()
    print(f"Spectrogram generation finished in {end_time - start_time:.2f} seconds.")

    print("\n--- 3. Saving Processed Data ---")
    if not all_bird_data:
        print("No spectrograms were generated.")
        return all_bird_data
        
    print(f"Saving processed data dictionary to: {config.PREPROCESSED_FILEPATH}")
    try:
        os.makedirs(os.path.dirname(config.PREPROCESSED_FILEPATH), exist_ok=True)
        np.save(config.PREPROCESSED_FILEPATH, all_bird_data, allow_pickle=True)
        print(f"Successfully saved data ({len(all_bird_data)} items).")
    except Exception as e:
        print(f"Error saving data to {config.PREPROCESSED_FILEPATH}: {e}")
        
    return all_bird_data

def plot_examples(spectrogram_data, df, config):
    """Plots and saves example spectrograms."""
    print("\n--- 4. Plotting Example Spectrograms (Optional) ---")
    if not spectrogram_data:
        print("No spectrograms generated, skipping plotting.")
        return
        
    if df is None or df.empty:
        print("Metadata dataframe is missing, cannot plot examples with labels.")
        return

    samples = []
    available_samples_df = df[df['samplename'].isin(spectrogram_data.keys())]
    max_plot_samples = min(4, len(available_samples_df))

    if max_plot_samples > 0:
        plot_df = available_samples_df.head(max_plot_samples)
        for _, row in plot_df.iterrows():
            if row['samplename'] in spectrogram_data:
                 samples.append((row['samplename'], row['class'], row['primary_label']))
            else:
                 print(f"Warning: Samplename {row['samplename']} from dataframe not found in generated spectrograms.")

    if samples:
        plt.figure(figsize=(16, 12))
        plot_count = 0
        for i, (samplename, class_name, species) in enumerate(samples):
            if i >= 4: 
                break 
            plt.subplot(2, 2, plot_count + 1)
            plt.imshow(spectrogram_data[samplename], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{class_name}: {species} (Sample: {samplename})")
            plt.colorbar(format='%+2.0f dB')
            plot_count += 1

        if plot_count > 0:
            plt.tight_layout()
            plot_filename = f"melspec_examples_{config._MODE_STR}.png" 
            plot_filepath = os.path.join(config.OUTPUT_DIR, plot_filename) 
            print(f"Saving example plot to: {plot_filepath}")
            try:
                os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
                plt.savefig(plot_filepath)
            except Exception as e:
                 print(f"Error saving plot to {plot_filepath}: {e}")
            plt.close()  
        else:
            print("Could not prepare any valid samples for plotting.")
            
    else:
        print("No samples found in the generated data to plot.")

def main(config):
    """Main function to run the preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    print(f"Using device: {config.device}") 
    print(f"Debug mode: {'ON' if config.debug else 'OFF'}") 

    working_df = load_and_prepare_metadata(config)

    all_bird_data = generate_and_save_spectrograms(working_df, config) 

    plot_examples(all_bird_data, working_df, config)

    print("\nPreprocessing script finished.")

if __name__ == "__main__":
    main(config)