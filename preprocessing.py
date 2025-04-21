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
        
    fabio_intervals = {}
    try:
        fabio_df = pd.read_csv(config.FABIO_CSV_PATH)
        fabio_intervals = {row['filename']: (row['start'], row['stop']) for _, row in fabio_df.iterrows()}
        print(f"Loaded Fabio intervals for {len(fabio_intervals)} files from {config.FABIO_CSV_PATH}")
    except FileNotFoundError:
        print(f"Warning: Fabio CSV file not found at {config.FABIO_CSV_PATH}. Fabio-specific cropping will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading Fabio CSV from {config.FABIO_CSV_PATH}: {e}. Fabio-specific cropping will not be applied.")

    vad_intervals = {}
    try:
        with open(config.TRANSFORMED_VOICE_DATA_PKL_PATH, 'rb') as f:
            print(f"Loading TRANSFORMED VAD data from {config.TRANSFORMED_VOICE_DATA_PKL_PATH}...")
            vad_intervals = pickle.load(f)
            print(f"Loaded {len(vad_intervals)} VAD entries with corrected keys.")
    except FileNotFoundError:
        print(f"Warning: TRANSFORMED VAD pickle file not found at {config.TRANSFORMED_VOICE_DATA_PKL_PATH}. Please run transform_vad_keys.py first. VAD-based cropping will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading TRANSFORMED VAD pickle from {config.TRANSFORMED_VOICE_DATA_PKL_PATH}: {e}. VAD-based cropping will not be applied.")

    print("\n--- 2. Generating and Saving Spectrograms (Individual Files) ---")
    start_time = time.time()
    utils.generate_spectrograms(df, config, fabio_intervals, vad_intervals)
    end_time = time.time()
    print(f"Spectrogram generation/saving process finished in {end_time - start_time:.2f} seconds.")

def plot_examples(df, config):
    """Plots and saves example spectrograms by loading individual files."""
    print("\n--- 3. Plotting Example Spectrograms (Optional) ---")
    
    preprocessed_dir = config.PREPROCESSED_DATA_DIR
    if not os.path.exists(preprocessed_dir) or not os.listdir(preprocessed_dir):
        print(f"Preprocessed data directory ({preprocessed_dir}) is empty or does not exist. Skipping plotting.")
        return
        
    if df is None or df.empty:
        print("Metadata dataframe is missing, cannot plot examples with labels.")
        return

    samples_to_plot = []
    potential_samples_df = df[['samplename', 'class', 'primary_label']].drop_duplicates('samplename').head(20)
    
    found_count = 0
    for _, row in potential_samples_df.iterrows():
        if found_count >= 4:
            break
            
        samplename = row['samplename']
        expected_filepath = os.path.join(preprocessed_dir, f"{samplename}.npy")
        
        if os.path.exists(expected_filepath):
            try:
                spectrogram_data = np.load(expected_filepath)
                samples_to_plot.append((samplename, row['class'], row['primary_label'], spectrogram_data))
                found_count += 1
            except Exception as e:
                print(f"Warning: Error loading example spectrogram {expected_filepath}: {e}")
        # else: # Optionally uncomment to see which potential samples were not found
        #     print(f"Debug: Did not find expected file {expected_filepath} for plotting.")

    if found_count < 4:
         print(f"Found only {found_count} valid spectrogram examples to plot out of {len(potential_samples_df)} checked.")

    if samples_to_plot:
        plt.figure(figsize=(16, 12))
        for i, (samplename, class_name, species, spec_data) in enumerate(samples_to_plot):
            plt.subplot(2, 2, i + 1)
            plt.imshow(spec_data, aspect='auto', origin='lower', cmap='viridis') 
            plt.title(f"{class_name}: {species} (Sample: {samplename})", fontsize=10)
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plot_mode_str = "sample" if config.debug_preprocessing_mode and getattr(config, 'N_MAX_PREPROCESS', None) is not None else "full"
        plot_filename = f"melspec_examples_{plot_mode_str}.png" 
        plot_filepath = os.path.join(config.OUTPUT_DIR, plot_filename) 
        print(f"Saving example plot to: {plot_filepath}")
        try:
            os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
            plt.savefig(plot_filepath)
            plt.close()
        except Exception as e:
             print(f"Error saving plot to {plot_filepath}: {e}")
            
    else:
        print("Could not find or load any valid spectrogram files to plot examples.")

def main(config):
    """Main function to run the preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    print(f"Using device: {config.device}") 
    print(f"Debug mode: {'ON' if config.debug else 'OFF'}") 
    print(f"Preprocessing Debug mode: {'ON' if config.debug_preprocessing_mode else 'OFF'}")

    working_df = load_and_prepare_metadata(config)

    generate_and_save_spectrograms(working_df, config) 

    plot_examples(working_df, config)

    print("\nPreprocessing script finished.")

if __name__ == "__main__":
    main(config)