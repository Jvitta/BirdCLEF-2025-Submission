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

    print(f'Found {len(train_df["primary_label"].unique())} unique species in main dataset initially.')

    # Prepare main dataframe
    main_working_df = train_df[['primary_label', 'filename']].copy()
    main_working_df['filepath'] = config.train_audio_dir + '/' + main_working_df.filename
    main_working_df['samplename'] = main_working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    main_working_df['class'] = main_working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
    print(f"Processed {len(main_working_df)} samples from main dataset.")

    # --- Load and process rare data --- #
    print("Loading rare species metadata...")
    rare_working_df = None
    try:
        rare_train_df = pd.read_csv(config.train_rare_csv_path, sep=';')
        print(f'Found {len(rare_train_df["primary_label"].unique())} unique species in rare dataset initially.')

        # Prepare rare dataframe
        rare_working_df = rare_train_df[['primary_label', 'filename']].copy()
        # Use the correct directory for rare audio files
        rare_working_df['filepath'] = config.train_audio_rare_dir + '/' + rare_working_df.filename
        # Ensure samplename generation is consistent
        rare_working_df['samplename'] = rare_working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        rare_working_df['class'] = rare_working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
        print(f"Processed {len(rare_working_df)} samples from rare dataset.")

    except FileNotFoundError:
        print(f"Warning: Rare training CSV not found at {config.train_rare_csv_path}. Proceeding without rare data.")
    except Exception as e:
        print(f"Error loading or processing rare training CSV: {e}. Proceeding without rare data.")
    # --- End Load Rare Data --- #

    # --- Combine DataFrames --- #
    if rare_working_df is not None:
        print("Combining main and rare datasets...")
        working_df = pd.concat([main_working_df, rare_working_df], ignore_index=True)
    else:
        print("Using only main dataset.")
        working_df = main_working_df
    # --- End Combine DataFrames --- #

    # Original debug limiting logic applied AFTER combining
    if config.debug and config.N_MAX_PREPROCESS is not None:
        print(f"DEBUG: Limiting processing to {config.N_MAX_PREPROCESS} samples from the combined dataset.")
        working_df = working_df.head(config.N_MAX_PREPROCESS).copy()

    print(f'Total samples to process: {len(working_df)}')
    print(f'Samples by class:')
    print(working_df['class'].value_counts())
    return working_df

def generate_and_save_spectrograms(df, config):
    """Generates spectrograms using utils function (with intervals) and saves as NPZ."""
    if df is None or df.empty:
        print("Working dataframe is empty, skipping spectrogram generation.")
        return # Return nothing, as utils.generate_spectrograms handles saving

    # --- Load Fabio Intervals --- #
    fabio_intervals = {}
    try:
        fabio_df = pd.read_csv(config.FABIO_CSV_PATH)
        fabio_intervals = {row['filename']: (row['start'], row['stop']) for _, row in fabio_df.iterrows()}
        print(f"Loaded Fabio intervals for {len(fabio_intervals)} files from {config.FABIO_CSV_PATH}")
    except FileNotFoundError:
        print(f"Warning: Fabio CSV file not found at {config.FABIO_CSV_PATH}. Fabio-specific cropping will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading Fabio CSV from {config.FABIO_CSV_PATH}: {e}. Fabio-specific cropping will not be applied.")
    # --- End Load Fabio Intervals ---

    # --- Load VAD Intervals --- #
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
    # --- End Load VAD Intervals ---

    print("\n--- 2. Generating and Saving Spectrograms (Single NPZ File) ---")
    start_time = time.time()
    # Call the utility function which now saves the NPZ file directly
    utils.generate_spectrograms(df, config, fabio_intervals, vad_intervals)
    end_time = time.time()
    print(f"Spectrogram generation/saving process finished in {end_time - start_time:.2f} seconds.")

def plot_examples(df, config):
    """Plots and saves example spectrograms by loading from the single NPZ file."""
    print("\n--- 3. Plotting Example Spectrograms (Optional) ---")

    npz_filepath = config.PREPROCESSED_NPZ_PATH
    if not os.path.exists(npz_filepath):
        print(f"Preprocessed data file ({npz_filepath}) does not exist. Skipping plotting.")
        return

    if df is None or df.empty:
        print("Metadata dataframe is missing, cannot plot examples with labels.")
        return

    samples_to_plot = []
    data_archive = None
    try:
        print(f"Loading NPZ archive: {npz_filepath}")
        # Load the archive. Allows reading individual arrays later.
        data_archive = np.load(npz_filepath)
        available_keys = set(data_archive.keys())
        print(f"Loaded archive with {len(available_keys)} samples.")

        # Try to find up to 4 samples present in the NPZ file
        # Using head(50) to increase chances of finding matches if some failed processing
        potential_samples_df = df[['samplename', 'class', 'primary_label']].drop_duplicates('samplename').head(50)

        found_count = 0
        for _, row in potential_samples_df.iterrows():
            if found_count >= 4:
                break

            samplename = row['samplename']
            if samplename in available_keys:
                try:
                    spectrogram_data = data_archive[samplename]
                    samples_to_plot.append((samplename, row['class'], row['primary_label'], spectrogram_data))
                    found_count += 1
                except Exception as e:
                    print(f"Warning: Error accessing/reading sample {samplename} from NPZ archive: {e}")
            # else: # Optionally uncomment to see which potential samples were not found
            #     print(f"Debug: Did not find key '{samplename}' in NPZ archive.")

        if found_count < 4:
            print(f"Found only {found_count} valid spectrogram examples to plot out of {len(potential_samples_df)} checked samples present in the NPZ.")

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
            print("Could not find or load any valid spectrogram samples from the NPZ file to plot examples.")

    except Exception as e:
        print(f"Error during NPZ loading or plotting: {e}")
    finally:
        # Ensure the archive file handle is closed if it was opened
        if data_archive is not None:
            try:
                data_archive.close()
            except Exception as e_close:
                 print(f"Warning: Error closing NPZ archive: {e_close}")

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