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
import math 
from tqdm import tqdm 

warnings.filterwarnings("ignore")

def load_and_prepare_metadata(config):
    """Loads taxonomy and training metadata, prepares the working dataframe."""
    print("--- 1. Loading and Preparing Metadata ---")
    print("Loading taxonomy data...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
    except Exception as e:
        print(f"Error loading taxonomy data: {e}")
        raise

    print("Loading training metadata...")
    try:
        train_df = pd.read_csv(config.train_csv_path)
    except Exception as e:
        print(f"Error loading training metadata: {e}")
        raise

    print(f'Found {len(train_df["primary_label"].unique())} unique species initially.')

    working_df = train_df[["primary_label", "rating", "filename"]].copy()
    working_df["filepath"] = config.train_audio_dir + "/" + working_df.filename
    working_df["class"] = working_df.primary_label.map(
        lambda x: species_class_map.get(x, "Unknown")
    )

    if config.debug and config.N_MAX_PREPROCESS is not None:
        print(f"DEBUG: Limiting processing to {config.N_MAX_PREPROCESS} samples.")
        working_df = working_df.head(config.N_MAX_PREPROCESS).copy() 

    print(f'Total samples to process: {len(working_df)}')
    return working_df

def generate_and_save_spectrograms(df, config):
    """Generates spectrograms using utils function and saves them."""
    fabio_intervals = {}
    try:
        fabio_df = pd.read_csv(config.FABIO_CSV_PATH)
        fabio_intervals = {row['filename']: (row['start'], row['stop']) for _, row in fabio_df.iterrows()}
        print(f"Loaded Fabio intervals for {len(fabio_intervals)} files from {config.FABIO_CSV_PATH}")
    except Exception as e:
        print(f"Error loading Fabio CSV from {config.FABIO_CSV_PATH}: {e}")
        raise

    vad_intervals = {}
    try:
        with open(config.VOICE_DATA_PKL_PATH, 'rb') as f:
            raw_vad_data = pickle.load(f)
            vad_intervals = raw_vad_data
            print(f"Loaded VAD intervals for {len(vad_intervals)} files from {config.VOICE_DATA_PKL_PATH}")
    except Exception as e:
        print(f"Error loading VAD pickle from {config.VOICE_DATA_PKL_PATH}: {e}")
        raise

    print("\n--- 2. Generating Spectrograms & Writing Directly to Zip ---")
    start_time = time.time()

    saved_chunk_keys = utils.generate_spectrograms(df, config, fabio_intervals, vad_intervals, config.PREPROCESSED_ZIP_PATH)
    end_time = time.time()
    print(f"Spectrogram generation and zip writing finished in {end_time - start_time:.2f} seconds.")

    if not saved_chunk_keys:
         print("No spectrogram chunks were successfully written to the zip file. Cannot create metadata.")
         return None
         
    print("\n--- 3. Creating Chunked Metadata ---")
    chunked_metadata = []
    df_lookup = df.set_index("filename").to_dict("index")

    for chunk_key in tqdm(saved_chunk_keys, desc="Building Chunk Metadata"):
        try:
            parts = chunk_key.split("_chunk")
            original_filename = parts[0]
            chunk_idx = int(parts[1])

            if original_filename in df_lookup:
                original_metadata = df_lookup[original_filename]

                chunk_row = {
                    "chunk_key": chunk_key,
                    "filename": original_filename,
                    "chunk_index": chunk_idx,
                    "primary_label": original_metadata.get("primary_label"),
                    "rating": original_metadata.get("rating"),
                    "filepath": original_metadata.get("filepath"),
                    "class": original_metadata.get("class"),
                    "secondary_labels": original_metadata.get(
                        "secondary_labels", "[]"
                    ),
                }
                chunked_metadata.append(chunk_row)
            else:
                print(
                    f"Warning: Original filename '{original_filename}' for chunk key '{chunk_key}' not found in metadata lookup. Skipping chunk metadata."
                )

        except (IndexError, ValueError) as e:
            print(
                f"Warning: Could not parse chunk key '{chunk_key}'. Error: {e}. Skipping chunk metadata."
            )
    
    chunked_df = pd.DataFrame(chunked_metadata)
    print(f"Generated chunked metadata DataFrame with {len(chunked_df)} rows.")

    print(f"Saving chunked metadata to: {config.CHUNKED_METADATA_PATH}")
    try:
        os.makedirs(os.path.dirname(config.CHUNKED_METADATA_PATH), exist_ok=True)
        chunked_df.to_csv(config.CHUNKED_METADATA_PATH, index=False)
        print(f"Successfully saved chunked metadata.")
    except Exception as e:
        print(f"Error saving chunked metadata to {config.CHUNKED_METADATA_PATH}: {e}")
        # If metadata fails, the zip file still exists but training will likely fail.
        # Consider deleting the zip file here if metadata is essential?

    return chunked_df

def plot_examples(spectrogram_data, chunked_df, config):
    """Plots and saves example spectrograms using chunked data."""
    print("\n--- 4. Plotting Example Spectrograms (Optional) ---")
    if not spectrogram_data:
        print("No spectrograms generated, skipping plotting.")
        return
        
    if chunked_df is None or chunked_df.empty:
        print("Chunked metadata dataframe is missing or empty, cannot plot examples with labels.")
        return

    # Use chunk keys directly from spectrogram_data for plotting
    available_chunk_keys = list(spectrogram_data.keys())
    if not available_chunk_keys:
        print("No spectrogram keys available to plot.")
        return
        
    # Plot a sample of the CHUNKS
    max_plot_samples = min(4, len(available_chunk_keys))
    print(f"Plotting examples for {max_plot_samples} spectrogram chunks.")
    
    # Select random chunk keys to plot
    plot_keys = np.random.choice(available_chunk_keys, max_plot_samples, replace=False)
    
    # Prepare plot grid
    ncols = 2
    nrows = math.ceil(max_plot_samples / ncols)
    plt.figure(figsize=(8 * ncols, 6 * nrows))
    plot_count = 0
    
    # Create lookup from chunked_df for labels
    metadata_lookup = chunked_df.set_index('chunk_key')

    for chunk_key in plot_keys:
        if chunk_key in metadata_lookup.index:
            metadata = metadata_lookup.loc[chunk_key]
            spectrogram = spectrogram_data[chunk_key]
            
            plt.subplot(nrows, ncols, plot_count + 1)
            plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            title = f"{metadata['class']}: {metadata['primary_label']}\n(Chunk Key: {chunk_key})"
            plt.title(title)
            plt.colorbar(format='%+2.0f dB')
            plot_count += 1
        else:
            print(f"Warning: Metadata for chunk key '{chunk_key}' not found in chunked DataFrame. Skipping plot.")

    if plot_count > 0:
        plt.tight_layout()
        plot_filename = f"melspec_chunk_examples_{config._MODE_STR}.png"
        plot_filepath = os.path.join(config.OUTPUT_DIR, plot_filename)
        print(f"Saving example chunk plot to: {plot_filepath}")
        try:
            os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
            plt.savefig(plot_filepath)
        except Exception as e:
             print(f"Error saving plot to {plot_filepath}: {e}")
        plt.close()
    else:
        print("Could not prepare any valid chunk samples for plotting.")

def main(config):
    """Main function to run the preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    print(f"Using device: {config.device}") 
    print(f"Debug preprocessing mode: {'ON' if config.debug_preprocessing_mode else 'OFF'}") 

    working_df = load_and_prepare_metadata(config)
    if working_df is None:
         print("Metadata loading failed. Exiting preprocessing.")
         return 
         
    # Generate chunks directly into zip, create metadata
    # This function now returns the chunked_metadata_df or None on failure
    chunked_metadata_df = generate_and_save_spectrograms(working_df, config)

    if chunked_metadata_df is None:
        print("Spectrogram generation or metadata creation failed. Preprocessing incomplete.")
    else:
         print(f"\nPreprocessing script finished successfully. Output: {config.PREPROCESSED_ZIP_PATH}, {config.CHUNKED_METADATA_PATH}")

if __name__ == "__main__":
    main(config)