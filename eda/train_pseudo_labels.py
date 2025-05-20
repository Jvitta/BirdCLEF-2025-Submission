import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This should be 'BirdCLEF-2025-Submission'
sys.path.append(project_root)

from config import config

def plot_train_audio_confidence_per_species(config_obj):
    """
    Loads BirdNET detections from training audio (NPZ file),
    and plots the confidence distribution for each species using facet plots.
    Saves the plot to eda/plots/birdnet_train_audio_confidence_per_species.png.
    """
    npz_path = config_obj.BIRDNET_DETECTIONS_NPZ_PATH
    print(f"Loading BirdNET detections for training audio from: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: NPZ file not found at {npz_path}")
        print("Please ensure you have run 'preprocess/birdnet_preprocessing.py' first.")
        return
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    all_detections_data = []
    # The NPZ file saves a dictionary where keys are filenames
    # and values are lists of detection dicts for that file's target species.
    # We need to associate the filename (which implies the species) with the confidence.
    
    # To get the primary_label for each filename, we need train.csv
    try:
        train_df_meta = pd.read_csv(config_obj.train_csv_path, usecols=['filename', 'primary_label'])
        file_to_label = dict(zip(train_df_meta['filename'], train_df_meta['primary_label']))
    except FileNotFoundError:
        print(f"Error: train.csv not found at {config_obj.train_csv_path}. Cannot map filenames to species.")
        return
    except Exception as e:
        print(f"Error loading train.csv: {e}")
        return

    for filename_key in data.files:
        detections_for_file = data[filename_key]
        primary_label = file_to_label.get(filename_key)
        if primary_label is None:
            # This might happen if the NPZ contains files not in train.csv or if keys are mangled.
            # print(f"Warning: Could not find primary_label for filename '{filename_key}' in train.csv. Skipping.")
            continue

        for det in detections_for_file:
            if isinstance(det, dict) and 'confidence' in det:
                all_detections_data.append({
                    'primary_label': primary_label,
                    'confidence': det['confidence']
                })
            # else: # Handle cases where items might not be dicts if structure varies
                # print(f"Warning: Unexpected detection format for {filename_key}: {det}")

    if not all_detections_data:
        print("No valid detection data found in the NPZ file or could not map to species.")
        return

    df = pd.DataFrame(all_detections_data)

    if df.empty:
        print("Detection data is empty after processing. No distribution to plot.")
        return

    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(project_root, "eda", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_dir, "birdnet_train_audio_confidence_per_species.png")

    unique_species = df['primary_label'].unique()
    n_species = len(unique_species)
    print(f"Found {n_species} unique species in BirdNET training audio detections.")

    if n_species == 0:
        print("No species found to plot.")
        return

    # Determine grid layout
    ncols = 5  # Number of columns in the facet grid
    nrows = (n_species + ncols - 1) // ncols # Calculate rows needed

    plt.style.use('seaborn-v0_8-whitegrid')

    g = sns.FacetGrid(df, col='primary_label', col_wrap=ncols, height=2.5, aspect=1.5, sharex=True, sharey=False)
    g.map(sns.histplot, 'confidence', bins=25, kde=False, color='dodgerblue') 
    
    g.set_titles("{col_name}", size=10)
    g.set_axis_labels("Confidence Score (BirdNET on Train Audio)", "Frequency")
    
    plt.subplots_adjust(top=0.92) 
    g.fig.suptitle(f'BirdNET Detection Confidence per Species (Training Audio)\\n{n_species} species shown, Filtered by Ground Truth', fontsize=16, y=0.98)

    try:
        plt.savefig(output_plot_path, dpi=150)
        print(f"Per-species confidence distribution plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(g.fig) # Close the plot

if __name__ == "__main__":
    plot_train_audio_confidence_per_species(config)
