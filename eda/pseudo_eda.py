import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This should be 'BirdCLEF-2025-Submission'
sys.path.append(project_root)

from config import config

def summarize_train_audio_detections(config_obj):
    """
    Loads BirdNET detections from training audio (NPZ file),
    calculates summary statistics, and prints them.
    """
    npz_path = config_obj.BIRDNET_DETECTIONS_NPZ_PATH
    print(f"--- EDA for BirdNET Detections on Training Audio ---")
    print(f"Loading BirdNET detections from: {npz_path}\n")

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
    try:
        train_df_meta = pd.read_csv(config_obj.train_csv_path, usecols=['filename', 'primary_label'])
        file_to_label = dict(zip(train_df_meta['filename'], train_df_meta['primary_label']))
    except FileNotFoundError:
        print(f"Error: train.csv not found at {config_obj.train_csv_path}. Cannot map filenames to species.")
        return
    except Exception as e:
        print(f"Error loading train.csv: {e}")
        return

    files_in_npz = data.files
    print(f"Found {len(files_in_npz)} files with detections in the NPZ.")

    for filename_key in files_in_npz:
        detections_for_file = data[filename_key]
        primary_label = file_to_label.get(filename_key)
        
        if primary_label is None:
            # print(f"Warning: Could not find primary_label for filename '{filename_key}' in train.csv. Skipping these detections.")
            continue

        for det in detections_for_file:
            if isinstance(det, dict) and 'confidence' in det:
                all_detections_data.append({
                    'filename': filename_key, # Keep filename for more detailed grouping if needed later
                    'primary_label': primary_label,
                    'confidence': det['confidence']
                })
    
    if not all_detections_data:
        print("No valid detection data could be extracted and mapped to species.")
        return

    df = pd.DataFrame(all_detections_data)
    print(f"\nTotal number of detections processed: {len(df)}")

    if df.empty:
        print("DataFrame is empty after processing. No statistics to show.")
        return

    print("\n--- Overall Confidence Statistics ---")
    print(df['confidence'].describe())

    num_unique_species = df['primary_label'].nunique()
    print(f"\nNumber of unique species with detections: {num_unique_species}")

    print("\n--- Per-Species Statistics ---")
    species_stats = df.groupby('primary_label')['confidence'].agg(
        count='count',
        mean_confidence='mean',
        median_confidence='median',
        min_confidence='min',
        max_confidence='max',
        std_confidence='std'
    ).sort_values(by='count', ascending=False)
    
    pd.set_option('display.max_rows', None) # Show all rows for species stats
    print(species_stats)
    pd.reset_option('display.max_rows')

    print("\n--- Top 10 Species by Detection Count ---")
    print(species_stats.head(10))

    print("\n--- Bottom 10 Species by Detection Count (min 1 detection) ---")
    print(species_stats[species_stats['count'] > 0].tail(10))
    
    # Count species with zero detections from the original train.csv if desired
    all_train_species = set(train_df_meta['primary_label'].unique())
    detected_species = set(df['primary_label'].unique())
    species_with_no_detections = all_train_species - detected_species
    if species_with_no_detections:
        print(f"\n--- Species in train.csv with NO Detections in NPZ (based on current filters) ---")
        print(f"Count: {len(species_with_no_detections)}")
        # for i, s in enumerate(sorted(list(species_with_no_detections))):
        #     print(f"  {s}")
        #     if i >= 19 and len(species_with_no_detections) > 20 : # Print first 20 if list is too long
        #         print(f"  ... and {len(species_with_no_detections) - 20} more.")
        #         break
    else:
        print("\nAll species from train.csv with audio files had at least one detection.")

    print("\n--- Detailed Debug for Specific Species ---")
    for species_code_debug in ['gycwor1', 'neocor']:
        print(f"\n--- Debugging: {species_code_debug} ---")
        species_df = df[df['primary_label'] == species_code_debug]
        if species_df.empty:
            print(f"No detections found for {species_code_debug} in the DataFrame.")
            continue
        
        print(f"Number of detections for {species_code_debug}: {len(species_df)}")
        print(f"Confidence scores sample (first 20):\n{species_df['confidence'].head(20).values}")
        # print(f"Confidence scores sample (last 20):\n{species_df['confidence'].tail(20).values}") # Optional
        print(f"Statistics for {species_code_debug} confidences:")
        print(species_df['confidence'].describe())
        
        plt.figure(figsize=(10, 5))
        sns.histplot(species_df['confidence'], bins=50, kde=True, color='purple')
        plt.title(f'Confidence Distribution for {species_code_debug}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        debug_plot_dir = os.path.join(project_root, "eda", "plots", "debug")
        os.makedirs(debug_plot_dir, exist_ok=True)
        debug_plot_path = os.path.join(debug_plot_dir, f"{species_code_debug}_confidence_debug.png")
        try:
            plt.savefig(debug_plot_path)
            print(f"Debug plot saved to: {debug_plot_path}")
        except Exception as e:
            print(f"Error saving debug plot: {e}")
        plt.close()

    print("\nEDA Finished.")

if __name__ == "__main__":
    summarize_train_audio_detections(config)









