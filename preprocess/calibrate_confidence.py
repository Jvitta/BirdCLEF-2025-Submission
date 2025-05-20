import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This correctly sets project_root to BirdCLEF-2025-Submission
sys.path.append(project_root)

from config import config

# Define the output path for the thresholds CSV
OUTPUT_THRESHOLDS_DIR = os.path.join(project_root, "data", "processed")
OUTPUT_THRESHOLDS_CSV = os.path.join(OUTPUT_THRESHOLDS_DIR, "birdnet_training_confidence_thresholds.csv")
PERCENTILE_TO_CALCULATE = 0.25 # Corresponds to the 25th percentile

def calculate_and_save_thresholds(config_obj):
    """
    Loads BirdNET detections from training audio (NPZ file),
    calculates the specified percentile confidence for each species,
    and saves these thresholds to a CSV file.
    """
    npz_path = config_obj.BIRDNET_DETECTIONS_NPZ_PATH
    print(f"--- Calculating Confidence Thresholds from BirdNET Detections on Training Audio ---")
    print(f"Loading BirdNET detections from: {npz_path}")

    try:
        data_npz = np.load(npz_path, allow_pickle=True)
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

    files_in_npz = data_npz.files
    print(f"Found {len(files_in_npz)} files with detections in the NPZ.")

    for filename_key in files_in_npz:
        detections_for_file = data_npz[filename_key]
        primary_label = file_to_label.get(filename_key)
        
        if primary_label is None:
            continue # Skip if filename not in train_csv for label mapping

        for det in detections_for_file:
            if isinstance(det, dict) and 'confidence' in det:
                all_detections_data.append({
                    'primary_label': primary_label,
                    'confidence': det['confidence']
                })
    
    if not all_detections_data:
        print("No valid detection data could be extracted and mapped to species.")
        return

    df_detections = pd.DataFrame(all_detections_data)
    print(f"Total number of 'true positive' detections processed: {len(df_detections)}")

    if df_detections.empty:
        print("DataFrame of detections is empty. Cannot calculate thresholds.")
        return

    print(f"\nCalculating {PERCENTILE_TO_CALCULATE*100:.0f}th percentile confidence threshold per species...")
    
    # Calculate the specified percentile for confidence per species
    species_thresholds = df_detections.groupby('primary_label')['confidence'].quantile(PERCENTILE_TO_CALCULATE).reset_index()
    species_thresholds.rename(columns={'confidence': f'confidence_threshold_p{int(PERCENTILE_TO_CALCULATE*100)}'}, inplace=True)
    
    # Add a count of detections per species for informational purposes
    detection_counts = df_detections.groupby('primary_label').size().reset_index(name='detection_count')
    species_thresholds = pd.merge(species_thresholds, detection_counts, on='primary_label', how='left')
    
    species_thresholds = species_thresholds.sort_values(by=f'confidence_threshold_p{int(PERCENTILE_TO_CALCULATE*100)}', ascending=True)

    if species_thresholds.empty:
        print("No species thresholds could be calculated.")
        return

    # Ensure the output directory exists
    os.makedirs(OUTPUT_THRESHOLDS_DIR, exist_ok=True)
    
    # --- DEBUG: Print absolute path ---
    abs_output_csv_path = os.path.abspath(OUTPUT_THRESHOLDS_CSV)
    print(f"DEBUG: Absolute path for saving CSV: {abs_output_csv_path}")
    # --- END DEBUG ---

    try:
        species_thresholds.to_csv(OUTPUT_THRESHOLDS_CSV, index=False)
        print(f"\nSuccessfully saved per-species confidence thresholds to: {OUTPUT_THRESHOLDS_CSV}")
        print(f"Total species with calculated thresholds: {len(species_thresholds)}")
        print("\nSample of calculated thresholds (first 5):")
        print(species_thresholds.head())
        print("\nSample of calculated thresholds (last 5):")
        print(species_thresholds.tail())

    except Exception as e:
        print(f"Error saving thresholds CSV: {e}")

def apply_calibrated_thresholds_to_soundscape(config_obj):
    """
    Loads soundscape pseudo-labels and filters them using the
    previously calculated species-specific confidence thresholds.
    Saves the filtered pseudo-labels to a new CSV file.
    """
    print(f"\n--- Applying Calibrated Confidence Thresholds to Soundscape Pseudo-Labels ---")
    
    threshold_col_name = f'confidence_threshold_p{int(PERCENTILE_TO_CALCULATE*100)}'
    
    # Check if the thresholds CSV exists
    if not os.path.exists(OUTPUT_THRESHOLDS_CSV):
        print(f"Error: Thresholds CSV file not found at {OUTPUT_THRESHOLDS_CSV}")
        print("Please ensure 'calculate_and_save_thresholds' has run successfully first.")
        return

    try:
        thresholds_df = pd.read_csv(OUTPUT_THRESHOLDS_CSV)
        print(f"Loaded {len(thresholds_df)} species-specific thresholds from: {OUTPUT_THRESHOLDS_CSV}")
    except Exception as e:
        print(f"Error loading thresholds CSV {OUTPUT_THRESHOLDS_CSV}: {e}")
        return

    if threshold_col_name not in thresholds_df.columns:
        print(f"Error: Threshold column '{threshold_col_name}' not found in {OUTPUT_THRESHOLDS_CSV}.")
        print(f"Available columns: {thresholds_df.columns.tolist()}")
        return

    # Load soundscape pseudo-labels
    soundscape_pseudo_path = config_obj.soundscape_pseudo_csv_path
    if not os.path.exists(soundscape_pseudo_path):
        print(f"Error: Soundscape pseudo-label CSV file not found at {soundscape_pseudo_path}")
        print("Please ensure BirdNET has generated pseudo-labels for soundscapes.")
        return
        
    try:
        soundscape_df = pd.read_csv(soundscape_pseudo_path)
        print(f"Loaded {len(soundscape_df)} soundscape pseudo-labels from: {soundscape_pseudo_path}")
    except Exception as e:
        print(f"Error loading soundscape pseudo-label CSV {soundscape_pseudo_path}: {e}")
        return

    if soundscape_df.empty:
        print("Soundscape pseudo-label DataFrame is empty. No labels to filter.")
        return

    # Merge soundscape labels with their respective thresholds
    # Using 'left' merge to keep all soundscape detections initially
    # Detections for species not in thresholds_df will have NaN for threshold_col_name
    merged_df = pd.merge(soundscape_df, thresholds_df[['primary_label', threshold_col_name]], 
                         on='primary_label', how='left')

    # Filter: keep rows where confidence >= species-specific threshold
    # Rows where threshold_col_name is NaN (species not in thresholds_df) will be dropped
    # as 'confidence >= NaN' is False.
    original_detection_count = len(merged_df)
    filtered_df = merged_df[merged_df['confidence'] >= merged_df[threshold_col_name]].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    num_dropped_due_to_missing_threshold = original_detection_count - len(merged_df[merged_df[threshold_col_name].notna()])
    num_dropped_by_thresholding = len(merged_df[merged_df[threshold_col_name].notna()]) - len(filtered_df)
    
    print(f"Original soundscape detections: {original_detection_count}")
    print(f"Detections for species without a calibrated threshold (dropped): {num_dropped_due_to_missing_threshold}")
    print(f"Detections dropped by applying species-specific thresholds: {num_dropped_by_thresholding}")
    print(f"Total calibrated soundscape detections: {len(filtered_df)}")

    if filtered_df.empty:
        print("No soundscape detections remained after applying calibrated thresholds.")
    else:
        print(f"Retained {len(filtered_df)} detections after filtering.")

    # --- Detailed Per-Species Filtering Report ---
    print("\n--- Per-Species Filtering Report ---")
    # Prepare a report DataFrame starting with species and their thresholds
    report_df = thresholds_df[['primary_label', threshold_col_name]].copy()

    # Get initial counts from the original soundscape_df for species that have a threshold
    relevant_soundscape_detections = soundscape_df[soundscape_df['primary_label'].isin(report_df['primary_label'])]
    initial_counts_per_species = relevant_soundscape_detections.groupby('primary_label').size().rename('initial_detections')
    report_df = report_df.merge(initial_counts_per_species, on='primary_label', how='left').fillna({'initial_detections': 0})
    report_df['initial_detections'] = report_df['initial_detections'].astype(int)

    # Get final counts from the filtered_df
    final_counts_per_species = filtered_df.groupby('primary_label').size().rename('final_detections')
    report_df = report_df.merge(final_counts_per_species, on='primary_label', how='left').fillna({'final_detections': 0})
    report_df['final_detections'] = report_df['final_detections'].astype(int)
    
    report_df['dropped_detections'] = report_df['initial_detections'] - report_df['final_detections']
    
    # Sort by most dropped to least, then by initial detections
    report_df = report_df.sort_values(by=['dropped_detections', 'initial_detections', 'primary_label'], ascending=[False, False, True])

    species_with_soundscape_detections_count = 0
    for _, row in report_df.iterrows():
        if row['initial_detections'] > 0: # Only print for species that had detections to begin with
            species_with_soundscape_detections_count += 1
            print(f"  Species: {row['primary_label']:<15} | "
                  f"Threshold: {row[threshold_col_name]:.4f} | "
                  f"Initial: {row['initial_detections']:<5} | "
                  f"Dropped: {row['dropped_detections']:<5} | "
                  f"Kept: {row['final_detections']:<5}")
    
    if species_with_soundscape_detections_count == 0:
        print("No species with defined thresholds had any detections in the soundscape data to report on.")
    print("--- End of Per-Species Filtering Report ---")
    # --- End Detailed Report ---

    # Ensure the output directory exists
    output_calibrated_csv_path = config_obj.soundscape_pseudo_calibrated_csv_path
    output_calibrated_dir = os.path.dirname(output_calibrated_csv_path)
    os.makedirs(output_calibrated_dir, exist_ok=True)

    try:
        # Save only the original columns from the soundscape data
        filtered_df[soundscape_df.columns].to_csv(output_calibrated_csv_path, index=False)
        print(f"Successfully saved calibrated soundscape pseudo-labels to: {output_calibrated_csv_path}")
    except Exception as e:
        print(f"Error saving calibrated soundscape pseudo-labels CSV: {e}")


if __name__ == "__main__":
    # First, calculate and save the thresholds from training data
    calculate_and_save_thresholds(config)
    
    # Then, apply these thresholds to the soundscape pseudo-labels
    apply_calibrated_thresholds_to_soundscape(config)
