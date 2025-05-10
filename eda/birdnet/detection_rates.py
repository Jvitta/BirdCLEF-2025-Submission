# eda/eda_birdnet_classification.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# Assuming config.py is two levels up (e.g., from eda/ -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

# Hardcode the single species not covered based on previous EDA
# (Same as in preprocess/birdnet_preprocessing.py)
UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots") # Save plots in eda/plots

print("--- BirdNET Detection Results EDA ---")

# --- Load Metadata to get total Aves count ---
print("Loading training metadata...")
total_analyzed_aves_files = 0
try:
    train_df = pd.read_csv(config.train_csv_path)
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    
    # Check for required columns before merge
    if 'scientific_name' not in train_df.columns:
        raise KeyError("'scientific_name' column missing from train.csv")
    if 'primary_label' not in train_df.columns or 'primary_label' not in taxonomy_df.columns:
        raise KeyError("'primary_label' column missing from one or both dataframes for merge")
    if 'class_name' not in taxonomy_df.columns:
        raise KeyError("'class_name' column missing from taxonomy.csv")

    # Merge only the class_name from taxonomy_df
    train_df = pd.merge(
        train_df,
        taxonomy_df[['primary_label', 'class_name']], # Select only needed cols
        on='primary_label',
        how='left'
    )
    
    # Check class_name after merge
    if train_df['class_name'].isnull().any():
         print("Warning: Some training files couldn't be matched with taxonomy to get class_name.")

    # Filter for Aves, excluding the one known uncovered species
    aves_df = train_df[
        (train_df['class_name'] == 'Aves') &
        (train_df['scientific_name'] != UNCOVERED_AVES_SCIENTIFIC_NAME) # Use local constant
    ]
    total_analyzed_aves_files = len(aves_df)
    print(f"Found {total_analyzed_aves_files} 'Aves' files eligible for BirdNET analysis in metadata.")
    
except FileNotFoundError as e:
    print(f"Error: Metadata file not found: {e}. Cannot calculate total Aves count.")
except Exception as e:
    print(f"Error loading or processing metadata: {e}. Cannot calculate total Aves count.")
# --- End Metadata Loading ---

# --- Load BirdNET Detections --- 
detections_path = config.BIRDNET_DETECTIONS_NPZ_PATH
print(f"Loading BirdNET detections from: {detections_path}")

try:
    # Load the npz file, allow pickle for loading object arrays
    with np.load(detections_path, allow_pickle=True) as data:
        # Create a dictionary from the NpzFile object
        birdnet_detections = {key: data[key] for key in data.files}
    files_in_npz = len(birdnet_detections)
    print(f"Loaded detections for {files_in_npz} files from NPZ.")
    if not birdnet_detections:
        print("Loaded NPZ file is empty. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    print(f"Error: Detections file not found at {detections_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading detections NPZ: {e}. Exiting.")
    sys.exit(1)
# --- End Loading Detections --- 

# --- Analyze Data ---
num_detections_per_file = []
all_confidences = []
files_with_zero_detections = 0

print("Analyzing detection data...")
for filename, detections_list in birdnet_detections.items():
    if not hasattr(detections_list, '__iter__'):
        print(f"Warning: Unexpected data format for {filename}. Expected list/array, got {type(detections_list)}. Skipping.")
        continue

    num_detections = len(detections_list)
    num_detections_per_file.append(num_detections)
    
    if num_detections == 0:
        files_with_zero_detections += 1

    for detection in detections_list:
        if isinstance(detection, dict) and 'confidence' in detection:
            all_confidences.append(detection['confidence'])


print(f"Finished analysis.")
print(f"Found {len(all_confidences)} total detections across all files in NPZ.")

# --- Summary Statistics --- (Revised)
print("\n--- Summary Statistics ---")

if total_analyzed_aves_files > 0:
    print(f"Total 'Aves' files eligible for analysis: {total_analyzed_aves_files}")
else:
    print("Could not determine total eligible 'Aves' files from metadata.")
    
print(f"Files processed and saved in NPZ:       {files_in_npz}")
print(f"Files in NPZ with zero detections:      {files_with_zero_detections}")

if files_in_npz > 0:
    perc_zero_in_npz = (files_with_zero_detections / files_in_npz) * 100
    print(f"  Percentage of NPZ files with 0 detections: {perc_zero_in_npz:.2f}%")

if total_analyzed_aves_files > 0:
    # This represents files that *should* have been processed but yielded no detections
    perc_zero_total_eligible = (files_with_zero_detections / total_analyzed_aves_files) * 100
    print(f"  Percentage of eligible Aves files resulting in 0 detections: {perc_zero_total_eligible:.2f}%")
    
    # Calculate files potentially missed during processing (not in NPZ)
    files_missed_processing = total_analyzed_aves_files - files_in_npz
    if files_missed_processing > 0:
        print(f"Eligible 'Aves' files potentially missed during processing (not in NPZ): {files_missed_processing}")

if num_detections_per_file:
    detections_series = pd.Series(num_detections_per_file)
    print("\nNumber of Detections per File (for files in NPZ):")
    print(detections_series.describe())
else:
    print("\nNo data on number of detections per file.")

if all_confidences:
    confidence_series = pd.Series(all_confidences)
    print("\nConfidence Scores of Detections (for all detections found):")
    print(confidence_series.describe())
else:
    print("\nNo confidence scores found.")

# --- Analysis per Species --- 
print("\n--- Analysis per Bird Species ---")
if 'aves_df' in locals() and not aves_df.empty and birdnet_detections:
    # Create a set of filenames that have at least one detection
    filenames_with_detections = {fn for fn, dlist in birdnet_detections.items() if hasattr(dlist, '__len__') and len(dlist) > 0}
    print(f"Identified {len(filenames_with_detections)} files with >= 1 detection in NPZ.")

    # Group aves_df by primary_label (species)
    species_stats = []
    grouped_species = aves_df.groupby('primary_label')

    for species_label, group in tqdm(grouped_species, desc="Analyzing Species"):
        total_files_species = group['filename'].nunique()
        
        # Count how many files for this species are in the set of files with detections
        files_with_dets_species = group[group['filename'].isin(filenames_with_detections)]['filename'].nunique()
        
        percentage_with_dets = (files_with_dets_species / total_files_species) * 100 if total_files_species > 0 else 0
        
        species_stats.append({
            'species': species_label,
            'total_files': total_files_species,
            'files_with_detections': files_with_dets_species,
            'percentage_detected': percentage_with_dets
        })

    if species_stats:
        species_stats_df = pd.DataFrame(species_stats)
        species_stats_df = species_stats_df.sort_values(by='percentage_detected', ascending=True)
        
        print("\nDetection Rate per Species (Lowest First):")
        # Print head and tail for brevity
        print("--- Lowest Detection Rates ---")
        print(species_stats_df.head(15))
        print("\n--- Highest Detection Rates ---")
        print(species_stats_df.tail(15))
        
        # Optionally save the full df
        stats_save_path = os.path.join(plot_dir, "species_detection_stats.csv")
        try:
            species_stats_df.to_csv(stats_save_path, index=False)
            print(f"\nSaved full species stats to: {stats_save_path}")
        except Exception as e:
            print(f"\nError saving species stats CSV: {e}")
    else:
        print("Could not calculate species-specific stats.")
else:
    print("Skipping species-specific analysis due to missing data (Aves metadata or detections).")

# --- Create Plots --- (Keep plots the same, title clarifies scope)
print("\nGenerating plots...")
os.makedirs(plot_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid') 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Distribution of Number of Detections per File (for files in NPZ)
if num_detections_per_file:
    sns.histplot(detections_series, bins=max(1, min(50, int(detections_series.max())+1)), kde=False, ax=axes[0]) # Ensure bins are int
    axes[0].set_title('Distribution of Detections per File (Files in NPZ)')
    axes[0].set_xlabel('Number of Detections (Target Species, > Threshold)')
    axes[0].set_ylabel('Number of Files')
else:
    axes[0].set_title('No Detection Count Data to Plot')

# Plot 2: Distribution of Confidence Scores (for all detections found)
if all_confidences:
    sns.histplot(confidence_series, bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Detection Confidence Scores (All Detections)')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
else:
    axes[1].set_title('No Confidence Score Data to Plot')

plt.tight_layout()
plot_save_path = os.path.join(plot_dir, "birdnet_detection_eda_plots.png")
try:
    plt.savefig(plot_save_path)
    print(f"Saved EDA plots to: {plot_save_path}")
except Exception as e:
    print(f"Error saving plots: {e}")
plt.close(fig)

print("\nEDA script finished.")
