import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import librosa # For getting audio duration
from tqdm.auto import tqdm
import random  # For random sampling
import shutil  # For file copying

# Assuming config.py is two levels up (e.g., from eda/ -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- EDA: Duration Analysis of Files with Zero BirdNET Detections ---")

# Hardcode the single species not covered based on previous EDA
UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'

# --- Load Metadata --- 
print("Loading training metadata...")
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
        
    # Minimal merge just to get necessary columns (only class_name from taxonomy)
    train_df = pd.merge(
        train_df, 
        taxonomy_df[['primary_label', 'class_name']], # Only select class_name
        on='primary_label', 
        how='left'
    )
    
    # Check class_name after merge
    if train_df['class_name'].isnull().any():
         print("Warning: Some training files couldn't be matched with taxonomy to get class_name.")

    # Filter for eligible Aves files
    aves_df = train_df[
        (train_df['class_name'] == 'Aves') &
        (train_df['scientific_name'] != UNCOVERED_AVES_SCIENTIFIC_NAME)
    ].copy() # Make a copy to avoid SettingWithCopyWarning
    print(f"Found {len(aves_df)} eligible 'Aves' files in metadata.")
    
except FileNotFoundError as e:
    print(f"Error: Metadata file not found: {e}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading or processing metadata: {e}. Exiting.")
    sys.exit(1)

# --- Load BirdNET Detections --- 
detections_path = config.BIRDNET_DETECTIONS_NPZ_PATH
print(f"Loading BirdNET detections from: {detections_path}")
try:
    with np.load(detections_path, allow_pickle=True) as data:
        birdnet_detections = {key: data[key] for key in data.files}
    print(f"Loaded detections for {len(birdnet_detections)} files from NPZ.")
except FileNotFoundError:
    print(f"Error: Detections file not found at {detections_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading detections NPZ: {e}. Exiting.")
    sys.exit(1)

# --- Identify Files with Zero Detections --- 
print("Identifying files with zero detections...")
zero_detection_filenames = set()
for filename, detections_list in birdnet_detections.items():
    if not (hasattr(detections_list, '__len__') and len(detections_list) > 0):
        zero_detection_filenames.add(filename)
print(f"Found {len(zero_detection_filenames)} files with zero detections in NPZ.")

# --- Filter Metadata for Zero-Detection Files --- 
zero_det_df = aves_df[aves_df['filename'].isin(zero_detection_filenames)].copy()
print(f"Filtered metadata to {len(zero_det_df)} zero-detection 'Aves' files.")

if zero_det_df.empty:
    print("No 'Aves' files with zero detections found. Nothing to analyze.")
    sys.exit(0)

# --- Calculate Durations --- 
print("\nCalculating durations for zero-detection files...")
durations = []
skipped_loading = 0
skipped_path = 0

for index, row in tqdm(zero_det_df.iterrows(), total=len(zero_det_df), desc="Getting Durations"):
    filename = row['filename']
    
    # Construct potential paths
    potential_main_path = os.path.join(config.train_audio_dir, filename)
    potential_rare_path = os.path.join(config.train_audio_rare_dir, filename) if config.USE_RARE_DATA else None
    
    audio_filepath = None
    if os.path.exists(potential_main_path):
        audio_filepath = potential_main_path
    elif potential_rare_path and os.path.exists(potential_rare_path):
        audio_filepath = potential_rare_path
        
    if audio_filepath:
        try:
            # Use librosa.get_duration - more efficient than loading full audio
            duration = librosa.get_duration(path=audio_filepath)
            durations.append(duration)
        except Exception as e:
            # print(f"Warning: Could not get duration for {filename}: {e}")
            skipped_loading += 1
    else:
        # print(f"Warning: Audio path not found for zero-detection file: {filename}")
        skipped_path += 1

print(f"Finished calculating durations.")
if skipped_path > 0:
    print(f"  Warning: Skipped {skipped_path} files because audio path was not found.")
if skipped_loading > 0:
    print(f"  Warning: Skipped {skipped_loading} files due to errors getting duration.")

if not durations:
    print("Could not retrieve any durations. Exiting analysis.")
    sys.exit(0)

# --- Analyze Durations --- 
print("\n--- Duration Analysis of Zero-Detection Files ---")
duration_series = pd.Series(durations)
print(duration_series.describe())

# Check percentage below certain thresholds
threshold_3s = 3.0
threshold_1_5s = 1.5

below_3s = sum(duration_series < threshold_3s)
perc_below_3s = (below_3s / len(duration_series)) * 100 if len(duration_series) > 0 else 0

below_1_5s = sum(duration_series < threshold_1_5s)
perc_below_1_5s = (below_1_5s / len(duration_series)) * 100 if len(duration_series) > 0 else 0

print(f"\nFiles with duration < {threshold_3s:.1f}s: {below_3s} ({perc_below_3s:.2f}%)")
print(f"Files with duration < {threshold_1_5s:.1f}s: {below_1_5s} ({perc_below_1_5s:.2f}%)")

# --- Create Plot --- 
print("\nGenerating duration histogram...")
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plot_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

sns.histplot(duration_series, bins=50, kde=False)
plt.title('Distribution of Audio Durations (Files with Zero BirdNET Detections)')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Files')
# Add vertical lines for thresholds
plt.axvline(threshold_3s, color='r', linestyle='--', label=f'{threshold_3s:.1f}s Threshold')
plt.axvline(threshold_1_5s, color='g', linestyle=':', label=f'{threshold_1_5s:.1f}s Threshold')
plt.legend()
plt.grid(True)

plot_save_path = os.path.join(plot_dir, "zero_detection_duration_histogram.png")
try:
    plt.savefig(plot_save_path)
    print(f"Saved duration histogram to: {plot_save_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.close()

# --- Copy Random Sample of Zero-Detection Files --- 
SAMPLE_SIZE = 20
if not zero_det_df.empty:
    print(f"\nSelecting random sample of {min(SAMPLE_SIZE, len(zero_det_df))} zero-detection files to copy...")
    
    sample_df = zero_det_df.sample(n=min(SAMPLE_SIZE, len(zero_det_df)), random_state=config.seed)
    
    # Create the destination directory
    sample_dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_detection_samples")
    os.makedirs(sample_dest_dir, exist_ok=True)
    print(f"Copying samples to: {sample_dest_dir}")
    
    copied_count = 0
    copy_errors = 0
    for index, row in sample_df.iterrows():
        filename = row['filename']
        # Find the source path again
        potential_main_path = os.path.join(config.train_audio_dir, filename)
        potential_rare_path = os.path.join(config.train_audio_rare_dir, filename) if config.USE_RARE_DATA else None
        
        source_path = None
        if os.path.exists(potential_main_path):
            source_path = potential_main_path
        elif potential_rare_path and os.path.exists(potential_rare_path):
            source_path = potential_rare_path
            
        if source_path:
            dest_path = os.path.join(sample_dest_dir, filename)
            # --- Ensure destination subdirectory exists --- 
            dest_subdir = os.path.dirname(dest_path)
            os.makedirs(dest_subdir, exist_ok=True)
            # --- End ensure subdir ---
            try:
                shutil.copy2(source_path, dest_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {filename}: {e}")
                copy_errors += 1
        else:
            print(f"  Warning: Source path not found for {filename} during copy operation.")
            
    print(f"Finished copying. Copied {copied_count} files.")
    if copy_errors > 0:
        print(f"  Encountered {copy_errors} errors during copying.")
else:
    print("\nSkipping file sampling as no zero-detection files were found.")

print("\nDuration EDA script finished.")
