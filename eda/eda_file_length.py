import os
import sys
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import time
import warnings

# Ensure project root is in path to import config
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import config

# Ignore librosa warnings (e.g., audioread backend)
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Hardcode the excluded species name locally (mirroring preprocessing)
UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'

def get_audio_duration(filepath):
    """Safely gets the duration of an audio file."""
    try:
        return librosa.get_duration(path=filepath)
    except FileNotFoundError:
        # print(f"Warning: File not found: {filepath}")
        return None
    except Exception as e:
        # print(f"Warning: Error reading duration for {filepath}: {e}")
        return None

def analyze_durations(durations_list, title):
    """Helper function to analyze and print duration stats."""
    print(f"\n--- {title} ---")
    num_processed = len(durations_list)
    if num_processed == 0:
        print("No audio file durations in this category.")
        return

    durations_arr = np.array(durations_list)

    # Define bins
    bins = [0, 5, 10, 15, 30, 60, np.inf]
    bin_labels = ['< 5s', '5-10s', '10-15s', '15-30s', '30-60s', '> 60s']

    # Calculate counts and percentages
    counts, _ = np.histogram(durations_arr, bins=bins)
    percentages = (counts / num_processed) * 100

    print(f"Total files in category: {num_processed}")
    print("Duration Distribution:")
    for i, label in enumerate(bin_labels):
        print(f"  {label:<8}: {counts[i]:>7} files ({percentages[i]:>6.2f}%)")

    # Additional stats
    print(f"\nMinimum duration: {durations_arr.min():.2f}s")
    print(f"Maximum duration: {durations_arr.max():.2f}s")
    print(f"Average duration: {durations_arr.mean():.2f}s")
    print(f"Median duration:  {np.median(durations_arr):.2f}s")

    # Specific check for < 15 seconds
    count_lt_15 = counts[0] + counts[1] + counts[2]
    percent_lt_15 = percentages[0] + percentages[1] + percentages[2]
    print(f"\nFiles < 15 seconds: {count_lt_15} ({percent_lt_15:.2f}%)")

    # --- Add Total Duration --- #
    total_duration_sec = durations_arr.sum()
    total_duration_min = total_duration_sec / 60
    total_duration_hr = total_duration_min / 60
    print(f"Total Duration:   {total_duration_sec:.2f} seconds (~{total_duration_min:.2f} minutes / ~{total_duration_hr:.2f} hours)")
    # -------------------------- #

def main():
    print("--- EDA: Audio File Duration Analysis (Overall & Random Chunks) ---")
    start_time = time.time()

    # 1. Load Metadata
    print(f"Loading main metadata from: {config.train_csv_path}")
    try:
        df_train = pd.read_csv(config.train_csv_path)
        df_list = [df_train]
        print(f"Loaded {len(df_train)} main records.")
    except Exception as e:
        print(f"Error loading main metadata: {e}. Exiting.")
        return

    if config.USE_RARE_DATA:
        print(f"USE_RARE_DATA is True. Loading rare metadata from: {config.train_rare_csv_path}")
        try:
            df_rare = pd.read_csv(config.train_rare_csv_path)
            df_list.append(df_rare)
            print(f"Loaded {len(df_rare)} rare records.")
        except Exception as e:
            print(f"Warning: Could not load rare metadata: {e}. Proceeding with main data only.")
    else:
        print("USE_RARE_DATA is False.")

    df_combined = pd.concat(df_list, ignore_index=True)
    required_cols_meta = {'filename', 'primary_label'}
    if not required_cols_meta.issubset(df_combined.columns):
        print(f"Error: Metadata missing required columns: {required_cols_meta - set(df_combined.columns)}. Exiting.")
        return

    # 2. Load Taxonomy
    print("\nLoading taxonomy data...")
    taxonomy_df = pd.DataFrame() # Initialize empty
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        required_taxonomy_cols = {'primary_label', 'class_name', 'scientific_name'}
        if not required_taxonomy_cols.issubset(taxonomy_df.columns):
            print(f"Error: Taxonomy file missing required columns: {required_taxonomy_cols - set(taxonomy_df.columns)}")
            taxonomy_df = pd.DataFrame() # Reset on error
    except Exception as e:
        print(f"Error loading taxonomy file: {e}")

    # Merge class_name and scientific_name
    if not taxonomy_df.empty:
        df_merged = pd.merge(
            df_combined,
            taxonomy_df[['primary_label', 'class_name', 'scientific_name']],
            on='primary_label',
            how='left'
        )
        print("Merged taxonomy data.")
    else:
        print("Warning: Could not merge taxonomy data. Class/Scientific name info unavailable.")
        df_merged = df_combined.copy()
        # Add placeholder columns if they don't exist
        if 'class_name' not in df_merged.columns: df_merged['class_name'] = None
        if 'scientific_name' not in df_merged.columns: df_merged['scientific_name'] = None

    # Create a mapping from filename to taxonomy info for quick lookup
    file_info_map = {}
    for _, row in df_merged.iterrows():
        # Store the first occurrence's info if filenames are duplicated across train/rare
        if row['filename'] not in file_info_map:
             file_info_map[row['filename']] = {
                 'class_name': row.get('class_name', None),
                 'scientific_name': row.get('scientific_name', None)
             }

    # 3. Load BirdNET Detections
    print(f"\nAttempting to load BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    all_birdnet_detections = {}
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            # Only keep keys where the value is a non-empty list/array of dicts (basic check)
            all_birdnet_detections = {key: data[key] for key in data.files if isinstance(data[key], (np.ndarray, list)) and len(data[key]) > 0 and isinstance(data[key][0], dict)}
        print(f"Successfully loaded valid BirdNET detections for {len(all_birdnet_detections)} files.")
    except FileNotFoundError:
        print(f"Warning: BirdNET detections file not found. Assuming random chunks for all Aves.")
    except Exception as e:
        print(f"Warning: Error loading BirdNET detections NPZ: {e}. Assuming random chunks for all Aves.")

    # 4. Get Durations and Categorize Files
    print("\nProcessing file durations...")
    filenames = df_merged['filename'].unique()
    total_files = len(filenames)
    print(f"Processing {total_files} unique filenames.")

    all_durations = []
    random_chunk_durations = []
    files_not_found = 0
    files_error = 0

    for filename in tqdm(filenames, desc="Getting durations"):
        # Find audio file path
        main_path = Path(config.train_audio_dir) / filename
        rare_path = Path(config.train_audio_rare_dir) / filename if config.USE_RARE_DATA else None
        filepath_to_check = None
        if main_path.exists():
            filepath_to_check = main_path
        elif rare_path and rare_path.exists():
            filepath_to_check = rare_path

        if not filepath_to_check:
            files_not_found += 1
            continue

        duration = get_audio_duration(filepath_to_check)
        if duration is None:
            files_error += 1
            continue

        # Append to all durations list
        all_durations.append(duration)

        # Determine if this file uses random chunks
        uses_random_chunk = True # Default to random
        file_info = file_info_map.get(filename)

        if file_info:
            class_name = file_info.get('class_name')
            scientific_name = file_info.get('scientific_name')

            if class_name == 'Aves' and scientific_name != UNCOVERED_AVES_SCIENTIFIC_NAME:
                # Check if valid BirdNET detections exist for this file
                if filename in all_birdnet_detections:
                     uses_random_chunk = False # Use BirdNET guidance

        # If it uses random chunk, add duration to the specific list
        if uses_random_chunk:
            random_chunk_durations.append(duration)

    # 5. Analyze and Report Durations
    print(f"\nSuccessfully obtained durations for {len(all_durations)}/{total_files} files.")
    if files_not_found > 0:
        print(f"Files not found: {files_not_found}")
    if files_error > 0:
        print(f"Errors reading duration: {files_error}")

    # Analyze all files
    analyze_durations(all_durations, "Overall Duration Analysis")

    # Analyze files processed with random chunks
    analyze_durations(random_chunk_durations, "Duration Analysis for Files Processed with Random Chunks")

    end_time = time.time()
    print(f"\nAnalysis finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
