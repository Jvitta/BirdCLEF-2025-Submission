import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

def calculate_chunk_interval(detection, target_duration):
    """Calculates the target 5s chunk interval based on detection center."""
    if not isinstance(detection, dict) or 'start_time' not in detection or 'end_time' not in detection:
        return None, None
    
    start_time = detection.get('start_time', 0)
    end_time = detection.get('end_time', 0)
    confidence = detection.get('confidence', 0)
    
    center_sec = (start_time + end_time) / 2.0
    chunk_start = center_sec - (target_duration / 2.0)
    chunk_end = center_sec + (target_duration / 2.0)
    
    # We don't need to clamp here as we just want to see the *intended* center
    return chunk_start, chunk_end, confidence

def main():
    print("--- EDA: BirdNET Chunk Time Distribution ---")
    print(f"Loading BirdNET detections from: {config.BIRDNET_DETECTIONS_NPZ_PATH}")
    try:
        with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
            # The items() method provides filename -> detection_array pairs
            all_birdnet_detections = dict(data.items()) 
        print(f"Loaded detections for {len(all_birdnet_detections)} files.")
    except FileNotFoundError:
        print(f"Error: BirdNET detections file not found at {config.BIRDNET_DETECTIONS_NPZ_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading BirdNET detections NPZ: {e}. Exiting.")
        return

    print(f"Loading taxonomy data from: {config.taxonomy_path}")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        if 'primary_label' not in taxonomy_df.columns or 'class_name' not in taxonomy_df.columns:
             raise ValueError("Taxonomy missing 'primary_label' or 'class_name'")
        print(f"Loaded taxonomy for {taxonomy_df['primary_label'].nunique()} species.")
    except Exception as e:
        print(f"Error loading taxonomy: {e}. Exiting.")
        return
        
    print(f"Loading training metadata from: {config.train_csv_path}")
    try:
        train_df = pd.read_csv(config.train_csv_path)
        if 'filename' not in train_df.columns or 'primary_label' not in train_df.columns:
             raise ValueError("Train metadata missing 'filename' or 'primary_label'")
        print(f"Loaded metadata for {train_df.shape[0]} files.")
    except Exception as e:
        print(f"Error loading training metadata: {e}. Exiting.")
        return

    # Merge to get class_name for each file
    merged_df = pd.merge(train_df[['filename', 'primary_label']], 
                         taxonomy_df[['primary_label', 'class_name']], 
                         on='primary_label', how='left')
    
    aves_files = merged_df[merged_df['class_name'] == 'Aves']['filename'].tolist()
    print(f"Identified {len(aves_files)} files belonging to class 'Aves'.")

    chunk_data = []
    n_versions = config.PRECOMPUTE_VERSIONS # Typically 3

    print(f"Processing {len(aves_files)} Aves files to extract top {n_versions} chunk intervals...")
    for filename in tqdm(aves_files, desc="Processing Aves Files"):
        if filename in all_birdnet_detections:
            detections_for_file = all_birdnet_detections[filename]
            
            # Ensure it's a list/array of dicts and filter invalid entries
            valid_detections = [d for d in detections_for_file 
                                if isinstance(d, dict) and 'confidence' in d]

            if not valid_detections:
                continue # Skip if no valid detections for this file

            # Sort by confidence descending
            try:
                 sorted_detections = sorted(valid_detections, key=lambda x: x.get('confidence', 0), reverse=True)
            except Exception as e_sort:
                 # print(f"Warning: Could not sort detections for {filename}: {e_sort}") # Optional warning
                 continue # Skip if sorting fails

            # Get top N detections
            top_n_detections = sorted_detections[:n_versions]

            file_chunk_starts = []
            for i, det in enumerate(top_n_detections):
                chunk_start, chunk_end, confidence = calculate_chunk_interval(det, config.TARGET_DURATION)
                if chunk_start is not None:
                    chunk_data.append({
                        'filename': filename,
                        'detection_rank': i + 1, # 1st, 2nd, 3rd most confident
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                        'confidence': confidence,
                        'det_start': det.get('start_time'),
                        'det_end': det.get('end_time')
                    })
                    file_chunk_starts.append(chunk_start)
            
            # Analyze proximity within the file (if more than 1 chunk)
            if len(file_chunk_starts) > 1:
                 file_chunk_starts.sort()
                 for j in range(len(file_chunk_starts) - 1):
                      diff = file_chunk_starts[j+1] - file_chunk_starts[j]
                      # Find corresponding entry to add the diff
                      for entry in chunk_data:
                           if entry['filename'] == filename and entry['chunk_start'] == file_chunk_starts[j+1]:
                                entry['start_diff_from_prev'] = diff
                                break


    if not chunk_data:
        print("No valid chunk intervals found for Aves files with BirdNET detections.")
        return

    chunk_df = pd.DataFrame(chunk_data)
    print(f"Collected data for {chunk_df.shape[0]} chunk intervals from {chunk_df['filename'].nunique()} Aves files.")

    # --- Visualization ---
    
    # 1. Distribution of Chunk Start Times
    plt.figure(figsize=(12, 6))
    sns.histplot(chunk_df['chunk_start'], bins=50, kde=True)
    plt.title(f'Distribution of Calculated Chunk Start Times (Top {n_versions} Detections per Aves File)')
    plt.xlabel('Chunk Start Time (seconds from audio start)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'eda_chunk_start_distribution.png'))
    print("Saved plot: eda_chunk_start_distribution.png")
    plt.close()

    # 2. Relationship between Detection Rank and Start Time
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=chunk_df, x='detection_rank', y='chunk_start')
    plt.title('Chunk Start Time vs. Detection Confidence Rank')
    plt.xlabel('Detection Rank (1 = Highest Confidence)')
    plt.ylabel('Chunk Start Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'eda_chunk_start_vs_rank.png'))
    print("Saved plot: eda_chunk_start_vs_rank.png")
    plt.close()

    # 3. Distribution of Start Time Differences (Proximity Analysis)
    # Filter out NaNs which occur for the first chunk of each file
    diff_data = chunk_df['start_diff_from_prev'].dropna()
    if not diff_data.empty:
        plt.figure(figsize=(12, 6))
        # Cap difference for visualization if needed (e.g., outliers)
        max_diff_to_plot = 60 # Show differences up to 60 seconds
        sns.histplot(diff_data[diff_data < max_diff_to_plot], bins=50, kde=False)
        plt.title(f'Distribution of Time Difference Between Consecutive Chunk Starts (Within Same File)')
        plt.xlabel('Time Difference (seconds)')
        plt.ylabel('Frequency')
        plt.xlim(left=0) # Difference should be non-negative
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, 'eda_chunk_start_difference_distribution.png'))
        print("Saved plot: eda_chunk_start_difference_distribution.png")
        plt.close()
        
        # Print some stats about the differences
        print("--- Chunk Proximity Analysis (Time difference between consecutive chunk starts within a file) ---")
        print(f"Mean difference: {diff_data.mean():.2f} seconds")
        print(f"Median difference: {diff_data.median():.2f} seconds")
        print(f"Min difference: {diff_data.min():.2f} seconds")
        print(f"Max difference: {diff_data.max():.2f} seconds")
        overlap_threshold = config.TARGET_DURATION # Difference less than chunk duration suggests overlap
        overlap_percentage = (diff_data < overlap_threshold).mean() * 100
        print(f"Percentage of consecutive chunks starting < {overlap_threshold}s apart (suggesting overlap): {overlap_percentage:.2f}%")
        very_close_threshold = 1.0 # Difference less than 1s
        very_close_percentage = (diff_data < very_close_threshold).mean() * 100
        print(f"Percentage of consecutive chunks starting < {very_close_threshold}s apart: {very_close_percentage:.2f}%")
    else:
        print("Could not perform proximity analysis (likely only one chunk per file on average).")

    print("EDA script finished.")


if __name__ == '__main__':
    main()
