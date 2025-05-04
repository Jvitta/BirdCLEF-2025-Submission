import os
import pandas as pd
import numpy as np
import time
import multiprocessing
from functools import partial
import traceback
import sys
from tqdm.auto import tqdm
import logging 
import io
import contextlib


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
logging.getLogger('birdnetlib').setLevel(logging.WARNING)

from config import config

# --- Configuration ---
NUM_WORKERS = config.num_workers
CONFIDENCE_THRESHOLD = config.birdnet_confidence_threshold
OUTPUT_PATH = config.BIRDNET_DETECTIONS_NPZ_PATH
# Hardcode the single species not covered based on previous EDA
UNCOVERED_AVES_SCIENTIFIC_NAME = 'Chrysuronia goudoti'

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def load_metadata(config):
    """Loads taxonomy and training metadata."""
    print("Loading metadata...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        train_df = pd.read_csv(config.train_csv_path)

        # Merge only the class_name from taxonomy_df
        train_df = pd.merge(
            train_df,
            taxonomy_df[['primary_label', 'class_name']], # Only select necessary columns from taxonomy
            on='primary_label',
            how='left' # Keep all rows from train_df
        )

        if train_df['class_name'].isnull().any():
             print("Warning: Some training files couldn't be matched with taxonomy to get class_name.")

        # Check for nulls in scientific_name remains valid as it was originally in train_df
        if train_df['scientific_name'].isnull().any():
             print("Warning: 'scientific_name' column contains null values in train_df.")
        
        # Also check for lat/lon columns needed later
        if 'latitude' not in train_df.columns or 'longitude' not in train_df.columns:
            print("Error: Missing latitude or longitude column in train_df. Exiting.")
            sys.exit(1)

        print(f"Loaded and merged metadata for {len(train_df)} training file entries.")
        return train_df, taxonomy_df
    except FileNotFoundError as e:
        print(f"Error: Metadata file not found: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading metadata: {e}. Exiting.")
        sys.exit(1)

def process_file_birdnet(args):
    """Worker function to analyze a single audio file with BirdNET."""
    # Unpack arguments including lat/lon
    filepath, filename, primary_label, scientific_name, class_name, latitude, longitude, config = args
    
    # Skip analysis if not an Aves class or if it's the specific uncovered species
    if class_name != 'Aves' or scientific_name == UNCOVERED_AVES_SCIENTIFIC_NAME:
        return filename, [] # Return filename and empty list

    analyzer = None # Initialize outside try
    recording = None # Initialize outside try
    filtered_detections = [] # Initialize outside try

    # --- Validate Lat/Lon --- 
    # Use None if lat/lon are not valid numbers (e.g., NaN, None)
    lat_val = latitude if pd.notna(latitude) and isinstance(latitude, (int, float)) else None
    lon_val = longitude if pd.notna(longitude) and isinstance(longitude, (int, float)) else None
    # Optional: Add range checks if needed (e.g., lat -90 to 90, lon -180 to 180)
    if lat_val is not None and (lat_val < -90 or lat_val > 90):
        lat_val = None # Invalidate if out of range
    if lon_val is not None and (lon_val < -180 or lon_val > 180):
        lon_val = None # Invalidate if out of range
    # --- End Validation --- 
    
    try:
        # --- Suppress stdout/stderr during noisy operations --- 
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # Initialize analyzer within the worker process 
            analyzer = Analyzer() # Re-initialize if not thread-safe or easily passable

            recording = Recording(
                analyzer=analyzer,
                path=filepath,
                min_conf=CONFIDENCE_THRESHOLD,
                lat=lat_val, 
                lon=lon_val,
            )
            # Analyze within the suppressed block as well if it's noisy
            recording.analyze()
        # --- End suppression --- 
            
        detections = recording.detections
        
        # Filter detections: Match primary label (scientific name) and confidence
        for det in detections:
            # Check if the detected scientific name matches the file's target scientific name
            detected_scientific = det.get('scientific_name', det.get('common_name', '')) 
            
            if detected_scientific == scientific_name and det.get('confidence', 0) >= CONFIDENCE_THRESHOLD :
                 filtered_detections.append({
                     'start_time': det.get('start_time'),
                     'end_time': det.get('end_time'),
                     'confidence': det.get('confidence')
                 })

        return filename, filtered_detections

    except Exception as e:
        # Log the actual error encountered, even if stdout was suppressed earlier
        print(f"ERROR processing {filename}: {e}") 
        # print(traceback.format_exc()) # Uncomment for detailed traceback during debugging
        return filename, [] # Return empty list on error

def main(config):
    """Main function to run BirdNET analysis and save detections."""
    print("--- Starting BirdNET Detection Preprocessing ---")
    start_time = time.time()

    train_df, taxonomy_df = load_metadata(config)

    # --- Prepare Tasks for Multiprocessing ---
    tasks = []
    print("Preparing tasks for multiprocessing...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing Tasks"):
        filepath = os.path.join(config.train_audio_dir, row['filename'])
        if not os.path.exists(filepath):
            # print(f"Warning: Audio file not found for {row['filename']}, skipping.")
            continue
            
        # Pass necessary arguments to the worker, including lat/lon
        task_args = (
            filepath,
            row['filename'], # Pass filename as key
            row['primary_label'],
            row['scientific_name'],
            row['class_name'],
            row['latitude'],
            row['longitude'],
            config
        )
        tasks.append(task_args)
    
    print(f"Prepared {len(tasks)} tasks.")
    if not tasks:
        print("No valid tasks found. Exiting.")
        return

    # --- Run Analysis using Multiprocessing ---
    all_detections = {}
    error_count = 0
    processed_count = 0

    print(f"Starting BirdNET analysis with {NUM_WORKERS} workers...")
    # Initialize Analyzer once if safe, otherwise handle in worker
    # analyzer = Analyzer() # Consider if Analyzer() is expensive or needs specific init

    # Use try-finally to ensure pool closure even if errors occur inside
    pool = None # Define pool outside try block
    try:
        pool = multiprocessing.Pool(processes=NUM_WORKERS)
        results_iterator = pool.imap_unordered(process_file_birdnet, tasks)

        for i, result in enumerate(tqdm(results_iterator, total=len(tasks), desc="Analyzing Files")):
            filename, detection_list = result
            if detection_list is None: # Check if worker explicitly returned None on failure
                print(f"Warning: Worker returned None for file {filename} (likely error).")
                error_count += 1
            else:
                all_detections[filename] = detection_list
                processed_count += 1
                # Log progress periodically (optional)
                # if i % 500 == 0 and i > 0:
                #      print(f"  Processed {processed_count}/{len(tasks)} files...")

    except Exception as e:
        print(f"\nCRITICAL ERROR during multiprocessing setup or result iteration: {e}")
        print(traceback.format_exc())
        # Decide whether to proceed with saving partial results or exit
        print("Attempting to save any partial results collected so far...")
    finally:
        if pool is not None:
             print("Shutting down worker pool...")
             pool.close() # Prevent new tasks
             pool.join()  # Wait for workers to finish
             print("Worker pool shut down.")


    print(f"\nAnalysis finished. Processed {processed_count} files successfully.")
    if error_count > 0:
        print(f"Encountered errors during processing for {error_count} files (returned empty list). Check logs above.")
    
    # --- Save Results ---
    if all_detections:
        print(f"Saving {len(all_detections)} detection results to: {OUTPUT_PATH}")
        try:
            # Convert lists to object arrays for saving compatibility if needed
            save_dict = {k: np.array(v, dtype=object) if isinstance(v, list) else v for k, v in all_detections.items()}
            np.savez_compressed(OUTPUT_PATH, **save_dict)
            print("Successfully saved detection results.")
        except Exception as e:
            print(f"CRITICAL ERROR saving NPZ file: {e}")
            print(traceback.format_exc())
    else:
        print("No detections were successfully processed or generated. Nothing to save.")

    end_time = time.time()
    print(f"--- BirdNET Detection Preprocessing finished in {end_time - start_time:.2f} seconds --- ")


if __name__ == "__main__":
    main(config)
