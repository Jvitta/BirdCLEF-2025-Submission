import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import multiprocessing
import functools
import io # Added for redirecting stdout/stderr

# Try importing birdnetlib and handle potential import errors
try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
except ImportError:
    print("Error: birdnetlib not found. Please install it: pip install birdnetlib")
    sys.exit(1)

# Ensure project root is in path to import config
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import config

# Suppress verbose TensorFlow logging and birdnetlib info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('birdnetlib').setLevel(logging.WARNING)

warnings.filterwarnings("ignore")

# --- Worker Function for Multiprocessing ---
def process_audio_file_worker(audio_path, cfg, competition_sci_names, sci_to_prim_label):
    """Processes a single audio file using BirdNET.
    
    Args:
        audio_path (Path): Path to the audio file.
        cfg (Config): Configuration object.
        competition_sci_names (set): Set of scientific names in the competition.
        sci_to_prim_label (dict): Mapping from scientific name to primary label.

    Returns:
        tuple: (list of detection dicts, str or None for error message)
    """
    file_detections = []
    analyzer_instance = None
    
    # Redirect stdout/stderr to suppress verbose library output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO() # Capture stdout
    sys.stderr = io.StringIO() # Capture stderr
    
    captured_stdout = ""
    captured_stderr = ""

    try:
        analyzer_instance = Analyzer()
        # Date parsing and Recording object creation
        filename_full = audio_path.name
        date_obj = None
        try:
            name_parts = filename_full.split('_')
            if len(name_parts) >= 2:
                date_str = name_parts[1]
                if len(date_str) == 8 and date_str.isdigit():
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                # else: # Supressing warnings for cleaner tqdm output during multiprocessing
                #     print(f"\nWarning: Date part '{date_str}' from filename {filename_full} not in YYYYMMDD format. Proceeding without date.")
            # else:
            #     print(f"\nWarning: Filename {filename_full} does not match expected SITE_DATE_TIME format. Proceeding without date.")
        except Exception: # Catches any date parsing error
            # print(f"\nWarning: Could not parse date from filename {filename_full}: {e_date}. Proceeding without date for this file.")
            pass # Proceed without date if parsing fails

        recording = Recording(
            analyzer=analyzer_instance,
            path=audio_path,
            lat=6.76,
            lon=-74.21,
            date=date_obj,
            min_conf=cfg.BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD
        )
        recording.analyze() # This is likely where most prints happen
        
        # Restore stdout/stderr immediately after critical section
        captured_stdout = sys.stdout.getvalue()
        captured_stderr = sys.stderr.getvalue()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        for det in recording.detections:
            sci_name = det.get('scientific_name')
            if sci_name in competition_sci_names:
                primary_label = sci_to_prim_label[sci_name]
                file_detections.append({
                    'filename': filename_full,
                    'start_time': det.get('start_time'),
                    'end_time': det.get('end_time'),
                    'primary_label': primary_label,
                    'confidence': det.get('confidence')
                })
        return file_detections, None
    except Exception as e:
        # Ensure stdout/stderr are restored even if an error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Optionally log captured_stdout/stderr here if e is critical
        # print(f"Captured stdout during error for {audio_path.name}:\n{captured_stdout}")
        # print(f"Captured stderr during error for {audio_path.name}:\n{captured_stderr}")
        return [], f"Error in worker for file {audio_path.name}: {e}"
    finally:
        # Ensure stdout/stderr are restored in all cases
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if analyzer_instance:
            del analyzer_instance


def generate_labels(config_obj): # Renamed config to config_obj to avoid conflict
    """Generates pseudo-labels using BirdNET Analyzer for Aves species."""
    print("--- Generating Pseudo-Labels using BirdNET (Multiprocessed) ---")
    print(f"Unlabeled Audio Directory: {config_obj.unlabeled_audio_dir}")
    print(f"Taxonomy Path: {config_obj.taxonomy_path}")
    print(f"Output CSV Path: {config_obj.train_pseudo_csv_path}")
    print(f"Confidence Threshold: {config_obj.BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD}")
    print(f"Using {config_obj.num_workers} worker processes.")

    # 1. Load Taxonomy and create mapping (remains the same)
    try:
        taxonomy_df = pd.read_csv(config_obj.taxonomy_path)
        competition_scientific_names = set(taxonomy_df['scientific_name'].tolist())
        sci_to_primary = dict(zip(taxonomy_df['scientific_name'], taxonomy_df['primary_label']))
        print(f"Loaded taxonomy for {len(competition_scientific_names)} species.")
    except FileNotFoundError:
        print(f"Error: Taxonomy file not found at {config_obj.taxonomy_path}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading taxonomy: {e}. Exiting.")
        return

    # 2. BirdNET Analyzer will be initialized IN EACH WORKER
    # No global analyzer instance created here anymore.
    print("BirdNET Analyzer will be initialized in each worker process.")

    # 3. Get list of audio files (remains the same)
    audio_files = list(Path(config_obj.unlabeled_audio_dir).glob('*.ogg'))
    if config_obj.debug:
        print(f"Debug mode: Limiting to {config_obj.debug_limit_files} audio files.")
        audio_files = audio_files[:config_obj.debug_limit_files]

    if not audio_files:
        print(f"Error: No OGG audio files found in {config_obj.unlabeled_audio_dir}. Exiting.")
        return
    print(f"Found {len(audio_files)} audio files to process.")

    # 4. Process files and generate labels using multiprocessing
    all_pseudo_labels = []
    files_with_errors_count = 0

    worker_fn = functools.partial(process_audio_file_worker,
                                  # analyzer_instance is removed from partial
                                  cfg=config_obj,
                                  competition_sci_names=competition_scientific_names,
                                  sci_to_prim_label=sci_to_primary)

    with multiprocessing.Pool(processes=config_obj.num_workers) as pool:
        # Using imap_unordered for potentially better tqdm updates with varying task times
        # tqdm will show progress as tasks are submitted, not necessarily as they complete out of order
        results_iterator = pool.imap_unordered(worker_fn, audio_files)
        
        for result_detections, error_message in tqdm(results_iterator, total=len(audio_files), desc="Analyzing Audio Files"):
            if error_message:
                print(f"\n{error_message}") # Print error message from worker
                files_with_errors_count += 1
            if result_detections: # Could be empty list if file had no target species or on error
                all_pseudo_labels.extend(result_detections)

    if files_with_errors_count > 0:
        print(f"\nFinished processing, encountered errors in {files_with_errors_count} files.")

    # 5. Create and Save DataFrame (remains largely the same)
    if not all_pseudo_labels:
        print("Warning: No pseudo-labels generated above the threshold or all files had errors.")
        pseudo_labels_df = pd.DataFrame(columns=['filename', 'start_time', 'end_time', 'primary_label', 'confidence'])
    else:
        pseudo_labels_df = pd.DataFrame(all_pseudo_labels)

    print(f"Generated {len(pseudo_labels_df)} pseudo-labels.")

    try:
        output_dir = Path(config_obj.train_pseudo_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_df.to_csv(config_obj.train_pseudo_csv_path, index=False)
        print(f"Pseudo-labels saved to: {config_obj.train_pseudo_csv_path}")
    except Exception as e:
        print(f"Error saving pseudo-labels CSV: {e}")

if __name__ == "__main__":
    # It's good practice for multiprocessing scripts to protect the main execution
    # especially on Windows or when 'spawn'/'forkserver' start methods are used.
    # No specific changes needed here if using 'fork' (default on Linux).
    generate_labels(config) 