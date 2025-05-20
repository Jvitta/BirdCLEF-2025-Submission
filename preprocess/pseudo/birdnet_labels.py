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
project_root = current_dir.parent # This should be src/
# To get to the main project root (BirdCLEF-2025-Submission), we go up one more level from src/
project_root = project_root.parent 
sys.path.append(str(project_root))

from config import config

# Suppress verbose TensorFlow logging and birdnetlib info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Let's keep birdnetlib warnings for now, as they might be informative with new settings
# logging.getLogger('birdnetlib').setLevel(logging.WARNING) 

warnings.filterwarnings("ignore", category=UserWarning, module='librosa') # Filter specific librosa warnings if needed


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
    original_stdout = sys.stdout 
    original_stderr = sys.stderr 
    captured_stdout_io = io.StringIO()
    captured_stderr_io = io.StringIO()
    worker_error_message = None

    try:
        sys.stdout = captured_stdout_io
        sys.stderr = captured_stderr_io

        # detection_overlap_seconds = getattr(cfg, 'BIRDNET_PSEUDO_OVERLAP_SECONDS', 2.0) 
        # User hardcoded this in the file based on the diff, so respecting that for now.
        detection_overlap_seconds = 2.0
        
        # Using the BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD from config as per user's last direct edit indication
        initial_min_conf = cfg.BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD 

        analyzer_instance = Analyzer()
        filename_full = audio_path.name
        date_obj = None
        try:
            name_parts = filename_full.split('_')
            if len(name_parts) >= 2:
                date_str = name_parts[1]
                if len(date_str) == 8 and date_str.isdigit():
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
        except Exception:
            pass 

        recording = Recording(
            analyzer=analyzer_instance,
            path=audio_path,
            lat=6.76, 
            lon=-74.21,
            date=date_obj,
            min_conf=initial_min_conf, 
            overlap=detection_overlap_seconds
        )
        recording.analyze() 
        
        # Restore stdout/stderr now that birdnetlib processing is done for this file
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
                    # 'scientific_name': sci_name, # User removed this in their last edit, respecting that.
                    'confidence': det.get('confidence')
                })
    except Exception as e:
        # Ensure stdout/stderr are restored before attempting to get value or format error message
        sys.stdout = original_stdout 
        sys.stderr = original_stderr
        
        # Get captured stderr content
        # captured_stdout_val = captured_stdout_io.getvalue() # For debugging if needed
        captured_stderr_val = captured_stderr_io.getvalue()
        
        error_detail = f"Error in worker for file {audio_path.name}: {str(e)}"
        if captured_stderr_val and captured_stderr_val.strip():
            error_detail += f"\n  Captured Stderr:\n{captured_stderr_val.strip()}"
        # if captured_stdout_val and captured_stdout_val.strip(): # Optional: include stdout for deeper debug
        #     error_detail += f"\n  Captured Stdout:\n{captured_stdout_val.strip()}"
        worker_error_message = error_detail
        
    finally:
        # Final safety net for restoring stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if analyzer_instance:
            del analyzer_instance
        # Close StringIO objects
        captured_stdout_io.close()
        captured_stderr_io.close()

    return file_detections, worker_error_message


def generate_labels(config_obj):
    """Generates pseudo-labels using BirdNET Analyzer for Aves species."""
    print("--- Generating Pseudo-Labels using BirdNET (Multiprocessed) ---")
    print(f"Unlabeled Audio Directory: {config_obj.unlabeled_audio_dir}")
    print(f"Taxonomy Path: {config_obj.taxonomy_path}")
    print(f"Output CSV Path: {config_obj.train_pseudo_csv_path}")
    print(f"Using BirdNET Confidence Threshold from config: {config_obj.BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD}")
    
    # Reflecting user's hardcoded value in the worker, but still good to print what's expected.
    # initial_min_conf = getattr(config_obj, 'BIRDNET_PSEUDO_INITIAL_MIN_CONF', 0.05)
    # top_k_per_species = getattr(config_obj, 'BIRDNET_PSEUDO_TOP_K_PER_SPECIES', 500)
    # print(f"BirdNET Initial Min Confidence for detection: {initial_min_conf}") # This was for TopK logic
    # print(f"Post-filtering to Top K detections per species: {top_k_per_species}") # This was for TopK logic
    detection_overlap_seconds = 2.0
    analysis_step_seconds = 3.0 - detection_overlap_seconds
    print(f"BirdNET Analysis Window: 3.0s, Overlap: {detection_overlap_seconds:.1f}s (Effective Step: {analysis_step_seconds:.1f}s)")
    
    print(f"Using {config_obj.num_workers} worker processes.")

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

    print("BirdNET Analyzer will be initialized in each worker process.")

    audio_files = list(Path(config_obj.unlabeled_audio_dir).glob('*.ogg'))
    if config_obj.debug:
        print(f"Debug mode: Limiting to {config_obj.debug_limit_files} audio files.")
        audio_files = audio_files[:config_obj.debug_limit_files]

    if not audio_files:
        print(f"Error: No OGG audio files found in {config_obj.unlabeled_audio_dir}. Exiting.")
        return
    print(f"Found {len(audio_files)} audio files to process.")

    all_pseudo_labels = [] # Changed from all_pseudo_labels_raw, as TopK is removed
    files_with_errors_count = 0

    worker_fn = functools.partial(process_audio_file_worker,
                                  cfg=config_obj, 
                                  competition_sci_names=competition_scientific_names,
                                  sci_to_prim_label=sci_to_primary)

    with multiprocessing.Pool(processes=config_obj.num_workers) as pool:
        results_iterator = pool.imap_unordered(worker_fn, audio_files)
        for result_detections, error_message in tqdm(results_iterator, total=len(audio_files), desc="Analyzing Audio Files"):
            if error_message:
                tqdm.write(f"{error_message}") 
                files_with_errors_count += 1
            if result_detections: 
                all_pseudo_labels.extend(result_detections)

    if files_with_errors_count > 0:
        print(f"Finished pass, encountered errors in {files_with_errors_count} files.")

    if not all_pseudo_labels: # Check all_pseudo_labels instead of all_pseudo_labels_raw
        print("Warning: No pseudo-labels generated from the pass.")
        final_pseudo_labels_df = pd.DataFrame(columns=['filename', 'start_time', 'end_time', 'primary_label', 'confidence'])
    else:
        final_pseudo_labels_df = pd.DataFrame(all_pseudo_labels) # Use all_pseudo_labels
        print(f"Generated {len(final_pseudo_labels_df)} pseudo-labels (min_conf: {config_obj.BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD}).")
        
    try:
        output_dir = Path(config_obj.train_pseudo_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        final_pseudo_labels_df.to_csv(config_obj.train_pseudo_csv_path, index=False)
        print(f"Pseudo-labels saved to: {config_obj.train_pseudo_csv_path}")
    except Exception as e:
        print(f"Error saving pseudo-labels CSV: {e}")

if __name__ == "__main__":
    generate_labels(config) 