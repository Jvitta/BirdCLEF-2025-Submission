import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
from tqdm.auto import tqdm

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow INFO, WARNING, ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('birdnetlib').setLevel(logging.WARNING) # Show only warnings and errors

warnings.filterwarnings("ignore")


def generate_labels(config):
    """Generates pseudo-labels using BirdNET Analyzer for Aves species."""
    print("--- Generating Pseudo-Labels using BirdNET ---")
    print(f"Unlabeled Audio Directory: {config.unlabeled_audio_dir}")
    print(f"Taxonomy Path: {config.taxonomy_path}")
    print(f"Output CSV Path: {config.train_pseudo_csv_path}")
    print(f"Confidence Threshold: {config.threshold}") # Using existing threshold for now

    # 1. Load Taxonomy and create mapping
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        # Create a set of scientific names for efficient lookup
        competition_scientific_names = set(taxonomy_df['scientific_name'].tolist())
        # Create a mapping from scientific name back to primary_label
        sci_to_primary = dict(zip(taxonomy_df['scientific_name'], taxonomy_df['primary_label']))
        print(f"Loaded taxonomy for {len(competition_scientific_names)} species.")
    except FileNotFoundError:
        print(f"Error: Taxonomy file not found at {config.taxonomy_path}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading taxonomy: {e}. Exiting.")
        return

    # 2. Initialize BirdNET Analyzer
    try:
        # Provide general coordinates for the El Silencio reserve
        analyzer = Analyzer(
            lat=6.76, 
            lon=-74.21
            # week_48=-1, # Default: use current week, -1 might disable week filtering if desired
            # min_conf=config.threshold # Set min_conf here OR filter later
        )
        print("BirdNET Analyzer initialized with location lat=6.76, lon=-74.21.")
    except Exception as e:
        print(f"Error initializing BirdNET Analyzer: {e}. Exiting.")
        return

    # 3. Get list of audio files
    audio_files = list(Path(config.unlabeled_audio_dir).glob('*.ogg'))
    if config.debug:
        print(f"Debug mode: Limiting to {config.debug_limit_files} audio files.")
        audio_files = audio_files[:config.debug_limit_files]

    if not audio_files:
        print(f"Error: No OGG audio files found in {config.unlabeled_audio_dir}. Exiting.")
        return
    print(f"Found {len(audio_files)} audio files to process.")

    # 4. Process files and generate labels
    pseudo_labels_list = []
    files_with_errors = 0

    for audio_path in tqdm(audio_files, desc="Analyzing Audio Files"):
        try:
            recording = Recording(
                analyzer=analyzer,
                path=audio_path,
                min_conf=config.threshold # Use threshold as min_conf for Analyzer
            )
            recording.analyze()
            detections = recording.detections

            filename_stem = audio_path.stem # Get filename without extension
            filename_full = audio_path.name # Get full filename with extension

            for det in detections:
                # Match based on scientific name
                sci_name = det.get('scientific_name')
                if sci_name in competition_scientific_names:
                    primary_label = sci_to_primary[sci_name]
                    pseudo_labels_list.append({
                        'filename': filename_full, # Use full filename with extension
                        'start_time': det.get('start_time'),
                        'end_time': det.get('end_time'),
                        'primary_label': primary_label,
                        'confidence': det.get('confidence')
                    })

        except Exception as e:
            print(f"\nWarning: Error processing file {audio_path.name}: {e}")
            files_with_errors += 1
            # Continue to the next file

    if files_with_errors > 0:
        print(f"\nFinished processing, encountered errors in {files_with_errors} files.")

    # 5. Create and Save DataFrame
    if not pseudo_labels_list:
        print("Warning: No pseudo-labels generated above the threshold.")
        # Save empty DataFrame with correct columns if needed
        pseudo_labels_df = pd.DataFrame(columns=['filename', 'start_time', 'end_time', 'primary_label', 'confidence'])
    else:
        pseudo_labels_df = pd.DataFrame(pseudo_labels_list)
        # Optional: Sort or further process the dataframe here if needed
        # pseudo_labels_df.sort_values(by=['filename', 'start_time'], inplace=True)

    print(f"Generated {len(pseudo_labels_df)} pseudo-labels.")

    try:
        output_dir = Path(config.train_pseudo_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_df.to_csv(config.train_pseudo_csv_path, index=False)
        print(f"Pseudo-labels saved to: {config.train_pseudo_csv_path}")
    except Exception as e:
        print(f"Error saving pseudo-labels CSV: {e}")


if __name__ == "__main__":
    generate_labels(config) 