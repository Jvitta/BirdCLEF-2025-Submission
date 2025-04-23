import torch
import os
import multiprocessing
from google.cloud import storage

class Config:
    seed = 42
    debug = False  # Master debug flag for limiting epochs, batches etc.
    debug_preprocessing_mode = False # Controls N_MAX_PREPROCESS and filename suffix
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    IS_CUSTOM_JOB = os.getenv('AIP_JOB_NAME') is not None
    if IS_CUSTOM_JOB:
        print("INFO: Detected execution in Vertex AI Custom Job. Using direct GCS access.")
    else:
        print("INFO: Running in interactive mode (or non-Vertex AI job). Using gcsfuse mount paths.")

    GCS_BUCKET_NAME = "birdclef-2025-data"
    GCS_PREPROCESSED_PATH_PREFIX = "preprocessed/"
    GCS_VOICE_SEP_PATH_PREFIX = "BC25 voice separation/"

    # --- Workbench/Local Paths (using gcsfuse mount) --- #
    PROJECT_ROOT = "/home/ext_jvittimberga_gmail_com/BirdCLEF-2025-Submission"
    GCS_MOUNT_POINT = "/home/ext_jvittimberga_gmail_com/gcs_mount"

    # These paths will be used primarily when IS_CUSTOM_JOB is False
    DATA_ROOT = os.path.join(GCS_MOUNT_POINT, "raw_data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models')
    PREPROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'preprocessed') 
    MODEL_INPUT_DIR = MODEL_OUTPUT_DIR

    # These derived paths use the mount point when IS_CUSTOM_JOB is False
    train_audio_dir = os.path.join(DATA_ROOT, 'train_audio')
    train_csv_path = os.path.join(DATA_ROOT, 'train.csv')
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes') 
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes') 
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')

    # Paths for VAD/Fabio - used via mount point in interactive mode
    VOICE_SEPARATION_DIR = os.path.join(GCS_MOUNT_POINT, "BC25 voice separation")
    FABIO_CSV_PATH = os.path.join(VOICE_SEPARATION_DIR, "fabio.csv")
    VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "train_voice_data.pkl") # Original VAD data
    TRANSFORMED_VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "transformed_train_voice_data.pkl") # Transformed VAD data

    # --- Preprocessing Settings --- #
    REMOVE_SPEECH_INTERVALS = False # Set to True to enable VAD/Fabio interval processing

    FS = 32000 
    TARGET_DURATION = 5.0  
    N_FFT = 1024
    # ... rest of Config class ...

# --- Paths --- #
VOICE_SEPARATION_DIR = "path_to_your_voice_separation_directory"
TRANSFORMED_VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "transformed_train_voice_data.pkl") # Transformed VAD data

# --- Preprocessing Flags --- #
REMOVE_SPEECH_INTERVALS = False # <<< NEW FLAG: Set to True to enable VAD/Fabio interval processing

FS = 32000 
TARGET_DURATION = 5.0 