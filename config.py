import torch
import os
import multiprocessing
from google.cloud import storage

class Config:
    seed = 40
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
    # Define the directory where preprocessing outputs go
    _PREPROCESSED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'preprocessed')
    # Define the specific path for the single NPZ file
    PREPROCESSED_NPZ_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'spectrograms.npz')
    MODEL_INPUT_DIR = MODEL_OUTPUT_DIR

    # These derived paths use the mount point when IS_CUSTOM_JOB is False
    train_audio_dir = os.path.join(DATA_ROOT, 'train_audio')
    train_csv_path = os.path.join(DATA_ROOT, 'train.csv')
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes')
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes')
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')
    # --- Add paths for rare species data ---
    train_rare_csv_path = os.path.join(DATA_ROOT, 'train_rare.csv')
    train_audio_rare_dir = os.path.join(DATA_ROOT, 'train_audio_rare')
    # --- End rare species data paths ---

    # Paths for VAD/Fabio - used via mount point in interactive mode
    VOICE_SEPARATION_DIR = os.path.join(GCS_MOUNT_POINT, "BC25 voice separation")
    FABIO_CSV_PATH = os.path.join(VOICE_SEPARATION_DIR, "fabio.csv")
    VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "train_voice_data.pkl") # Original VAD data
    TRANSFORMED_VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "transformed_train_voice_data.pkl") # Transformed VAD data

    FS = 32000 
    TARGET_DURATION = 5.0  
    N_FFT = 1024
    HOP_LENGTH = 128
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256) 

    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1
    num_classes = 206  

    LOAD_PREPROCESSED_DATA = False
    REMOVE_SPEECH_INTERVALS = False
    USE_RARE_DATA = False

    # --- On-the-fly Preloading Configuration ---
    PRELOAD_CHUNK_DURATION_SEC = 15.0 # Set to None or 0 to load full files

    epochs = 10
    train_batch_size = 32
    val_batch_size = 64
    use_amp = False

    criterion = 'BCEWithLogitsLoss'
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    scheduler = 'CosineAnnealingLR' 
    min_lr = 1e-6
    T_max = epochs 

    aug_prob = 0.5
    mixup_alpha = 0.5 

    inference_batch_size = 16
    use_tta = False
    tta_count = 3
    threshold = 0.7 

    use_specific_folds_inference = False
    inference_folds = [0, 1]

    debug_limit_batches = 5 
    debug_limit_files = 3 

config = Config()