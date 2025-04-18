import torch
import os

# --- Environment Detection --- 
# Simple check for Kaggle environment variables
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# --- Base Path Definitions --- 
if IS_KAGGLE:
    print("Config: Detected Kaggle environment")
    # Kaggle specific paths
    BASE_INPUT_DIR = '/kaggle/input/birdclef-2025' # Adjust if dataset name changes
    BASE_OUTPUT_DIR = '/kaggle/working/'
    # Specific Kaggle input datasets if needed
    VAD_INPUT_DIR = '/kaggle/input/bc25-separation-voice-from-data-by-silero-vad'
    PRECOMPUTED_MODEL_DIR = '/kaggle/input/birdclef-m136-fft1024-hop64-fullv2/models'
else:
    print("Config: Assuming GCP/External environment (using gcsfuse mount point)")
    # *** REPLACE WITH YOUR ACTUAL GCSFUSE MOUNT POINT ***
    GCS_MOUNT_POINT = '/mnt/gcs_bucket' 
    print(f"Config: Using GCS mount point: {GCS_MOUNT_POINT}")

    # Define base paths relative to the mount point
    # Adjust these sub-paths if your bucket structure is different
    BASE_INPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'input/birdclef-2025')
    BASE_OUTPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'working/')
    VAD_INPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'input/bc25-separation-voice-from-data-by-silero-vad')
    PRECOMPUTED_MODEL_DIR = os.path.join(GCS_MOUNT_POINT, 'input/birdclef-m136-fft1024-hop64-fullv2/models')

    # Ensure the output directory structure exists on the mounted drive
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'models/'), exist_ok=True)


class Config:
    # --- General ---
    seed = 42
    debug = False  # Master debug flag for limiting epochs, batches etc.
    debug_preprocessing_mode = False # Controls N_MAX_PREPROCESS and filename suffix
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Paths ---
    DATA_ROOT = BASE_INPUT_DIR
    OUTPUT_DIR = BASE_OUTPUT_DIR # General output for logs, figures, etc.
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models/') # Where trained models are saved
    MODEL_INPUT_DIR = PRECOMPUTED_MODEL_DIR # Path to load models FROM

    # Paths for precomputed VAD/interval data
    FABIO_CSV_PATH = os.path.join(VAD_INPUT_DIR, 'fabio.csv')
    VOICE_DATA_PKL_PATH = os.path.join(VAD_INPUT_DIR, 'train_voice_data.pkl')

    # Derived paths
    train_audio_dir = os.path.join(DATA_ROOT, 'train_audio')
    train_csv_path = os.path.join(DATA_ROOT, 'train.csv')
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes')
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes') # Populated during inference
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')

    # --- Audio & Spectrogram Parameters (Using values from original inference.py) ---
    FS = 32000 # Sample Rate
    TARGET_DURATION = 5.0  # seconds

    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256) # Final shape after potential resizing

    # --- Model ---
    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1
    num_classes = 206  # Number of bird species (adjust if taxonomy changes)

    # --- Preprocessing ---
    # Limit samples during preprocessing ONLY if debug_preprocessing_mode is True
    N_MAX_PREPROCESS = 50 if debug_preprocessing_mode else None 
    # Define base name and mode string for preprocessed file
    _PREPROCESSED_FILENAME_BASE = f"spectrogram_m{N_MELS}_fft{N_FFT}_hop{HOP_LENGTH}"
    # Mode string now depends on the dedicated flag
    _MODE_STR = f"sample{N_MAX_PREPROCESS}" if debug_preprocessing_mode else "full"
    # Construct the full filename and path
    PREPROCESSED_FILENAME = f"{_PREPROCESSED_FILENAME_BASE}_{_MODE_STR}.npy"
    PREPROCESSED_FILEPATH = os.path.join(OUTPUT_DIR, 'preprocessed', PREPROCESSED_FILENAME)

    # Path for the metadata file corresponding to the chunks
    CHUNKED_METADATA_FILENAME = f"train_metadata_chunked_{_MODE_STR}.csv"
    CHUNKED_METADATA_PATH = os.path.join(OUTPUT_DIR, CHUNKED_METADATA_FILENAME)

    # Path to specific preprocessed file for verification/testing

    # --- Training ---
    LOAD_PREPROCESSED_DATA = True # If True, load from PREPROCESSED_FILEPATH; if False, generate on-the-fly
    epochs = 10
    train_batch_size = 32
    val_batch_size = 64

    criterion = 'BCEWithLogitsLoss'
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    # Optimizer
    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    # Scheduler
    scheduler = 'CosineAnnealingLR' # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'OneCycleLR', None
    min_lr = 1e-6
    T_max = epochs # For CosineAnnealingLR

    # Augmentations
    aug_prob = 0.5
    mixup_alpha = 0.5 # 0 means no mixup

    # --- Inference ---
    inference_batch_size = 16
    use_tta = False
    tta_count = 3
    threshold = 0.7 # Prediction threshold (currently unused in provided code)

    # Fold selection for inference
    use_specific_folds_inference = False
    inference_folds = [0, 1]

    # Debug settings for loops
    debug_limit_batches = 5 # Max batches per epoch if debug=True
    debug_limit_files = 3 # Max files for inference if debug=True

    # --- New Paths for Chunked Saving --- #
    TEMP_CHUNK_DIR = os.path.join(OUTPUT_DIR, "temp_preprocessed_chunks") # Temp dir for individual .npy files
    PREPROCESSED_ZIP_PATH = os.path.join(OUTPUT_DIR, "preprocessed_chunks.zip") # Final zip archive

# --- Instantiate config ---
config = Config()

# --- Post-Instantiation Checks/Setup (Optional but Recommended) --- 
# Example: Update num_classes based on taxonomy
try:
    if os.path.exists(config.taxonomy_path):
        import pandas as pd
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        actual_num_classes = taxonomy_df['primary_label'].nunique()
        if config.num_classes != actual_num_classes:
             print(f"Config Warning: num_classes ({config.num_classes}) does not match taxonomy unique labels ({actual_num_classes}). Updating config.")
             config.num_classes = actual_num_classes
    else:
        print(f"Config Warning: Taxonomy file not found at {config.taxonomy_path}. Using default num_classes: {config.num_classes}.")
except Exception as e:
    print(f"Config Error during taxonomy check: {e}. Using default num_classes: {config.num_classes}.")


print(f"Config: Device set to \"{config.device}\"")
print(f"Config: Using DATA_ROOT: {config.DATA_ROOT}")
print(f"Config: Using OUTPUT_DIR: {config.OUTPUT_DIR}")