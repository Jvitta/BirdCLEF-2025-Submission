import torch
import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    print("Config: Detected Kaggle environment")

    BASE_INPUT_DIR = '/kaggle/input/birdclef-2025'
    BASE_OUTPUT_DIR = '/kaggle/working/'
    VAD_INPUT_DIR = '/kaggle/input/bc25-separation-voice-from-data-by-silero-vad'
    PRECOMPUTED_MODEL_DIR = '/kaggle/input/birdclef-m136-fft1024-hop64-fullv2/models'
else:
    print("Config: Assuming GCP/External environment (using gcsfuse mount point)")
    # *** REPLACE WITH YOUR ACTUAL GCSFUSE MOUNT POINT ***
    GCS_MOUNT_POINT = '/mnt/gcs_bucket' 
    print(f"Config: Using GCS mount point: {GCS_MOUNT_POINT}")

    BASE_INPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'data/birdclef-2025')
    BASE_OUTPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'working/')
    VAD_INPUT_DIR = os.path.join(GCS_MOUNT_POINT, 'data/bc25-separation-voice-from-data-by-silero-vad')
    PRECOMPUTED_MODEL_DIR = os.path.join(GCS_MOUNT_POINT, 'data/birdclef-m136-fft1024-hop64-fullv2/models')

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'models/'), exist_ok=True)


class Config:
    seed = 42
    debug = False  
    debug_preprocessing_mode = False
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_ROOT = BASE_INPUT_DIR
    OUTPUT_DIR = BASE_OUTPUT_DIR
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models/')
    MODEL_INPUT_DIR = PRECOMPUTED_MODEL_DIR

    FABIO_CSV_PATH = os.path.join(VAD_INPUT_DIR, 'fabio.csv')
    VOICE_DATA_PKL_PATH = os.path.join(VAD_INPUT_DIR, 'train_voice_data.pkl')

    train_audio_dir = os.path.join(DATA_ROOT, 'train_audio')
    train_csv_path = os.path.join(DATA_ROOT, 'train.csv')
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes')
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes')
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')
    
    FS = 32000
    TARGET_DURATION = 5.0

    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)

    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1
    num_classes = 206

    N_MAX_PREPROCESS = 50 if debug_preprocessing_mode else None 
    _PREPROCESSED_FILENAME_BASE = f"spectrogram_m{N_MELS}_fft{N_FFT}_hop{HOP_LENGTH}"
    _MODE_STR = f"sample{N_MAX_PREPROCESS}" if debug_preprocessing_mode else "full"
    PREPROCESSED_FILENAME = f"{_PREPROCESSED_FILENAME_BASE}_{_MODE_STR}.npy"
    PREPROCESSED_FILEPATH = os.path.join(OUTPUT_DIR, 'preprocessed', PREPROCESSED_FILENAME)

    CHUNKED_METADATA_FILENAME = f"train_metadata_chunked_{_MODE_STR}.csv"
    CHUNKED_METADATA_PATH = os.path.join(OUTPUT_DIR, CHUNKED_METADATA_FILENAME)

    LOAD_PREPROCESSED_DATA = True
    epochs = 10
    train_batch_size = 32
    val_batch_size = 64

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

    TEMP_CHUNK_DIR = os.path.join(OUTPUT_DIR, "temp_preprocessed_chunks")
    PREPROCESSED_ZIP_PATH = os.path.join(OUTPUT_DIR, "preprocessed_chunks.zip")

config = Config()

print(f"Config: Device set to \"{config.device}\"")
print(f"Config: Using DATA_ROOT: {config.DATA_ROOT}")
print(f"Config: Using OUTPUT_DIR: {config.OUTPUT_DIR}")