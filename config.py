import torch
import os
import multiprocessing

class Config:
    seed = 42
    debug = False  # Master debug flag for limiting epochs, batches etc.
    debug_preprocessing_mode = False # Controls N_MAX_PREPROCESS and filename suffix
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PROJECT_ROOT = "/home/jupyter/BirdCLEF-2025-Submission"
    GCS_MOUNT_POINT = "/home/jupyter/gcs_mount"

    DATA_ROOT = os.path.join(GCS_MOUNT_POINT, "raw_data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models')
    PREPROCESSED_DATA_DIR = os.path.join(GCS_MOUNT_POINT, 'preprocessed')
    MODEL_INPUT_DIR = MODEL_OUTPUT_DIR

    train_audio_dir = os.path.join(DATA_ROOT, 'train_audio')
    train_csv_path = os.path.join(DATA_ROOT, 'train.csv')
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes') 
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes') 
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')

    # --- Add Paths for Voice Separation Data --- #
    VOICE_SEPARATION_DIR = os.path.join(GCS_MOUNT_POINT, "BC25 voice separation")
    FABIO_CSV_PATH = os.path.join(VOICE_SEPARATION_DIR, "fabio.csv")
    VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "train_voice_data.pkl")
    TRANSFORMED_VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "transformed_train_voice_data.pkl")
    # --- End Voice Separation Paths --- #

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


    # Comment out single-file related variables
    # N_MAX_PREPROCESS = 50 if debug_preprocessing_mode else None
    # _PREPROCESSED_FILENAME_BASE = f"spectrogram_m{N_MELS}_fft{N_FFT}_hop{HOP_LENGTH}"
    # _MODE_STR = f"sample{N_MAX_PREPROCESS}" if debug_preprocessing_mode else "full"
    # PREPROCESSED_FILENAME = f"{_PREPROCESSED_FILENAME_BASE}_{_MODE_STR}.npy"
    # PREPROCESSED_FILEPATH = os.path.join(PREPROCESSED_DATA_DIR, PREPROCESSED_FILENAME)

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

config = Config()