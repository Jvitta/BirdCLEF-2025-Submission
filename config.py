import torch
import os
import multiprocessing

class Config:
    seed = 40
    debug = True  # Master debug flag for limiting epochs, batches etc.
    debug_preprocessing_mode = True # Controls N_MAX_PREPROCESS and filename suffix
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    test_audio_dir = os.path.join(DATA_ROOT, 'test_soundscapes')
    sample_submission_path = os.path.join(DATA_ROOT, 'sample_submission.csv')
    taxonomy_path = os.path.join(DATA_ROOT, 'taxonomy.csv')
    # Rare species data paths
    train_audio_rare_dir = os.path.join(DATA_ROOT, 'train_audio_rare')
    train_rare_csv_path = os.path.join(DATA_ROOT, 'train_rare.csv')
    # Pseudo-label paths
    unlabeled_audio_dir = os.path.join(DATA_ROOT, 'train_soundscapes')
    train_pseudo_csv_path = os.path.join(DATA_ROOT, 'train_pseudo.csv')

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
    in_channels = 3
    num_classes = 206

    LOAD_PREPROCESSED_DATA = True
    REMOVE_SPEECH_INTERVALS = True
    USE_RARE_DATA = False
    USE_PSEUDO_LABELS = False

    REMOVE_SPEECH_ONLY_NON_AVES = True # Apply speech removal only to non-Aves classes if REMOVE_SPEECH_INTERVALS is True

    PRECOMPUTE_VERSIONS = 3 # Number of different 5s chunks per primary file
    MIXING_RATIO_PRIMARY = 0.75 # Weight of primary audio in mix (background = 1.0 - this)

    epochs = 10
    train_batch_size = 32
    val_batch_size = 64
    use_amp = False

    criterion = 'FocalLossBCE'
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2.0
    focal_loss_bce_weight = 1.0 # Focal weight will be calculated as 2.0 - bce_weight

    label_smoothing_factor = 0.1

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 0.0005759790964526907
    min_lr = 1e-6
    weight_decay = 1.3461944764663799e-05

    scheduler = 'CosineAnnealingLR' 
    T_max = epochs 

    # --- Augmentation Parameters ---
    # Batch-level augmentations (Mixup/CutMix)
    batch_augment_prob = 1.0     # Probability of applying Mixup OR CutMix to a batch
    mixup_vs_cutmix_ratio = 1.0  # If augmenting, probability of choosing Mixup (vs CutMix)
    mixup_alpha = 0.3901120986458487 # Mixup alpha parameter (higher means more similar mixes)
    cutmix_alpha = 1.0           # CutMix alpha parameter (controls patch size distribution)

    # Spectrogram Augmentation (applied manually in Dataset)
    time_mask_prob = 0.446875227279031 # Probability of applying time masking
    freq_mask_prob = 0.263834841662896 # Probability of applying frequency masking
    contrast_prob = 0.445985839941462  # Probability of applying random contrast
    max_time_mask_width = 30     # Maximum width of time mask
    max_freq_mask_height = 26    # Maximum height of frequency mask

    inference_batch_size = 16
    use_tta = False
    tta_count = 3
    # Threshold for generating pseudo labels
    threshold = 0.75
    pseudo_label_usage_threshold = 0.80
    use_specific_folds_inference = False
    inference_folds = [0, 1]

    debug_limit_batches = 5 
    debug_limit_files = 500

    # --- New: Smoothing Parameter ---
    smoothing_neighbor_weight = 0.125

    # --- BirdNET Preprocessing Config ---
    birdnet_confidence_threshold = 0.1 # Minimum confidence for BirdNET detection to be considered
    BIRDNET_DETECTIONS_NPZ_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'birdnet_detections.npz')

config = Config()

# Optional: Add another print here to confirm the final config object shape if needed
# print(f"[Config Instance] TARGET_SHAPE: {config.TARGET_SHAPE}")