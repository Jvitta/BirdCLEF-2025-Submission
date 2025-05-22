import torch
import os
import multiprocessing

class Config:
    # --- Root Paths --- #
    PROJECT_ROOT = "/home/ext_jvittimberga_gmail_com/BirdCLEF-2025-Submission"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

    RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models')
    _PREPROCESSED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'preprocessed')
    PREPROCESSED_NPZ_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'spectrograms.npz')
    PREPROCESSED_NPZ_PATH_VAL = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'spectrograms_val.npz')
    SOUNDSCAPE_VAL_NPZ_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'soundscape_val.npz')
    PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')

    # These derived paths use the mount point
    train_audio_dir = os.path.join(RAW_DATA_DIR, 'train_audio')
    train_csv_path = os.path.join(RAW_DATA_DIR, 'train.csv')
    test_audio_dir = os.path.join(RAW_DATA_DIR, 'test_soundscapes')
    sample_submission_path = os.path.join(RAW_DATA_DIR, 'sample_submission.csv')
    taxonomy_path = os.path.join(RAW_DATA_DIR, 'taxonomy.csv')

    # Rare species data paths
    train_audio_rare_dir = os.path.join(RAW_DATA_DIR, 'train_audio_rare')
    train_rare_csv_path = os.path.join(RAW_DATA_DIR, 'train_rare.csv')

    # Pseudo-label paths
    unlabeled_audio_dir = os.path.join(RAW_DATA_DIR, 'train_soundscapes')
    soundscape_pseudo_csv_path = os.path.join(RAW_DATA_DIR, 'soundscape_pseudo.csv')
    soundscape_pseudo_calibrated_csv_path = os.path.join(RAW_DATA_DIR, 'soundscape_pseudo_calibrated.csv')
    soundscape_val_path = os.path.join(RAW_DATA_DIR, 'soundscape_val_metadata.csv')

    # Paths for VAD/Fabio - used via mount point in interactive mode
    VOICE_SEPARATION_DIR = os.path.join(DATA_ROOT, "BC25 voice separation")
    FABIO_CSV_PATH = os.path.join(VOICE_SEPARATION_DIR, "fabio.csv")
    VOICE_DATA_PKL_PATH = os.path.join(VOICE_SEPARATION_DIR, "train_voice_data_final.pkl") # Original VAD data
 
    # Path for manual annotations from the UI
    ANNOTATED_SEGMENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "annotator_ui", "annotated_segments.csv")
    BIRDNET_DETECTIONS_NPZ_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, 'birdnet_detections.npz')
    ADAIN_PER_FREQUENCY_STATS_PATH = os.path.join(_PREPROCESSED_OUTPUT_DIR, "adain_per_frequency_stats.npz")
 
    EL_SILENCIO_LAT = 6.76       # Latitude of target location (El Silencio, Yondó)
    EL_SILENCIO_LON = -74.21     # Longitude of target location

    BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD = 0.05 # Threshold for BirdNET-generated pseudo labels

    seed = 43
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'mn20_as' 

    if 'mn' in model_name:
        width_mult = 2.0
        pretrained = True 
        in_channels = 1 
        num_classes = 206

        epochs = 10 # 10
        optimizer = 'AdamW'
        lr = 0.0006124336720699518 #0.0004 
        min_lr = 1e-6
        weight_decay = 0.0009586824835340106 #1e-4 
        dropout = 0.2
        scheduler = 'CosineAnnealingLR' 
        T_max = 15

        FS = 32000 
        TARGET_DURATION = 5.0  
        N_FFT = 1024
        HOP_LENGTH = 320
        WIN_LENGTH = 800
        N_MELS = 128
        FMIN = 20
        FMAX = None # if None, fmax = sr // 2 - fmax_aug_range // 2
        FMIN_AUG_RANGE = 10
        FMAX_AUG_RANGE = 1000
        PREPROCESS_TARGET_SHAPE = (128, 500) # Expected Preprocessing shape
        TARGET_SHAPE = (128, 1000) # Final shape for training/inference

        LOAD_PREPROCESSED_DATA = True
        REMOVE_SPEECH_INTERVALS = True
        USE_RARE_DATA = True
        USE_PSEUDO_LABELS = False
        USE_WEIGHTED_SAMPLING = False
        USE_MANUAL_ANNOTATIONS = True 
        REMOVE_SPEECH_ONLY_NON_AVES = True # Apply speech removal only to non-Aves classes if REMOVE_SPEECH_INTERVALS is True
        use_amp = False
        EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS = True
        ENABLE_DISTANCE_WEIGHTING = False
        USE_GLOBAL_OVERSAMPLING = False

        NUM_SPECTROGRAM_SAMPLES_TO_LOG = 30
        PRECOMPUTE_VERSIONS = 3 # Number of different chunks per primary file
        GLOBAL_OVERSAMPLING_MIN_SAMPLES = 5
        MIXING_RATIO_PRIMARY = 0.75 # Weight of primary audio in mix (background = 1.0 - this)

        # --- Dynamic Chunk Precomputation --- #
        DYNAMIC_CHUNK_COUNTING = True        # Enable/disable dynamic chunk counting based on species rarity
        MAX_CHUNKS_RARE = 7                  # Max chunks for files from species with very few files (e.g., 1 file)
        MIN_CHUNKS_COMMON = 1                # Min chunks for files from very common species
        COMMON_SPECIES_FILE_THRESHOLD = 200  # Species with >= this many files get MIN_CHUNKS_COMMON.
                                            # Interpolation happens for counts < COMMON_SPECIES_FILE_THRESHOLD down to 1 file.

        criterion = 'FocalLossBCE'
        focal_loss_alpha = 0.25
        focal_loss_gamma = 2.0
        focal_loss_bce_weight = 0.6 # Focal weight will be calculated as 2.0 - bce_weight

        label_smoothing_factor = 0.17663005428851927 #0.1

        n_fold = 5
        selected_folds = [0, 1, 2, 3, 4]

        train_batch_size = 32
        val_batch_size = 64
        inference_batch_size = 64

        # --- Augmentation Parameters ---
        batch_augment_prob = 1.0  # Probability of applying Mixup OR CutMix to a batch
        mixup_vs_cutmix_ratio = 1.0  # If augmenting, probability of choosing Mixup (vs CutMix)
        mixup_alpha = 0.4194592538670868 #0.3901120986458487 
        cutmix_alpha = 1.0           

        # Spectrogram Augmentation (applied manually in Dataset)
        time_mask_prob = 0.446875227279031 # Probability of applying time masking
        freq_mask_prob = 0.263834841662896 # Probability of applying frequency masking
        contrast_prob = 0.445985839941462  # Probability of applying random contrast
        max_time_mask_width = 45 #30     # Maximum width of time mask
        max_freq_mask_height = 15 #26    # Maximum height of frequency mask

        use_tta = False
        tta_count = 3
        # Threshold for generating pseudo labels
        threshold = 0.75
        pseudo_label_usage_threshold = 0.90
        use_specific_folds_inference = False
        inference_folds = [0, 1]

        debug = False
        debug_limit_batches = 5 
        debug_limit_files = 500

        # --- New: Smoothing Parameter ---
        smoothing_neighbor_weight = 0.125

        # --- BirdNET Preprocessing Config ---
        birdnet_confidence_threshold = 0.05
         # Minimum confidence for BirdNET detection to be considered
    
        # --- AdaIN Statistics ---
        ADAIN_MODE = 'none'  # Options: 'none', 'global', 'per_frequency'
        ADAIN_EPSILON = 1e-6 # Epsilon for numerical stability in division

        # --- Distance-based Loss Weighting (New) ---
        DISTANCE_WEIGHTING_THRESHOLD_KM = 1000.0 # Distance (km) within which weight is 1.0
        DISTANCE_WEIGHTING_FALLOFF_RANGE_KM = 4000.0 # Distance (km) over which weight falls off
        MIN_DISTANCE_WEIGHT = 0.7          # Minimum weight for distant samples
        # If a sample has no lat/lon, what weight to assign?
        # Options: 'min', 'mean_of_calculated_weights', or a specific float value e.g. 0.5
        DEFAULT_WEIGHT_FOR_MISSING_COORDS = 'min'

    elif 'efficientnet' in model_name:

        # Spectrogram & Audio Parameters for EfficientNet
        FS = 32000 
        TARGET_DURATION = 5.0  
        N_FFT = 1024
        HOP_LENGTH = 128
        N_MELS = 136
        FMIN = 20
        FMAX = 16000

        PREPROCESS_TARGET_SHAPE = (256, 256) # Consistent intermediate shape
        TARGET_SHAPE = PREPROCESS_TARGET_SHAPE # Final model input shape for EN

        # Model Architecture
        pretrained = True
        in_channels = 3
        num_classes = 206

        # Data Handling
        LOAD_PREPROCESSED_DATA = True
        REMOVE_SPEECH_INTERVALS = True
        USE_RARE_DATA = False 
        USE_PSEUDO_LABELS = False
        REMOVE_SPEECH_ONLY_NON_AVES = True
        USE_WEIGHTED_SAMPLING = False
        USE_MANUAL_ANNOTATIONS = False
        EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS = True
        ENABLE_DISTANCE_WEIGHTING = False 

        NUM_SPECTROGRAM_SAMPLES_TO_LOG = 30
        PRECOMPUTE_VERSIONS = 3
        MIXING_RATIO_PRIMARY = 0.75

        # Dynamic Chunking (using values from MN for now, can be tuned for EN)
        DYNAMIC_CHUNK_COUNTING = False
        MAX_CHUNKS_RARE = 7
        MIN_CHUNKS_COMMON = 1
        COMMON_SPECIES_FILE_THRESHOLD = 200

        # Training Parameters
        epochs = 10
        train_batch_size = 32
        val_batch_size = 64
        use_amp = False

        # Loss Function
        criterion = 'FocalLossBCE'
        focal_loss_alpha = 0.25
        focal_loss_gamma = 2.0
        focal_loss_bce_weight = 0.6
        label_smoothing_factor = 0.1

        # Folds
        n_fold = 5
        selected_folds = [0, 1, 2, 3, 4]

        # Optimizer & Scheduler
        optimizer = 'AdamW'
        lr = 0.0005759790964526907
        min_lr = 1e-6
        weight_decay = 1.3461944764663799e-05
        scheduler = 'CosineAnnealingLR' 
        T_max = epochs 

        # Augmentations
        batch_augment_prob = 1.0
        mixup_vs_cutmix_ratio = 1.0
        mixup_alpha = 0.3901120986458487 
        cutmix_alpha = 1.0           
        time_mask_prob = 0.446875227279031
        freq_mask_prob = 0.263834841662896
        contrast_prob = 0.445985839941462
        max_time_mask_width = 30
        max_freq_mask_height = 26

        # Inference & Debug
        inference_batch_size = 64
        use_tta = False
        tta_count = 3
        threshold = 0.75
        pseudo_label_usage_threshold = 0.90
        use_specific_folds_inference = False
        inference_folds = [0, 1]
        debug = False
        debug_limit_batches = 5 
        debug_limit_files = 500

        # Other Parameters
        smoothing_neighbor_weight = 0.125
        birdnet_confidence_threshold = 0.1
        
        ADAIN_MODE = 'none'
        ADAIN_EPSILON = 1e-6

        # --- Distance-based Loss Weighting (New) ---
        DISTANCE_WEIGHTING_THRESHOLD_KM = 1000.0 # Distance (km) within which weight is 1.0
        DISTANCE_WEIGHTING_FALLOFF_RANGE_KM = 4000.0 # Distance (km) over which weight falls off
        MIN_DISTANCE_WEIGHT = 0.3          # Minimum weight for distant samples
        # If a sample has no lat/lon, what weight to assign?
        # Options: 'min', 'mean_of_calculated_weights', or a specific float value e.g. 0.5
        DEFAULT_WEIGHT_FOR_MISSING_COORDS = 'min'

    _wandb_log_params = [
        'seed', 'TARGET_DURATION', 'N_MELS', 'PREPROCESS_TARGET_SHAPE', 'TARGET_SHAPE',
        'model_name', 'width_mult', 'REMOVE_SPEECH_INTERVALS', 'USE_RARE_DATA', 'USE_PSEUDO_LABELS',
        'USE_WEIGHTED_SAMPLING', 'USE_MANUAL_ANNOTATIONS', 'PRECOMPUTE_VERSIONS', 'use_amp', 'criterion',
        'focal_loss_alpha', 'focal_loss_gamma', 'focal_loss_bce_weight',
        'label_smoothing_factor', 'n_fold', 'selected_folds', 'epochs', 'optimizer',
        'lr', 'min_lr', 'weight_decay', 'scheduler', 'T_max', 'train_batch_size',
        'val_batch_size', 'batch_augment_prob', 'mixup_vs_cutmix_ratio',
        'mixup_alpha', 'cutmix_alpha', 'time_mask_prob', 'freq_mask_prob',
        'contrast_prob', 'max_time_mask_width', 'max_freq_mask_height',
        'pseudo_label_usage_threshold', 'smoothing_neighbor_weight',
        'BIRDNET_PSEUDO_CONFIDENCE_THRESHOLD', 'birdnet_confidence_threshold',
        'ADAIN_MODE', 'ADAIN_EPSILON',
        'DYNAMIC_CHUNK_COUNTING', 'MAX_CHUNKS_RARE', 'MIN_CHUNKS_COMMON',
        'COMMON_SPECIES_FILE_THRESHOLD',
        'EXCLUDE_FILES_WITH_ONLY_LOW_QUALITY_MANUAL_ANNOTATIONS',
        'ENABLE_DISTANCE_WEIGHTING', 'EL_SILENCIO_LAT', 'EL_SILENCIO_LON',
        'DISTANCE_WEIGHTING_THRESHOLD_KM', 'MIN_DISTANCE_WEIGHT', 'DEFAULT_WEIGHT_FOR_MISSING_COORDS'
    ]

    def get_wandb_config(self):
        cfg_dict = {}
        for attr_name in self._wandb_log_params:
            if hasattr(self, attr_name):
                cfg_dict[attr_name] = getattr(self, attr_name)
        return cfg_dict

config = Config()