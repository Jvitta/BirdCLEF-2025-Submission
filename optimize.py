import optuna
import pandas as pd
import copy
import sys
import os
import logging
import matplotlib
import plotly
import shutil # Add shutil for file copying
import numpy as np # Add numpy for loading
from tqdm.auto import tqdm # Add tqdm for loading progress
import time # Add time for loading timer
import functools # Add functools for passing args to objective

from config import config as base_config

from src.training.train_mn import run_training, set_seed, calculate_auc # Import calculate_auc

# load data once for HPO
print("Loading main training metadata for HPO...")
try:
    main_train_df_full = pd.read_csv(base_config.train_csv_path)
    # Add the necessary samplename derivation
    if 'filename' in main_train_df_full.columns:
        main_train_df_full['samplename'] = main_train_df_full.filename.map(
            lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0]
        )
        main_train_df_full['data_source'] = 'main' # Add data_source column
        # Ensure filepath is created, similar to train_mn.py
        main_train_df_full['filepath'] = main_train_df_full['filename'].apply(
            lambda f: os.path.join(base_config.train_audio_dir, f)
        )
    else:
        print("Error: 'filename' column not found in main training CSV. Cannot create 'samplename' or 'filepath'. Exiting.")
        sys.exit(1)
    
    # Select necessary columns, now including those needed for grouping
    required_cols_main = [
        'samplename', 'primary_label', 'secondary_labels', 'data_source', 
        'latitude', 'longitude', 'author', 'filepath', 'filename' # Added grouping and file path cols
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in required_cols_main if col not in main_train_df_full.columns]
    if missing_cols:
        print(f"Error: Required columns missing from main training CSV: {missing_cols}. Exiting.")
        sys.exit(1)
    
    main_train_df = main_train_df_full[required_cols_main].copy()
    print(f"Successfully loaded and prepared {len(main_train_df)} samples for HPO, including grouping columns.")
    del main_train_df_full # Free up memory

except FileNotFoundError:
    print(f"Error: Main training CSV not found at {base_config.train_csv_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading main training CSV: {e}. Exiting.")
    sys.exit(1)

def objective(trial, preloaded_data, preloaded_val_data):
    """Runs one training trial with hyperparameters suggested by Optuna."""
    global main_train_df

    cfg = copy.deepcopy(base_config)

    # --- Hyperparameter Suggestions --- #
    # Keep learning rate search tight around the known good value
    cfg.lr = trial.suggest_float("lr", 2e-4, 7e-4, log=True) 

    # Keep weight decay search tight around the known good value
    cfg.weight_decay = trial.suggest_float("weight_decay", 5e-5, 1e-3, log=True) 

    # Dropout
    cfg.dropout = trial.suggest_float("dropout", 0.1, 0.4)

    # Max Frequency Mask Height (for 128 mel bins)
    cfg.max_freq_mask_height = trial.suggest_int("max_freq_mask_height", 8, 24)

    # Max Time Mask Width (for 500 time frames in preprocessed spec)
    cfg.max_time_mask_width = trial.suggest_int("max_time_mask_width", 20, 80)

    # Augmentation Probabilities
    cfg.time_mask_prob = trial.suggest_float("time_mask_prob", 0.2, 0.7)
    cfg.freq_mask_prob = trial.suggest_float("freq_mask_prob", 0.1, 0.5)

    # Label Smoothing
    cfg.label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.0, 0.25)

    # Mixup Alpha
    cfg.mixup_alpha = trial.suggest_float("mixup_alpha", 0.2, 0.6)
    
    # Focal Loss Gamma
    cfg.focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 1.5, 3.0)

    # Keep other augmentation probabilities fixed for this study, as their effect might be captured by mask sizes
    # cfg.time_mask_prob = base_config.time_mask_prob # Now tuned
    # cfg.freq_mask_prob = base_config.freq_mask_prob # Now tuned
    # cfg.contrast_prob = base_config.contrast_prob # Now tuned
    # --- End Hyperparameter Suggestions ---

    # --- Trial Configuration ---
    # Use base_config.epochs as max epochs PER FOLD for HPO trials
    hpo_epochs_per_fold = base_config.epochs 
    cfg.epochs = hpo_epochs_per_fold # Set this for run_training
    print(f"INFO: Running HPO trial with max {cfg.epochs} epochs per fold.")

    trial_model_dir_base = os.path.join(base_config.MODEL_OUTPUT_DIR, f"hpo_trial_{trial.number}")
    os.makedirs(trial_model_dir_base, exist_ok=True) # Create a base directory for the trial

    cfg.debug = False
    cfg.debug_limit_batches = float('inf')

    trial_seed = base_config.seed + trial.number # Use a consistent seed derivation for the trial
    # Seed will be set per fold inside run_training if needed, or use trial_seed globally if preferred for HPO
    # For now, run_training sets its own seed based on config.seed which will be trial_seed here.
    cfg.seed = trial_seed 

    print(f"\n--- Starting Optuna Trial {trial.number} (Overall Seed: {cfg.seed}) ---")
    print(f"Parameters: {trial.params}")

    fold_aucs = []
    folds_to_run_hpo = [0, 1] # Run on Fold 0 and Fold 1

    try:
        for i, fold_num in enumerate(folds_to_run_hpo):
            print(f"\n--- Trial {trial.number}, Processing Fold {fold_num} ---")
            cfg_fold = copy.deepcopy(cfg) # Use a fresh copy of cfg for each fold to avoid state leakage
            cfg_fold.selected_folds = [fold_num]
            
            # Set model output directory for this specific fold to keep checkpoints separate if needed
            cfg_fold.MODEL_OUTPUT_DIR = os.path.join(trial_model_dir_base, f"fold_{fold_num}")
            os.makedirs(cfg_fold.MODEL_OUTPUT_DIR, exist_ok=True)

            current_hpo_step_offset = i * hpo_epochs_per_fold

            # Pass the trial object for potential pruning callback within run_training
            fold_best_auc = run_training(
                main_train_df, 
                cfg_fold, 
                trial=trial, 
                all_spectrograms=preloaded_data, 
                val_spectrogram_data=preloaded_val_data,
                custom_run_name=f"hpo_trial_{trial.number}_fold{fold_num}",
                hpo_step_offset=current_hpo_step_offset
            )

            if fold_best_auc is None or not isinstance(fold_best_auc, (int, float)) or pd.isna(fold_best_auc) or not abs(fold_best_auc) > 0:
                 print(f"Warning: Trial {trial.number}, Fold {fold_num} resulted in invalid AUC ({fold_best_auc}). Appending 0.0.")
                 fold_aucs.append(0.0)
            else:
                 print(f"--- Trial {trial.number}, Fold {fold_num} | Best Val AUC for Fold: {fold_best_auc:.4f} ---")
                 fold_aucs.append(fold_best_auc)
        
        # If any fold was pruned, optuna.TrialPruned would have been raised and caught below.
        # If we reach here, both folds completed (or returned a value that wasn't a prune exception).
        final_objective_value = np.mean(fold_aucs) if fold_aucs else 0.0
        print(f"\n--- Finished Optuna Trial {trial.number} | Average Val AUC (Folds {folds_to_run_hpo}): {final_objective_value:.4f} ---")

    except optuna.TrialPruned:
        print(f"--- Optuna Trial {trial.number} was pruned during one of its folds ---")
        raise # Re-raise the exception to signal pruning to Optuna

    except Exception as e:
        print(f"\n--- Non-pruning Exception occurred during Optuna trial {trial.number} ---")
        import traceback
        traceback.print_exc()
        print(f"--- End of traceback for trial {trial.number} ---")
        final_objective_value = 0.0 # Report 0.0 AUC for failed trials

    # Clean up trial-specific model directory base (contains subdirs for fold_0, fold_1)
    try:
        shutil.rmtree(trial_model_dir_base)
        print(f"Removed trial base directory: {trial_model_dir_base}")
    except OSError as e:
        print(f"Warning: Could not remove trial base directory {trial_model_dir_base}: {e}")

    return final_objective_value

if __name__ == "__main__":
    # --- Study Configuration ---
    study_name = "BirdCLEF_HPO_New_Validation_Set2" # Focused study name
    n_trials = 200 # Adjust number of trials as needed for this focused study

    # --- Define paths for GCP --- #
    # Database will be stored in the OUTPUT_DIR defined in config.py
    db_filename = "hpo_new_validation_set_study_results2.db" # Specific DB file
    db_filepath = os.path.join(base_config.OUTPUT_DIR, db_filename)
    storage_path = f"sqlite:///{db_filepath}" # Use absolute path for Optuna

    # --- Database Handling (Simplified for GCP) --- #
    storage_path_to_use = storage_path
    print(f"Optuna database path: {db_filepath}")
    if os.path.exists(db_filepath):
        print("Existing database found. Optuna will load study if it exists.")
    else:
        print("No existing database found. Optuna will create a new one.")
        # Ensure the output directory exists for the new database
        os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

    # Define HPO settings for clarity
    hpo_trial_epochs = base_config.epochs # Use base epochs
    # hpo_trial_folds = [0, 1, 2, 3, 4] # We run one fold per trial

    print(f"\n--- Starting Optuna Optimization --- ")
    print(f"Study Name: {study_name}")
    print(f"Number of Trials: {n_trials}")
    print(f"Storage (using): {storage_path_to_use}")
    print(f"Metric to Optimize: Validation AUC on single fold per trial (Maximizing)")
    print(f"HPO Trial Epochs: {hpo_trial_epochs}")
    # print(f"HPO Trial Folds: {hpo_trial_folds}")

    # Adjust n_warmup_steps for MedianPruner based on epochs per HPO fold
    hpo_trial_epochs_per_fold = base_config.epochs # Assuming this is epochs PER FOLD in HPO
    # Allow each fold a few epochs to start learning before pruning becomes aggressive.
    # For a 2-fold HPO trial, this means pruning effectively starts a few epochs into the second fold.
    warmup_steps_for_pruner = hpo_trial_epochs_per_fold // 2 # e.g., if 10 epochs/fold, warmup is 5 steps.

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        # Configure MedianPruner: Prune if value is worse than median after n_warmup_steps epochs
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup_steps_for_pruner, interval_steps=1),
        storage=storage_path_to_use, # Use the defined path
        load_if_exists=True # Load if the db file exists and study_name matches
    )

    # Add base config parameters and tuned parameters as user attributes for reference
    study.set_user_attr("base_config_epochs", base_config.epochs)
    study.set_user_attr("hpo_trial_epochs", hpo_trial_epochs)
    study.set_user_attr("hpo_fixed_fold", 2) # Indicate that HPO runs on a fixed fold
    # study.set_user_attr("hpo_trial_folds", hpo_trial_folds) # Removed as we use a fixed fold
    # Updated list of tuned parameters
    tuned_params_list = [
        "lr", "weight_decay", "dropout", 
        "max_freq_mask_height", "max_time_mask_width", 
        "time_mask_prob", "freq_mask_prob",
        "label_smoothing_factor", "mixup_alpha", "focal_loss_gamma"
    ]
    study.set_user_attr("tuned_parameters", tuned_params_list)
    # Store fixed parameters for reference
    study.set_user_attr("fixed_optimizer", base_config.optimizer)
    study.set_user_attr("fixed_scheduler", base_config.scheduler)
    # study.set_user_attr("fixed_mixup_alpha", base_config.mixup_alpha) # mixup_alpha is tuned
    study.set_user_attr("base_config_mixup_alpha", base_config.mixup_alpha) # Log base value for context

    # Log base augmentation probabilities and contrast, as mask sizes are tuned
    study.set_user_attr("base_config_augmentations_fixed_values", {
        # These are now tuned, so we log their original base_config values for reference
        "base_time_mask_prob": base_config.time_mask_prob,
        "base_freq_mask_prob": base_config.freq_mask_prob,
        "base_dropout": base_config.dropout,
        "base_focal_loss_gamma": base_config.focal_loss_gamma
        # max_time_mask_width and max_freq_mask_height are tuned, so not listed as fixed here
        # "max_time_mask_width": base_config.max_time_mask_width,
        # "max_freq_mask_height": base_config.max_freq_mask_height
    })

    # --- Pre-load NPZ data once --- #
    global_all_spectrograms = None
    if base_config.LOAD_PREPROCESSED_DATA:
        npz_path = base_config.PREPROCESSED_NPZ_PATH
        print(f"Attempting to pre-load NPZ file into RAM for HPO: {npz_path}")
        if not os.path.exists(npz_path):
             print(f"Error: LOAD_PREPROCESSED_DATA is True, but NPZ file {npz_path} does not exist.")
             print("       Please run preprocessing.py first.")
             sys.exit(1)
        else:
            try:
                print("Loading... (This might take a moment for large files)")
                start_load_time = time.time()
                with np.load(npz_path) as data_archive:
                    global_all_spectrograms = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading NPZ into RAM")}
                end_load_time = time.time()
                print(f"Successfully pre-loaded {len(global_all_spectrograms)} samples into RAM in {end_load_time - start_load_time:.2f} seconds.")
            except Exception as e:
                print(f"Error loading NPZ file into RAM: {e}")
                print("Cannot continue without preloaded data when LOAD_PREPROCESSED_DATA is True.")
                sys.exit(1)
    else:
        print("\nConfigured to generate spectrograms on-the-fly (no pre-loading for HPO).")
    # --- End Pre-loading --- #

    # --- Pre-load DEDICATED VALIDATION Spectrograms (if available) ---
    global_val_spectrograms = None
    if hasattr(base_config, 'PREPROCESSED_NPZ_PATH_VAL') and base_config.PREPROCESSED_NPZ_PATH_VAL:
        val_npz_path = base_config.PREPROCESSED_NPZ_PATH_VAL
        print(f"Attempting to pre-load DEDICATED VALIDATION NPZ file into RAM for HPO: {val_npz_path}")
        if os.path.exists(val_npz_path):
            try:
                print("Loading DEDICATED VALIDATION NPZ... (This might take a moment)")
                start_load_time_val = time.time()
                with np.load(val_npz_path) as data_archive_val:
                    global_val_spectrograms = {key: data_archive_val[key] for key in tqdm(data_archive_val.keys(), desc="Loading VAL NPZ into RAM")}
                end_load_time_val = time.time()
                print(f"Successfully pre-loaded {len(global_val_spectrograms)} DEDICATED VALIDATION samples into RAM in {end_load_time_val - start_load_time_val:.2f} seconds.")
            except Exception as e:
                print(f"Error loading DEDICATED VALIDATION NPZ file ({val_npz_path}) into RAM: {e}")
                global_val_spectrograms = None # Ensure it's None on error
        else:
            print(f"Info: DEDICATED VALIDATION NPZ file not found at {val_npz_path}. HPO will proceed without it.")
    else:
        print("\nInfo: config.PREPROCESSED_NPZ_PATH_VAL not defined. HPO will proceed without dedicated validation specs.")
    # --- End DEDICATED VALIDATION Pre-loading ---

    try:
        # Use functools.partial to pass fixed arguments (like loaded data) to the objective
        objective_with_data = functools.partial(objective, 
                                                preloaded_data=global_all_spectrograms,
                                                preloaded_val_data=global_val_spectrograms) # Pass val data
        study.optimize(objective_with_data, n_trials=n_trials, timeout=None)
    except KeyboardInterrupt:
        print("\nOptimization stopped manually. Saving current results.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Optuna Optimization Finished ---")
    print(f"Study Name: {study.study_name}")
    print(f"Storage (Working Copy): {storage_path_to_use}")
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        # Ensure best_trial exists and is valid before accessing attributes
        if study.best_trial:
            best_trial = study.best_trial
            print("\nBest trial:")
            print(f"  Value (Best Val AUC): {best_trial.value:.5f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            print(f"  Trial Number: {best_trial.number}")
        else:
            print("\nNo completed trials found or best trial unavailable.") # Handles case where study ran but no trials completed successfully
    except ValueError: # Optuna raises ValueError if study has no trials or no completed trials
        print("\nNo completed trials found. Cannot determine the best trial.")

    try:
        history_plot_path = os.path.join(base_config.OUTPUT_DIR, "optuna_history.png")
        importance_plot_path = os.path.join(base_config.OUTPUT_DIR, "optuna_importance.png")

        os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(history_plot_path)
        print(f"Optimization history plot saved to: {history_plot_path}")

        # Check if enough completed trials and diverse parameters exist for importance plot
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 1 and len(study.best_params) > 1: # Need >1 trial and >1 parameter varied
            try:
                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_image(importance_plot_path)
                print(f"Parameter importance plot saved to: {importance_plot_path}")
            except Exception as plot_err:
                 # More specific check for importance plot issues
                if "contains only one parameter" in str(plot_err) or "must contain more than one" in str(plot_err):
                    print("\nSkipping parameter importance plot (requires multiple parameters).")
                else:
                    print(f"\nCould not generate parameter importance plot: {plot_err}")
        else:
             print("Skipping parameter importance plot (not enough completed trials or parameters varied).")


    except Exception as e:
        print(f"\nCould not generate or save plots: {e}")

    print("\nOptimization script complete.")