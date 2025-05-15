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

from birdclef_training import run_training, set_seed, calculate_auc # Import calculate_auc

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
    else:
        print("Error: 'filename' column not found in main training CSV. Cannot create 'samplename'. Exiting.")
        sys.exit(1)
    
    # Select only necessary columns to mimic training script setup
    required_cols_main = ['samplename', 'primary_label', 'secondary_labels', 'data_source'] # Add 'data_source'
    if not all(col in main_train_df_full.columns for col in required_cols_main):
        missing = [col for col in required_cols_main if col not in main_train_df_full.columns]
        print(f"Error: Required columns missing from main training CSV: {missing}. Exiting.")
        sys.exit(1)
    
    main_train_df = main_train_df_full[required_cols_main].copy()
    print(f"Successfully loaded and prepared {len(main_train_df)} samples for HPO.")
    del main_train_df_full # Free up memory

except FileNotFoundError:
    print(f"Error: Main training CSV not found at {base_config.train_csv_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading main training CSV: {e}. Exiting.")
    sys.exit(1)

def objective(trial, preloaded_data):
    """Runs one training trial with hyperparameters suggested by Optuna."""
    global main_train_df

    cfg = copy.deepcopy(base_config)

    # --- Determine Fold for this Trial ---
    # HPO will run on a single, fixed fold for stability (e.g., Fold 2)
    fixed_hpo_fold = 2
    # fold_to_run = trial.number % cfg.n_fold # Removed: Using a fixed fold
    print(f"INFO: HPO Trial {trial.number} will run on fixed Fold {fixed_hpo_fold}")
    cfg.selected_folds = [fixed_hpo_fold] # Configure training to run only this fold
    # --- End Fold Determination ---

    # --- Hyperparameter Suggestions --- #
    # Keep learning rate search tight around the known good value
    cfg.lr = trial.suggest_float("lr", 3e-4, 7e-4, log=True) 

    # Keep weight decay search tight around the known good value
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True) 

    # Focal Loss Hyperparameters (Removed based on user feedback for this HPO run)
    # cfg.focal_loss_alpha = trial.suggest_float("focal_loss_alpha", 0.05, 0.95) 
    # cfg.focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 0.5, 5.0) 
    # cfg.focal_loss_bce_weight = trial.suggest_float("focal_loss_bce_weight", 0.0, 2.0)

    # Max Frequency Mask Height (for 128 mel bins)
    cfg.max_freq_mask_height = trial.suggest_int("max_freq_mask_height", 10, 20)

    # Max Time Mask Width (for 500 time frames in preprocessed spec)
    cfg.max_time_mask_width = trial.suggest_int("max_time_mask_width", 30, 70)

    # Label Smoothing
    cfg.label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.0, 0.25)

    # Mixup Alpha
    cfg.mixup_alpha = trial.suggest_float("mixup_alpha", 0.3, 0.5)

    # Keep other augmentation probabilities fixed for this study, as their effect might be captured by mask sizes
    cfg.time_mask_prob = base_config.time_mask_prob
    cfg.freq_mask_prob = base_config.freq_mask_prob
    cfg.contrast_prob = base_config.contrast_prob
    # --- End Hyperparameter Suggestions ---

    # --- Trial Configuration ---
    cfg.epochs = base_config.epochs # Use the base config epochs for HPO trials
    print(f"INFO: Running HPO trial with {cfg.epochs} epochs.")

    # cfg.n_fold = 5 # Use full 5 folds for HPO evaluation
    print(f"INFO: Running HPO trial with fold {cfg.selected_folds}")
    # --- End Trial Configuration ---


    trial_model_dir = os.path.join(base_config.MODEL_OUTPUT_DIR, f"hpo_trial_{trial.number}")
    cfg.MODEL_OUTPUT_DIR = trial_model_dir
    os.makedirs(cfg.MODEL_OUTPUT_DIR, exist_ok=True)

    cfg.debug = False
    cfg.debug_limit_batches = float('inf')


    trial_seed = base_config.seed + trial.number
    set_seed(trial_seed)
    cfg.seed = trial_seed

    print(f"\n--- Starting Optuna Trial {trial.number} (Seed: {cfg.seed}) ---")
    print(f"Parameters: {trial.params}")

    try:
        # Pass the trial object for potential pruning callback within run_training
        # The training function should now handle pruning internally
        single_fold_best_auc = run_training(main_train_df, cfg, trial=trial, all_spectrograms=preloaded_data)

        # Handle None or invalid return values robustly
        if single_fold_best_auc is None or not isinstance(single_fold_best_auc, (int, float)) or pd.isna(single_fold_best_auc) or not abs(single_fold_best_auc) > 0:
             print(f"Warning: Trial {trial.number} (Fold {fixed_hpo_fold}) resulted in invalid AUC ({single_fold_best_auc}). Reporting as 0.0.")
             single_fold_best_auc = 0.0
        else:
             print(f"\n--- Finished Optuna Trial {trial.number} (Fold {fixed_hpo_fold}) | Best Val AUC for Fold: {single_fold_best_auc:.4f} ---")

    except optuna.TrialPruned:
        print(f"--- Optuna Trial {trial.number} (Fold {fixed_hpo_fold}) was pruned ---")
        raise # Re-raise the exception to signal pruning to Optuna

    except Exception as e:
        print(f"\n--- Non-pruning Exception occurred during Optuna trial {trial.number} ---")
        import traceback
        traceback.print_exc()
        print(f"--- End of traceback for trial {trial.number} ---")
        single_fold_best_auc = 0.0 # Report 0.0 AUC for failed trials

    # Clean up trial-specific model directory (Uncommented to save space)
    try:
        shutil.rmtree(trial_model_dir)
        print(f"Removed trial directory: {trial_model_dir}")
    except OSError as e:
        print(f"Warning: Could not remove trial directory {trial_model_dir}: {e}")

    return single_fold_best_auc

if __name__ == "__main__":
    # --- Study Configuration ---
    study_name = "BirdCLEF_HPO_Augmentations_LR_WD" # Focused study name
    n_trials = 200 # Adjust number of trials as needed for this focused study

    # --- Define paths for GCP --- #
    # Database will be stored in the OUTPUT_DIR defined in config.py
    db_filename = "hpo_augmentations_study_results.db" # Specific DB file
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

    # Create study using the storage path in the *working* directory
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        # Configure MedianPruner: Prune if value is worse than median after n_warmup_steps epochs
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
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
        "lr", "weight_decay", "max_freq_mask_height", "max_time_mask_width", "label_smoothing_factor", "mixup_alpha"
    ]
    study.set_user_attr("tuned_parameters", tuned_params_list)
    # Store fixed parameters for reference
    study.set_user_attr("fixed_optimizer", base_config.optimizer)
    study.set_user_attr("fixed_scheduler", base_config.scheduler)
    # study.set_user_attr("fixed_mixup_alpha", base_config.mixup_alpha) # mixup_alpha is tuned
    study.set_user_attr("base_config_mixup_alpha", base_config.mixup_alpha) # Log base value for context

    # Log base augmentation probabilities and contrast, as mask sizes are tuned
    study.set_user_attr("base_config_augmentations", {
        "time_mask_prob": base_config.time_mask_prob,
        "freq_mask_prob": base_config.freq_mask_prob,
        "contrast_prob": base_config.contrast_prob
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

    try:
        # Use functools.partial to pass fixed arguments (like loaded data) to the objective
        objective_with_data = functools.partial(objective, preloaded_data=global_all_spectrograms)
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