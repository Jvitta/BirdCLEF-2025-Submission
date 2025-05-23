import optuna
import pandas as pd
import copy
import sys
import os
import shutil
import numpy as np
from tqdm.auto import tqdm
import time
import functools
import gc # For garbage collection

from config import config as base_config
from src.training.train_mn import run_training

# --- Global Data Loading (once for all HPO trials) ---
print("--- Loading DataFrames for HPO ---")
main_df_global = pd.DataFrame() # Renamed to avoid clash if any local main_df exists
pseudo_df_global = None

# Load Main Training Metadata
print("Loading main training metadata...")
try:
    main_train_df_full = pd.read_csv(base_config.train_csv_path)
    main_train_df_full['filepath'] = main_train_df_full['filename'].apply(
        lambda f: os.path.join(base_config.train_audio_dir, f)
    )
    main_train_df_full['samplename'] = main_train_df_full.filename.map(
        lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0]
    )
    main_train_df_full['data_source'] = 'main'
    
    required_cols_main = [ # These are the columns expected by run_training after its internal processing
        'filename', 'primary_label', 'secondary_labels', 'filepath', 'samplename', 'data_source',
        'latitude', 'longitude', 'author' 
    ]
    # Ensure all these columns exist in the loaded CSV for main_df
    for col in required_cols_main:
        if col not in main_train_df_full.columns:
            if col == 'secondary_labels': # provide default if missing
                 main_train_df_full[col] = [[] for _ in range(len(main_train_df_full))]
            elif col in ['latitude', 'longitude', 'author']: # provide default if missing
                 main_train_df_full[col] = np.nan
            else: # For other critical columns like filename, primary_label, error out
                print(f"CRITICAL ERROR: Required column '{col}' missing from {base_config.train_csv_path}. Exiting.")
                sys.exit(1)

    main_df_global = main_train_df_full[required_cols_main].copy()
    print(f"Loaded {len(main_df_global)} main training samples from {base_config.train_csv_path}.")
    del main_train_df_full; gc.collect()
except Exception as e:
    print(f"CRITICAL ERROR loading main training CSV {base_config.train_csv_path}: {e}. Exiting.")
    sys.exit(1)

# Load Rare Species Data if enabled
if base_config.USE_RARE_DATA:
    print("\nLoading Rare Species Data...")
    try:
        rare_train_df_full = pd.read_csv(base_config.train_rare_csv_path)
        rare_train_df_full['filepath'] = rare_train_df_full['filename'].apply(
            lambda f: os.path.join(base_config.train_audio_rare_dir, f)
        )
        rare_train_df_full['samplename'] = rare_train_df_full.filename.map(
            lambda x: str(x.split('/')[0]) + '-' + str(x.split('/')[-1].split('.')[0])
        )
        rare_train_df_full['data_source'] = 'main' # Consistent with how train_mn.py treats it
        
        # Ensure all required_cols_main exist, adding defaults if missing
        for col in required_cols_main:
            if col not in rare_train_df_full.columns:
                if col == 'secondary_labels':
                     rare_train_df_full[col] = [[] for _ in range(len(rare_train_df_full))]
                elif col in ['latitude', 'longitude', 'author']:
                     rare_train_df_full[col] = np.nan
                # No need to error for filename/primary_label here as train_rare.csv should have them

        rare_df = rare_train_df_full[required_cols_main].copy()
        print(f"Loaded {len(rare_df)} rare training samples from {base_config.train_rare_csv_path}.")
        main_df_global = pd.concat([main_df_global, rare_df], ignore_index=True)
        print(f"Combined main_df size (main + rare): {len(main_df_global)} samples.")
        del rare_train_df_full, rare_df; gc.collect()
    except Exception as e:
        print(f"ERROR loading or processing rare labels CSV {base_config.train_rare_csv_path}: {e}")
        # Decide if this is critical enough to exit; for HPO, maybe continue if main_df is not empty
        if main_df_global.empty:
            print("Exiting as main_df is empty after rare data loading error.")
            sys.exit(1)

# Load Pseudo-Label Metadata if enabled
if base_config.USE_PSEUDO_LABELS:
    print("\nLoading Pseudo-Label Metadata...")
    try:
        pseudo_labels_df_full = pd.read_csv(base_config.soundscape_pseudo_calibrated_csv_path)
        pseudo_labels_df_full['samplename'] = pseudo_labels_df_full.apply(
            lambda row: f"{row['filename']}_{int(row['start_time'])}_{int(row['end_time'])}", axis=1
        )
        pseudo_labels_df_full['filepath'] = pseudo_labels_df_full['filename'].apply(
            lambda f: os.path.join(base_config.unlabeled_audio_dir, f)
        )
        pseudo_labels_df_full['data_source'] = 'pseudo'
        
        original_pseudo_rows = len(pseudo_labels_df_full)
        pseudo_labels_df_full = pseudo_labels_df_full.drop_duplicates(subset=['samplename'], keep='first').reset_index(drop=True)
        deduplicated_count = original_pseudo_rows - len(pseudo_labels_df_full)
        if deduplicated_count > 0:
            print(f"  Deduplicated pseudo_labels_df: Removed {deduplicated_count} rows with duplicate 'samplename's.")
        print(f"  Size of pseudo_labels_df after deduplication: {len(pseudo_labels_df_full)}")

        # Add potentially missing columns from required_cols_main for structural consistency
        # (e.g., lat, lon, author, secondary_labels which are NaN/empty list in train_mn.py for pseudo)
        for col in required_cols_main:
            if col not in pseudo_labels_df_full.columns:
                if col == 'secondary_labels':
                    pseudo_labels_df_full[col] = [[] for _ in range(len(pseudo_labels_df_full))]
                elif col in ['latitude', 'longitude', 'author']:
                    pseudo_labels_df_full[col] = np.nan
                # Other cols like 'filename', 'primary_label' are expected to be in pseudo_labels_df_full
        
        # Ensure all columns expected by run_training are present in pseudo_df.
        # These are basically the columns in main_df.
        pseudo_df_global = pseudo_labels_df_full[required_cols_main].copy()
        
        # Add 'confidence' if it exists in pseudo_labels_df_full and is not already in required_cols_main
        if 'confidence' in pseudo_labels_df_full.columns and 'confidence' not in pseudo_df_global.columns:
            pseudo_df_global['confidence'] = pseudo_labels_df_full['confidence']

        print(f"Loaded and prepared {len(pseudo_df_global)} pseudo-label samples.")
        del pseudo_labels_df_full; gc.collect()
    except Exception as e:
        print(f"ERROR loading or processing pseudo labels CSV {base_config.soundscape_pseudo_calibrated_csv_path}: {e}")
        pseudo_df_global = pd.DataFrame() # Ensure pseudo_df is an empty DataFrame
# --- End Global Data Loading ---


def objective(trial, main_df_obj, pseudo_df_obj, 
              main_train_specs_obj, main_val_specs_obj, 
              pseudo_train_specs_obj, pseudo_val_specs_obj):
    """Runs one training trial with hyperparameters suggested by Optuna."""
    cfg = copy.deepcopy(base_config)

    # --- Hyperparameter Suggestions --- #
    cfg.lr = trial.suggest_float("lr", 7e-4, 1.5e-3, log=True) 
    cfg.weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-4, log=True) 
    cfg.dropout = trial.suggest_float("dropout", 0.40, 0.60)
    cfg.label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.10, 0.25)
    cfg.mixup_alpha = trial.suggest_float("mixup_alpha", 0.15, 0.35)
    cfg.focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 2.5, 4.0)
    cfg.focal_loss_bce_weight = trial.suggest_float("focal_loss_bce_weight", 0.20, 0.80)
    cfg.T_max = trial.suggest_int("T_max_value", cfg.epochs, int(cfg.epochs * 1.5)) # cfg.epochs is hpo_epochs_per_fold
    # --- End Hyperparameter Suggestions ---

    hpo_epochs_per_fold = base_config.epochs 
    cfg.epochs = hpo_epochs_per_fold
    print(f"INFO: Running HPO trial with max {cfg.epochs} epochs per fold.")

    trial_model_dir_base = os.path.join(base_config.MODEL_OUTPUT_DIR, f"hpo_trial_{trial.number}")
    os.makedirs(trial_model_dir_base, exist_ok=True)

    cfg.debug = False # Ensure debug is off for HPO
    # cfg.debug_limit_batches = float('inf') # Not needed if debug is False

    trial_seed = base_config.seed + trial.number # Vary seed per trial
    cfg.seed = trial_seed 

    print(f"\n--- Starting Optuna Trial {trial.number} (Overall Seed: {cfg.seed}) ---")
    print(f"Fixed spectrogram augs: time_mask_prob={cfg.time_mask_prob}, freq_mask_prob={cfg.freq_mask_prob}, "
          f"max_time_mask_width={cfg.max_time_mask_width}, max_freq_mask_height={cfg.max_freq_mask_height}")
    print(f"Parameters: {trial.params}")

    fold_aucs = []
    folds_to_run_hpo = [0, 1] # Example: run on Fold 0 and Fold 1 for HPO

    try:
        for i, fold_num in enumerate(folds_to_run_hpo):
            print(f"\n--- Trial {trial.number}, Processing Fold {fold_num} ---")
            cfg_fold = copy.deepcopy(cfg) 
            cfg_fold.selected_folds = [fold_num] # Run only this fold
            
            cfg_fold.MODEL_OUTPUT_DIR = os.path.join(trial_model_dir_base, f"fold_{fold_num}")
            os.makedirs(cfg_fold.MODEL_OUTPUT_DIR, exist_ok=True)

            current_hpo_step_offset = i * hpo_epochs_per_fold

            # For HPO, we pass copies of the globally loaded DataFrames
            # The spectrogram dicts can be passed directly as they are not modified in run_training
            current_main_df = main_df_obj.copy() if main_df_obj is not None else pd.DataFrame()
            current_pseudo_df = pseudo_df_obj.copy() if pseudo_df_obj is not None else None
            
            fold_best_auc = run_training(
                main_df=current_main_df, 
                pseudo_df=current_pseudo_df,
                config=cfg_fold,
                main_train_spectrograms=main_train_specs_obj,
                main_val_spectrograms=main_val_specs_obj,
                pseudo_train_spectrograms=pseudo_train_specs_obj,
                pseudo_val_spectrograms=pseudo_val_specs_obj,
                trial=trial, 
                custom_run_name=f"hpo_trial_{trial.number}_fold{fold_num}",
                hpo_step_offset=current_hpo_step_offset
            )

            if fold_best_auc is None or not isinstance(fold_best_auc, (int, float)) or pd.isna(fold_best_auc):
                 print(f"Warning: Trial {trial.number}, Fold {fold_num} resulted in invalid AUC ({fold_best_auc}). Appending 0.0.")
                 fold_aucs.append(0.0)
            else:
                 print(f"--- Trial {trial.number}, Fold {fold_num} | Best Val AUC for Fold: {fold_best_auc:.4f} ---")
                 fold_aucs.append(fold_best_auc)
        
        final_objective_value = np.mean(fold_aucs) if fold_aucs else 0.0
        print(f"\n--- Finished Optuna Trial {trial.number} | Average Val AUC (Folds {folds_to_run_hpo}): {final_objective_value:.4f} ---")

    except optuna.TrialPruned:
        print(f"--- Optuna Trial {trial.number} was pruned during one of its folds ---")
        raise 
    except Exception as e:
        print(f"\n--- Non-pruning Exception occurred during Optuna trial {trial.number} ---")
        import traceback
        traceback.print_exc()
        print(f"--- End of traceback for trial {trial.number} ---")
        final_objective_value = 0.0 
    
    # Clean up trial-specific model directory to save space
    try:
        shutil.rmtree(trial_model_dir_base)
        # print(f"Removed trial base directory: {trial_model_dir_base}") # Optional: for verbose logging
    except OSError as e:
        print(f"Warning: Could not remove trial base directory {trial_model_dir_base}: {e}")

    return final_objective_value

if __name__ == "__main__":
    study_name = "BirdCLEF_HPO_UnifiedCV_V1" 
    n_trials = 250 
    db_filename = "hpo_unifiedcv_v1.db" 
    db_filepath = os.path.join(base_config.OUTPUT_DIR, db_filename)
    storage_path = f"sqlite:///{db_filepath}"
    storage_path_to_use = storage_path
    print(f"Optuna database path: {db_filepath}")
    if os.path.exists(db_filepath):
        print("Existing database found. Optuna will load study if it exists.")
    else:
        print("No existing database found. Optuna will create a new one.")
        os.makedirs(base_config.OUTPUT_DIR, exist_ok=True) # Ensure output dir exists for DB

    hpo_trial_epochs = base_config.epochs # Max epochs per fold in HPO
    print(f"\n--- Starting Optuna Optimization --- ")
    print(f"Study Name: {study_name}")
    print(f"Number of Trials: {n_trials}")
    print(f"Storage (using): {storage_path_to_use}")
    print(f"Metric to Optimize: Average Validation AUC over folds { [0, 1] } (Maximizing)") # Updated metric desc
    print(f"HPO Trial Epochs (per fold): {hpo_trial_epochs}")

    # Pruner setup
    warmup_steps_for_pruner = hpo_trial_epochs // 2 if hpo_trial_epochs > 1 else 0

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=base_config.seed), # Use a consistent seed for TPE
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup_steps_for_pruner, interval_steps=1),
        storage=storage_path_to_use,
        load_if_exists=True
    )

    study.set_user_attr("base_config_epochs_per_fold", base_config.epochs) # For clarity
    study.set_user_attr("hpo_folds_run_per_trial", [0, 1]) 
    tuned_params_list = ["lr", "weight_decay", "dropout", 
                         "label_smoothing_factor", "mixup_alpha", 
                         "focal_loss_gamma", "focal_loss_bce_weight", "T_max_value"]
    study.set_user_attr("tuned_parameters", tuned_params_list)
    study.set_user_attr("fixed_optimizer", base_config.optimizer)
    study.set_user_attr("fixed_scheduler", base_config.scheduler)
    
    # Log base config values for fixed parameters for reference
    fixed_params_log = {
        "time_mask_prob": base_config.time_mask_prob,
        "freq_mask_prob": base_config.freq_mask_prob,
        "max_time_mask_width": base_config.max_time_mask_width,
        "max_freq_mask_height": base_config.max_freq_mask_height,
        "original_dropout": base_config.dropout, # Base value before HPO tunes it
        "original_lr": base_config.lr, # Base value before HPO tunes it
        "original_weight_decay": base_config.weight_decay, # Base value
        "original_label_smoothing_factor": base_config.label_smoothing_factor,
        "original_mixup_alpha": base_config.mixup_alpha,
        "original_focal_loss_gamma": base_config.focal_loss_gamma,
        "original_focal_loss_bce_weight": base_config.focal_loss_bce_weight,
        "original_T_max": base_config.T_max
    }
    study.set_user_attr("base_config_fixed_parameter_references", fixed_params_log)

    # --- Spectrogram Loading (New Logic) ---
    print("\n--- Loading Preprocessed Spectrograms for HPO ---")
    main_train_specs_global, main_val_specs_global = {}, {}
    pseudo_train_specs_global, pseudo_val_specs_global = {}, {}

    main_train_npz_path = os.path.join(base_config._PREPROCESSED_OUTPUT_DIR, f"spectrograms_{base_config.model_name}_train.npz")
    main_val_npz_path = os.path.join(base_config._PREPROCESSED_OUTPUT_DIR, f"spectrograms_{base_config.model_name}_val.npz")
    
    print(f"Attempting to load MAIN TRAIN spectrograms from: {main_train_npz_path}")
    try:
        with np.load(main_train_npz_path) as data_archive:
            main_train_specs_global = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Main Train Specs (HPO)")}
        print(f"Successfully loaded {len(main_train_specs_global)} main train spectrograms for HPO.")
    except Exception as e:
        print(f"CRITICAL ERROR loading main train NPZ for HPO ({main_train_npz_path}): {e}. Exiting.")
        sys.exit(1)

    print(f"Attempting to load MAIN VAL spectrograms from: {main_val_npz_path}")
    try:
        with np.load(main_val_npz_path) as data_archive:
            main_val_specs_global = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Main Val Specs (HPO)")}
        print(f"Successfully loaded {len(main_val_specs_global)} main val spectrograms for HPO.")
    except Exception as e:
        print(f"CRITICAL ERROR loading main val NPZ for HPO ({main_val_npz_path}): {e}. Exiting.")
        sys.exit(1)

    if base_config.USE_PSEUDO_LABELS:
        pseudo_train_npz_path = os.path.join(base_config._PREPROCESSED_OUTPUT_DIR, f"pseudo_spectrograms_{base_config.model_name}_train.npz")
        pseudo_val_npz_path = os.path.join(base_config._PREPROCESSED_OUTPUT_DIR, f"pseudo_spectrograms_{base_config.model_name}_val.npz")

        print(f"Attempting to load PSEUDO TRAIN spectrograms from: {pseudo_train_npz_path}")
        try:
            with np.load(pseudo_train_npz_path) as data_archive:
                pseudo_train_specs_global = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Pseudo Train Specs (HPO)")}
            print(f"Successfully loaded {len(pseudo_train_specs_global)} pseudo train spectrograms for HPO.")
        except Exception as e:
            print(f"ERROR loading pseudo train NPZ for HPO ({pseudo_train_npz_path}): {e}. Pseudo train specs will be empty.")
            # Consider exiting if pseudo labels are critical and failed to load
            # sys.exit(1)

        print(f"Attempting to load PSEUDO VAL spectrograms from: {pseudo_val_npz_path}")
        try:
            with np.load(pseudo_val_npz_path) as data_archive:
                pseudo_val_specs_global = {key: data_archive[key] for key in tqdm(data_archive.keys(), desc="Loading Pseudo Val Specs (HPO)")}
            print(f"Successfully loaded {len(pseudo_val_specs_global)} pseudo val spectrograms for HPO.")
        except Exception as e:
            print(f"ERROR loading pseudo val NPZ for HPO ({pseudo_val_npz_path}): {e}. Pseudo val specs will be empty.")
            # sys.exit(1)
    else: # Ensure these are empty dicts if not using pseudo labels
        pseudo_train_specs_global = {}
        pseudo_val_specs_global = {}
    # --- End Spectrogram Loading ---

    # --- DataFrame Filtering (after all data is loaded) ---
    print("\n--- Filtering DataFrames by TRAIN Spectrogram Availability (HPO Preload) ---")
    if main_df_global.empty:
        print("CRITICAL ERROR: main_df_global is empty before filtering. Check initial loading. Exiting.")
        sys.exit(1)
        
    original_main_count = len(main_df_global)
    if main_train_specs_global: 
        loaded_keys_main = set(main_train_specs_global.keys())
        main_df_global = main_df_global[main_df_global['samplename'].isin(loaded_keys_main)].reset_index(drop=True)
        removed_main_count = original_main_count - len(main_df_global)
        if removed_main_count > 0:
            print(f"  Filtered main_df_global (HPO): Removed {removed_main_count} samples with missing train specs.")
    else: # Should not happen if NPZ loading is critical
        print("Warning: main_train_specs_global is empty, cannot filter main_df_global.")
    print(f"  Final main_df_global size for HPO: {len(main_df_global)} samples.")
    if main_df_global.empty: # Critical if main_df becomes empty after filtering
        print("CRITICAL ERROR: main_df_global became empty after filtering. No main training data available. Exiting.")
        sys.exit(1)

    if base_config.USE_PSEUDO_LABELS and pseudo_df_global is not None and not pseudo_df_global.empty:
        original_pseudo_count = len(pseudo_df_global)
        if pseudo_train_specs_global:
            loaded_keys_pseudo = set(pseudo_train_specs_global.keys())
            pseudo_df_global = pseudo_df_global[pseudo_df_global['samplename'].isin(loaded_keys_pseudo)].reset_index(drop=True)
            removed_pseudo_count = original_pseudo_count - len(pseudo_df_global)
            if removed_pseudo_count > 0:
                print(f"  Filtered pseudo_df_global (HPO): Removed {removed_pseudo_count} samples with missing train specs.")
        else: # If pseudo_train_specs_global is empty but we expected pseudo labels
            print("Warning: pseudo_train_specs_global is empty. If USE_PSEUDO_LABELS is True, this means no pseudo specs were loaded or found.")
            if not pseudo_df_global.empty: # If there was pseudo metadata, it implies a loading issue for specs
                 print("           This might lead to pseudo_df_global becoming empty or issues during training if pseudo data is expected.")
                 pseudo_df_global = pd.DataFrame() # Make it empty if specs are missing

        print(f"  Final pseudo_df_global size for HPO: {len(pseudo_df_global)} samples.")
    elif base_config.USE_PSEUDO_LABELS:
        print("  Info: USE_PSEUDO_LABELS is True, but pseudo_df_global is None or empty before filtering.")
    # --- End DataFrame Filtering ---

    # Pass all loaded data via functools.partial
    # Ensure copies of DataFrames are passed to objective if they might be modified by cfg settings inside objective
    # Spectrogram dicts are large; pass them directly if run_training doesn't modify them (it shouldn't)
    objective_with_data = functools.partial(objective, 
                                            main_df_obj=main_df_global, 
                                            pseudo_df_obj=pseudo_df_global,
                                            main_train_specs_obj=main_train_specs_global,
                                            main_val_specs_obj=main_val_specs_global,
                                            pseudo_train_specs_obj=pseudo_train_specs_global,
                                            pseudo_val_specs_obj=pseudo_val_specs_global)
    try:
        study.optimize(objective_with_data, n_trials=n_trials, timeout=None)
    except KeyboardInterrupt:
        print("\nOptimization stopped manually. Saving current results.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Optuna Optimization Finished ---")
    print(f"Study Name: {study.study_name}")
    print(f"Storage: {storage_path_to_use}") # Use the actual storage path
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        if study.best_trial:
            best_trial = study.best_trial
            print("\nBest trial:")
            print(f"  Value (Best Avg Val AUC): {best_trial.value:.5f}") 
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            print(f"  Trial Number: {best_trial.number}")
        else:
            print("\nNo completed trials found or best trial unavailable.")
    except ValueError: 
        print("\nNo completed trials found. Cannot determine the best trial.")

    try:
        # Ensure unique filenames for plots based on study_name
        history_plot_path = os.path.join(base_config.OUTPUT_DIR, f"{study.study_name}_history.png")
        importance_plot_path = os.path.join(base_config.OUTPUT_DIR, f"{study.study_name}_importance.png")

        os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(history_plot_path)
        print(f"Optimization history plot saved to: {history_plot_path}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        # Check if there are parameters to plot for importance
        if len(completed_trials) > 1 and study.best_params: 
            try:
                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_image(importance_plot_path)
                print(f"Parameter importance plot saved to: {importance_plot_path}")
            except Exception as plot_err:
                # Broader check for common importance plot issues
                if "one parameter" in str(plot_err) or "Plotly" in str(plot_err) or not study.best_params:
                    print("\nSkipping parameter importance plot (requires multiple varied parameters or other Plotly/Optuna issue).")
                else:
                    print(f"\nCould not generate parameter importance plot: {plot_err}")
        else:
             print("Skipping parameter importance plot (not enough completed trials or no parameters varied).")
    except Exception as e:
        print(f"\nCould not generate or save plots: {e}")

    print("\nOptimization script complete.")