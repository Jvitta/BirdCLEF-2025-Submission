import optuna
import pandas as pd
import copy
import sys
import os
import logging
import matplotlib
import plotly
import shutil # Add shutil for file copying

# Dynamically add attributes to config class if they don't exist
# This avoids needing to modify config.py immediately for HPO runs
def ensure_cfg_attrs(cfg, attrs):
    for attr in attrs:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, None) # Set to None initially

from config import config as base_config
# Ensure base_config has the necessary attributes HPO will modify
ensure_cfg_attrs(base_config, [
    'time_mask_prob', 'freq_mask_prob', 'contrast_prob',
    'max_time_mask_width', 'max_freq_mask_height'
])

from birdclef_training import run_training, set_seed, calculate_auc # Import calculate_auc

# load data once for HPO
print("Loading main training metadata for HPO...")
try:
    main_train_df = pd.read_csv(base_config.train_csv_path)

except FileNotFoundError:
    print(f"Error: Main training CSV not found at {base_config.train_csv_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading main training CSV: {e}. Exiting.")
    sys.exit(1)

def objective(trial):
    """Runs one training trial with hyperparameters suggested by Optuna."""
    global main_train_df

    cfg = copy.deepcopy(base_config)

    # --- Hyperparameter Suggestions ---
    cfg.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # Keep optimizer fixed for this run
    # cfg.optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.7) # Tune mixup alpha
    # Keep scheduler fixed for this run
    # cfg.scheduler = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"])

    # Granular augmentation parameters
    # Note: apply_spec_augmentations in birdclef_training.py needs to be updated
    # to use these config values instead of hardcoded probabilities/sizes.
    # Remove the overarching cfg.aug_prob suggestion
    cfg.time_mask_prob = trial.suggest_float("time_mask_prob", 0.0, 0.8)
    cfg.freq_mask_prob = trial.suggest_float("freq_mask_prob", 0.0, 0.8)
    cfg.contrast_prob = trial.suggest_float("contrast_prob", 0.0, 0.8)
    cfg.max_time_mask_width = trial.suggest_int("max_time_mask_width", 5, 40)
    cfg.max_freq_mask_height = trial.suggest_int("max_freq_mask_height", 5, 30)
    # --- End Hyperparameter Suggestions ---

    # --- Trial Configuration ---
    cfg.epochs = 15 # Set epochs to 15 for HPO trials
    print(f"INFO: Running HPO trial with {cfg.epochs} epochs.")

    cfg.n_fold = 5 # Use full 5 folds for HPO evaluation
    cfg.selected_folds = [0, 1, 2, 3, 4]
    print(f"INFO: Running HPO trial with folds {cfg.selected_folds}")
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
        # Pass the trial object for potential pruning callback within run_training if implemented later
        # Currently, pruning is handled by Optuna based on the return value
        mean_oof_auc = run_training(main_train_df, cfg) # Removed trial=trial argument for now

        # Handle None or invalid return values robustly
        if mean_oof_auc is None or not isinstance(mean_oof_auc, (int, float)) or pd.isna(mean_oof_auc) or not abs(mean_oof_auc) > 0:
             print(f"Warning: Trial {trial.number} resulted in invalid AUC ({mean_oof_auc}). Reporting as 0.0.")
             mean_oof_auc = 0.0
        else:
             print(f"\n--- Finished Optuna Trial {trial.number} | Mean OOF AUC: {mean_oof_auc:.4f} ---")

        # --- Optuna Pruning Check ---
        # Inform Optuna about the intermediate value (mean_oof_auc after full run)
        # This might not be strictly necessary if run_training doesn't report intermediate values,
        # but good practice if it ever does. Pruning is mainly based on MedianPruner config.
        trial.report(mean_oof_auc, step=cfg.epochs) # Report final value at last step

        # Check if trial should be pruned based on reported value
        if trial.should_prune():
            raise optuna.TrialPruned()
        # --- End Optuna Pruning Check ---

    except optuna.TrialPruned:
        print(f"--- Optuna Trial {trial.number} was pruned ---")
        # Ensure pruned trials return a value Optuna understands (like 0 for maximization)
        # Although raising TrialPruned handles it, explicitly returning can be clearer.
        return 0.0 # Return 0.0 for pruned trial in maximization

    except Exception as e:
        print(f"\n--- Non-pruning Exception occurred during Optuna trial {trial.number} ---")
        import traceback
        traceback.print_exc()
        print(f"--- End of traceback for trial {trial.number} ---")
        mean_oof_auc = 0.0 # Report 0.0 AUC for failed trials

    # Clean up trial-specific model directory (optional)
    # try:
    #     shutil.rmtree(trial_model_dir)
    # except OSError as e:
    #     print(f"Warning: Could not remove trial directory {trial_model_dir}: {e}")

    return mean_oof_auc

if __name__ == "__main__":
    # --- Study Configuration ---
    study_name = "BirdCLEF_HPO_GCP_Augmentation_LR_WD" # Updated study name
    n_trials = 50 # Start with 50 trials

    # --- Define paths for GCP --- #
    # Database will be stored in the OUTPUT_DIR defined in config.py
    db_filename = "hpo_study_results.db"
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
    hpo_trial_epochs = 15 # Updated HPO epochs
    hpo_trial_folds = [0, 1, 2, 3, 4] # Updated HPO folds

    print(f"\n--- Starting Optuna Optimization ---")
    print(f"Study Name: {study_name}")
    print(f"Number of Trials: {n_trials}")
    print(f"Storage (using): {storage_path_to_use}")
    print(f"Metric to Optimize: Mean AUC over folds {hpo_trial_folds} (Maximizing)")
    print(f"Base Config Epochs (for reference): {base_config.epochs}")
    print(f"HPO Trial Epochs: {hpo_trial_epochs}")
    print(f"HPO Trial Folds: {hpo_trial_folds}")

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
    study.set_user_attr("hpo_trial_folds", hpo_trial_folds)
    # Updated list of tuned parameters
    tuned_params_list = [
        "lr", "weight_decay", "mixup_alpha",
        "time_mask_prob", "freq_mask_prob", "contrast_prob",
        "max_time_mask_width", "max_freq_mask_height"
    ]
    study.set_user_attr("tuned_parameters", tuned_params_list)
    # Store fixed parameters for reference
    study.set_user_attr("fixed_optimizer", base_config.optimizer)
    study.set_user_attr("fixed_scheduler", base_config.scheduler)


    try:
        study.optimize(objective, n_trials=n_trials, timeout=None)
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
            print(f"  Value (Mean OOF AUC): {best_trial.value:.5f}")
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