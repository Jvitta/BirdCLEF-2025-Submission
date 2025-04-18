import optuna
import pandas as pd
import copy
import sys
import os
import logging
import matplotlib
import plotly
import shutil # Add shutil for file copying

from config import config as base_config
from birdclef_training import run_training, set_seed 

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

    cfg.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam"]) 
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.5) 
    cfg.scheduler = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"]) 
    cfg.aug_prob = trial.suggest_float("aug_prob", 0.3, 0.8)

    cfg.epochs = 5
    print(f"INFO: Running HPO trial with {cfg.epochs} epochs.")

    cfg.n_fold = 2 
    cfg.selected_folds = [0, 1] 
    print(f"INFO: Running HPO trial with folds {cfg.selected_folds}")

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
        mean_oof_auc = run_training(main_train_df, cfg, trial=trial)

        if mean_oof_auc is None or not isinstance(mean_oof_auc, (int, float)) or not abs(mean_oof_auc) > 0:
             print(f"Warning: Trial {trial.number} resulted in invalid AUC ({mean_oof_auc}). Reporting as 0.0.")
             mean_oof_auc = 0.0 
        print(f"\n--- Finished Optuna Trial {trial.number} | Mean OOF AUC: {mean_oof_auc:.4f} ---")

    except optuna.TrialPruned:
        print(f"--- Optuna Trial {trial.number} was pruned ---") 
        mean_oof_auc = 0.0 

    except Exception as e:
        print(f"\n--- Non-pruning Exception occurred during Optuna trial {trial.number} ---")
        import traceback
        traceback.print_exc()
        print(f"--- End of traceback for trial {trial.number} ---")
        mean_oof_auc = 0.0

    return mean_oof_auc

if __name__ == "__main__":
    # --- Study Configuration ---
    study_name = "BirdCLEF_HPO_OptimizeLR_WD_Mixup_Scheduler"
    n_trials = 20

    # --- Define paths --- #
    # Path to the *original* database in the input dataset
    # *** IMPORTANT: Ensure 'hpo-results' matches your input dataset name ***
    previous_output_slug = "hpo-results"
    input_db_path = f"/kaggle/input/{previous_output_slug}/hpo_results.db"

    # Path to where we *will* work with the database (writable)
    working_db_name = "hpo_results_writable.db"
    storage_path_working = f"sqlite:////kaggle/working/{working_db_name}" # Absolute path for Optuna
    working_db_file_path = f"/kaggle/working/{working_db_name}" # Filesystem path for copy

    # --- Copy database from input to working directory --- #
    storage_path_to_use = storage_path_working # Default to working path
    if os.path.exists(input_db_path):
        try:
            print(f"Copying database from {input_db_path} to {working_db_file_path}...")
            shutil.copy2(input_db_path, working_db_file_path)
            print("Database copied successfully.")
            # Use the copied path for the study
            storage_path_to_use = storage_path_working
        except Exception as e:
            print(f"ERROR: Could not copy database file: {e}")
            print(f"Optuna will attempt to use/create database at {storage_path_working}")
            # Fallback to using/creating a new db in working dir if copy fails
            storage_path_to_use = storage_path_working
    else:
        print(f"Warning: Input database file not found at {input_db_path}.")
        print(f"Optuna will create a new database at {storage_path_working}")
        # If the input db doesn't exist, create a new one in working dir
        storage_path_to_use = storage_path_working

    # Define HPO settings ...
    hpo_trial_epochs = 5
    hpo_trial_folds = [0, 1]

    print(f"\n--- Starting Optuna Optimization ---")
    print(f"Study Name: {study_name}")
    print(f"Number of Trials: {n_trials}")
    # Print the path Optuna will ACTUALLY use
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        storage=storage_path_to_use, # Use the path in /kaggle/working/
        load_if_exists=True # Load if the *copied* file exists
    )

    # Add base config parameters as user attributes for reference
    study.set_user_attr("base_config_epochs", base_config.epochs)
    study.set_user_attr("hpo_trial_epochs", hpo_trial_epochs)
    study.set_user_attr("hpo_trial_folds", hpo_trial_folds)
    study.set_user_attr("tuned_parameters", ["lr", "optimizer", "weight_decay", "mixup_alpha", "scheduler", "aug_prob"])

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
    # Use the variable holding the *working* storage path
    print(f"Storage (Working Copy): {storage_path_to_use}")
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Mean OOF AUC): {best_trial.value:.5f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print(f"  Trial Number: {best_trial.number}")
    except ValueError:
        print("\nNo completed trials found. Cannot determine the best trial.")

    try:
        history_plot_path = os.path.join(base_config.OUTPUT_DIR, "optuna_history.png")
        importance_plot_path = os.path.join(base_config.OUTPUT_DIR, "optuna_importance.png")

        os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_image(history_plot_path)
        print(f"Optimization history plot saved to: {history_plot_path}")

        if len(study.trials) > 1 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            try:
                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_image(importance_plot_path)
                print(f"Parameter importance plot saved to: {importance_plot_path}")
            except Exception as plot_err:
                print(f"\nCould not generate parameter importance plot: {plot_err}")
        else:
            print("Skipping parameter importance plot (not enough completed trials or parameters).")

    except Exception as e:
        print(f"\nCould not generate or save plots: {e}")

    print("\nOptimization script complete.") 