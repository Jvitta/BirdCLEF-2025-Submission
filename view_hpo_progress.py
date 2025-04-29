import optuna
import sys
from pathlib import Path
import os # Added for directory creation

# --- Configuration (MUST match optimize.py) ---
STUDY_NAME = "BirdCLEF_HPO_GCP_Augmentation_LR_WD" # The name of your study
OUTPUT_DIR = Path("./outputs")
DB_FILENAME = "hpo_study_results.db"
PLOT_SUBDIR = "hpo_live_plots" # Subdirectory for plots
# --- End Configuration ---

def main():
    """Loads the study from the database and saves relevant Optuna plots."""

    db_filepath = OUTPUT_DIR / DB_FILENAME
    storage_url = f"sqlite:///{db_filepath.resolve()}"

    # --- Create plot directory ---
    plot_dir = OUTPUT_DIR / PLOT_SUBDIR
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured plot directory exists: {plot_dir}")
    except OSError as e:
        print(f"Error creating plot directory {plot_dir}: {e}")
        sys.exit(1)
    # --- End create plot directory ---

    if not db_filepath.exists():
        print(f"Error: Database file not found at {db_filepath}")
        print("Make sure the optimize.py script has run at least one trial.")
        sys.exit(1)

    print(f"Loading study '{STUDY_NAME}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)
    except KeyError:
        print(f"Error: Study '{STUDY_NAME}' not found in the database.")
        print("Please ensure the STUDY_NAME matches the one in optimize.py.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the study: {e}")
        sys.exit(1)


    print(f"Loaded study with {len(study.trials)} trials.")

    # --- Print Top 5 Trials --- #
    print("\n--- Top 5 Trials (Best Validation AUC first) ---")
    try:
        # Get all completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Sort them by value in descending order (best first since we maximize)
        completed_trials.sort(key=lambda t: t.value, reverse=True)

        # Get the top 5 (or fewer if less than 5 completed)
        top_trials = completed_trials[:5] 

        if not top_trials:
            print("No completed trials found in the study yet.")
        else:
            for i, trial in enumerate(top_trials):
                print(f"  Rank {i+1}:")
                print(f"    Trial Number: {trial.number}")
                print(f"    Value (Val AUC): {trial.value:.5f}")
                print(f"    Params:")
                for key, value in trial.params.items():
                    # Format floats nicely, keep others as is
                    param_value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
                    print(f"      {key}: {param_value_str}")
                print("  " + "-"*20) # Separator
    except Exception as e:
        print(f"Could not retrieve or print top trials: {e}")
    # --- End Print Top 5 Trials --- #

    if not study.trials:
        print("No trials found in the study yet.")
        return

    print("Generating plots...")

    # Plot functions and filenames
    plot_functions = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        "slice": optuna.visualization.plot_slice,
    }

    for plot_name, plot_func in plot_functions.items():
        plot_filename = plot_dir / f"hpo_{plot_name}_live.png"
        print(f"  Generating {plot_name} plot...")
        try:
            fig = plot_func(study)

            # --- Customize parallel coordinate plot specifically ---
            if plot_name == "parallel_coordinate":
                if fig.data and hasattr(fig.data[0], 'line'):
                    # Use a colorscale with better contrast, e.g., Plasma (dark blue -> yellow)
                    fig.data[0].line.colorscale = 'Plasma'
                    print(f"    Applied 'Plasma' colorscale to parallel coordinate plot.")
            # --- End customization ---

            fig.write_image(plot_filename)
            print(f"    Plot saved to: {plot_filename}")
        except ValueError as e:
             # Common error for importance/parallel/slice if only 1 trial or no completed trials
             print(f"    Could not generate {plot_name} plot: {e}")
             if "contains no completed trials" in str(e).lower() or \
                "should contain more than one" in str(e).lower() or \
                "importance for study with no completed trials" in str(e).lower():
                print("     (This often requires multiple completed trials)")
             else:
                 print(f"     Skipping {plot_name} plot due to unexpected ValueError.")
        except Exception as e:
            print(f"    An unexpected error occurred during {plot_name} plot generation: {e}")
            print(f"     Skipping {plot_name} plot.")


if __name__ == "__main__":
    main() 