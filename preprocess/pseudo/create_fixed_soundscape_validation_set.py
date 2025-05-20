import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import numpy as np
from tqdm import tqdm # Added for progress bar
import multiprocessing
import random # For randomizing target validation file count

try:
    from skmultilearn.model_selection import IterativeStratification
    SKML_AVAILABLE = True
except ImportError:
    SKML_AVAILABLE = False
    print("Warning: scikit-multilearn is not installed. IterativeStratification will not be available. Falling back to simpler stratification.")

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Assuming src/data_handling is two levels down
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config

# Configuration for the script
RARE_SPECIES_SOUNDSCAPE_DETECTION_THRESHOLD = 10  # Species with <= this many detections are "rare"
VALIDATION_SET_SIZE_FRACTION = 0.20 # 20% of candidate files for validation
MIN_TARGET_VAL_FILES = 475
MAX_TARGET_VAL_FILES = 550
N_STRATIFICATION_TRIALS = 4000
OUTPUT_FILENAME_LIST_PATH = os.path.join(project_root, "data", "processed", "fixed_soundscape_validation_filenames.txt")
OUTPUT_TRAIN_POOL_FILENAME_LIST_PATH = os.path.join(project_root, "data", "processed", "soundscape_train_pool_filenames.txt")

def evaluate_sgkf_trial(args):
    """
    Worker function for multiprocessing. Evaluates StratifiedGroupKFold for one trial.
    Each trial internally runs n_sgkf_splits (usually 5) and returns the best among them.
    """
    trial_num, base_seed, df_soundscape_global, splittable_candidate_filenames_global, \
    min_target_val_files, max_target_val_files = args

    current_seed = base_seed + trial_num
    
    # Determine target number of validation files for THIS trial
    if not splittable_candidate_filenames_global:
        return (float('inf'), [], [], trial_num, -1, 0) # No files to split
        
    target_n_val_files_for_trial = random.randint(min_target_val_files, max_target_val_files)
    # Ensure target_n_val_files is not more than available splittable files
    target_n_val_files_for_trial = min(target_n_val_files_for_trial, len(splittable_candidate_filenames_global) -1) # -1 to ensure train set not empty
    if target_n_val_files_for_trial <=0:
        target_n_val_files_for_trial = 1 # Must have at least 1 val file if splittable_candidate_filenames_global is not empty

    current_trial_val_fraction = target_n_val_files_for_trial / len(splittable_candidate_filenames_global)
    if current_trial_val_fraction <= 0 or current_trial_val_fraction >= 1:
        # Fallback if fraction is bad, aim for roughly 20% for n_splits calculation
        current_trial_val_fraction = 0.2 
    
    n_sgkf_splits = max(2, int(np.ceil(1.0 / current_trial_val_fraction)))
    current_target_val_percentage = current_trial_val_fraction * 100

    df_detections_for_sgkf = df_soundscape_global[df_soundscape_global['filename'].isin(splittable_candidate_filenames_global)].copy()
    if df_detections_for_sgkf.empty:
        return (float('inf'), [], [], trial_num, -1, 0) # No detections to stratify

    sgkf_total_detections_per_species = df_detections_for_sgkf['primary_label'].value_counts()
    if sgkf_total_detections_per_species.empty:
        return (float('inf'), [], [], trial_num, -1, 0) # No species to stratify by

    sgkf = StratifiedGroupKFold(n_splits=n_sgkf_splits, shuffle=True, random_state=current_seed)
    
    trial_best_fold_info = {'cost': float('inf'), 'val_filenames': [], 'train_filenames': [], 'fold_in_trial': -1, 'actual_n_val_files': 0}

    for fold_idx, (train_detection_indices, val_detection_indices) in enumerate(sgkf.split(df_detections_for_sgkf, df_detections_for_sgkf['primary_label'], df_detections_for_sgkf['filename'])):
        fold_val_detections = df_detections_for_sgkf.iloc[val_detection_indices]
        current_fold_val_filenames = sorted(list(fold_val_detections['filename'].unique()))
        actual_n_val_files_this_fold = len(current_fold_val_filenames)

        if not current_fold_val_filenames: continue

        fold_val_counts_per_species = fold_val_detections['primary_label'].value_counts()
        cost = 0

        for species, total_detections_in_pool in sgkf_total_detections_per_species.items():
            val_detections_species = fold_val_counts_per_species.get(species, 0)
            if total_detections_in_pool > 0:
                val_percentage_species = (val_detections_species / total_detections_in_pool) * 100
                cost += (val_percentage_species - current_target_val_percentage)**2
            elif val_detections_species > 0:
                cost += (current_target_val_percentage)**2
        
        if len(sgkf_total_detections_per_species) > 0:
            cost = cost / len(sgkf_total_detections_per_species)
        else:
            cost = float('inf')

        if cost < trial_best_fold_info['cost']:
            trial_best_fold_info['cost'] = cost
            trial_best_fold_info['val_filenames'] = current_fold_val_filenames
            trial_best_fold_info['train_filenames'] = sorted(list(set(splittable_candidate_filenames_global) - set(current_fold_val_filenames)))
            trial_best_fold_info['fold_in_trial'] = fold_idx
            trial_best_fold_info['actual_n_val_files'] = actual_n_val_files_this_fold
            
    return (
        trial_best_fold_info['cost'], 
        trial_best_fold_info['val_filenames'], 
        trial_best_fold_info['train_filenames'], 
        trial_num, 
        trial_best_fold_info['fold_in_trial'],
        trial_best_fold_info['actual_n_val_files']
    )

def create_fixed_soundscape_validation_set():
    """
    Creates a fixed validation set from soundscape pseudo-labels.
    Uses multi-trial StratifiedGroupKFold with dynamic validation sizes and best fold selection, with multiprocessing.
    """
    print("--- Creating Fixed Soundscape Validation Set ---")

    # Load soundscape pseudo-labels
    soundscape_path = config.soundscape_pseudo_calibrated_csv_path
    if not os.path.exists(soundscape_path):
        print(f"Error: Soundscape pseudo-label file not found at {soundscape_path}")
        return
    try:
        df_soundscape = pd.read_csv(soundscape_path)
        if not all(col in df_soundscape.columns for col in ['primary_label', 'filename']):
            print("Error: Soundscape CSV must contain 'primary_label' and 'filename'.")
            return
    except Exception as e:
        print(f"Error loading soundscape CSV: {e}")
        return
    print(f"Loaded {len(df_soundscape)} detections from {df_soundscape['filename'].nunique()} unique soundscape files.")

    # 1. Identify Rare Soundscape Species
    species_counts = df_soundscape['primary_label'].value_counts()
    rare_species_list = species_counts[species_counts <= RARE_SPECIES_SOUNDSCAPE_DETECTION_THRESHOLD].index.tolist()
    print(f"Identified {len(rare_species_list)} rare species (<= {RARE_SPECIES_SOUNDSCAPE_DETECTION_THRESHOLD} detections each).")
    detections_of_rare_species = df_soundscape[df_soundscape['primary_label'].isin(rare_species_list)]
    files_containing_rare_species = sorted(list(detections_of_rare_species['filename'].unique()))
    print(f"Found {len(files_containing_rare_species)} unique files containing at least one rare species. These will be added to the training pool.")

    # 2. Identify Candidate Files for Splitting
    all_soundscape_filenames = sorted(list(df_soundscape['filename'].unique()))
    candidate_filenames_initial = sorted(list(set(all_soundscape_filenames) - set(files_containing_rare_species)))
    if not candidate_filenames_initial:
        print("Error: No candidate files after excluding rare species. Cannot create validation set.")
        return

    # 3. Pre-filter: move files with unique heuristic stratification labels to training pool
    df_candidate_detections_for_heuristic = df_soundscape[df_soundscape['filename'].isin(candidate_filenames_initial)]
    candidate_species_counts_heuristic = df_candidate_detections_for_heuristic['primary_label'].value_counts()
    temp_stratify_labels_heuristic = []
    for fn_h in candidate_filenames_initial:
        detections_in_file_h = df_candidate_detections_for_heuristic[df_candidate_detections_for_heuristic['filename'] == fn_h]
        species_in_file_h = detections_in_file_h['primary_label'].unique()
        if not species_in_file_h.any(): 
            temp_stratify_labels_heuristic.append("NO_LABEL_FALLBACK_" + fn_h.replace(".ogg","")) 
            continue
        counts_of_species_in_this_file_h = candidate_species_counts_heuristic.reindex(species_in_file_h).fillna(float('inf'))
        rarest_species_in_file_h = counts_of_species_in_this_file_h.idxmin()
        temp_stratify_labels_heuristic.append(rarest_species_in_file_h)
    
    label_counts_heuristic = pd.Series(temp_stratify_labels_heuristic).value_counts()
    unstratifiable_labels_heuristic = label_counts_heuristic[label_counts_heuristic < 2].index.tolist()
    
    files_to_force_train_pool_heuristic = []
    splittable_candidate_filenames = [] # These are the final candidates for SGKF/IterativeStrat/etc.
    for fn_h, label_h in zip(candidate_filenames_initial, temp_stratify_labels_heuristic):
        if label_h in unstratifiable_labels_heuristic:
            files_to_force_train_pool_heuristic.append(fn_h)
        else:
            splittable_candidate_filenames.append(fn_h)
            
    if unstratifiable_labels_heuristic:
        print(f"Identified {len(files_to_force_train_pool_heuristic)} files with pre-filter heuristic labels appearing only once. Moved to training pool.")
    
    print(f"Have {len(splittable_candidate_filenames)} candidate files remaining for the main splitting process.")

    val_filenames_list = []
    train_pool_from_split_list = []
    sgkf_multitrial_success = False

    if not splittable_candidate_filenames:
        print("No splittable candidate files after heuristic pre-filtering. All candidates moved to training pool.")
        train_pool_from_split_list = list(candidate_filenames_initial) # All initial candidates go to train if none are splittable
    elif len(splittable_candidate_filenames) < max(2, MIN_TARGET_VAL_FILES / VALIDATION_SET_SIZE_FRACTION * 0.1): # Heuristic: if too few files to make a meaningful val set
        print(f"Only {len(splittable_candidate_filenames)} splittable files remaining, too few for robust splitting. Adding all to training pool.")
        train_pool_from_split_list = list(splittable_candidate_filenames)
    else:
        print(f"Attempting StratifiedGroupKFold + Best Fold Selection Strategy ({N_STRATIFICATION_TRIALS} trials with multiprocessing)...")
        overall_best_fold_info = {'cost': float('inf'), 'val_filenames': [], 'train_filenames': [], 'trial': -1, 'fold_in_trial': -1, 'actual_n_val_files': 0}
        
        # Prepare arguments for multiprocessing
        # Pass df_soundscape and splittable_candidate_filenames to avoid pickling large DF repeatedly if it were part of args for each call
        # However, for safety with multiprocessing, it's often better to ensure data passed is minimal or truly shared if possible.
        # For this case, df_soundscape is read once and then parts are used. splittable_candidate_filenames is a list of strings.
        process_args = [
            (i, config.seed, df_soundscape, splittable_candidate_filenames, MIN_TARGET_VAL_FILES, MAX_TARGET_VAL_FILES) 
            for i in range(N_STRATIFICATION_TRIALS)
        ]

        try:
            # Use context manager for Pool
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(tqdm(pool.imap_unordered(evaluate_sgkf_trial, process_args), total=N_STRATIFICATION_TRIALS, desc="SGKF Trials"))
            
            for res_cost, res_val_fn, res_train_fn, res_trial, res_fold_idx, res_actual_n_val in results:
                if res_cost < overall_best_fold_info['cost']:
                    overall_best_fold_info['cost'] = res_cost
                    overall_best_fold_info['val_filenames'] = res_val_fn
                    overall_best_fold_info['train_filenames'] = res_train_fn # This is relative to splittable_candidate_filenames
                    overall_best_fold_info['trial'] = res_trial
                    overall_best_fold_info['fold_in_trial'] = res_fold_idx
                    overall_best_fold_info['actual_n_val_files'] = res_actual_n_val
            
            if not overall_best_fold_info['val_filenames'] or overall_best_fold_info['cost'] == float('inf'):
                print("StratifiedGroupKFold (multi-trial) did not yield any valid folds with finite cost. Falling back.")
                raise ValueError("Multi-trial SGKF failed to produce a usable split.")

            val_filenames_list = overall_best_fold_info['val_filenames']
            train_pool_from_split_list = overall_best_fold_info['train_filenames'] # These are correct: all splittable not in val_filenames_list
            
            print(f"Selected best fold overall from {N_STRATIFICATION_TRIALS} trials (Trial {overall_best_fold_info['trial']}, Fold {overall_best_fold_info['fold_in_trial']}) with cost: {overall_best_fold_info['cost']:.4f}")
            print(f"  Target val file count was {MIN_TARGET_VAL_FILES}-{MAX_TARGET_VAL_FILES}. Actual best validation size: {overall_best_fold_info['actual_n_val_files']} files.")
            print(f"  Final val_filenames_list size: {len(val_filenames_list)}, Train pool from split size: {len(train_pool_from_split_list)}")
            sgkf_multitrial_success = True

        except Exception as e_sgkf_multi:
            print(f"StratifiedGroupKFold strategy (multi-trial with multiprocessing) failed: {e_sgkf_multi}.")
            sgkf_multitrial_success = False # Explicitly set to ensure fallback path

        # ---- Fallback Logics (IterativeStratification, then simple heuristic/random) ----
        if not sgkf_multitrial_success:
            print("Falling back after multi-trial SGKF failure.")
            iterative_stratification_success = False # Reset for this fallback path
            if SKML_AVAILABLE and len(splittable_candidate_filenames) > 1:
                print("Attempting IterativeStratification (multi-label stratification) as fallback...")
                try:
                    X_iter_strat = np.arange(len(splittable_candidate_filenames)).reshape(-1, 1)
                    df_candidate_detections_for_iter = df_soundscape[df_soundscape['filename'].isin(splittable_candidate_filenames)] # Use full df_soundscape here
                    unique_species_in_splittable_iter = sorted(df_candidate_detections_for_iter['primary_label'].unique())
                    species_to_col_idx_iter = {species: i for i, species in enumerate(unique_species_in_splittable_iter)}
                    y_ml_iter = np.zeros((len(splittable_candidate_filenames), len(unique_species_in_splittable_iter)), dtype=int)
                    filename_to_row_idx_iter = {fn: i for i, fn in enumerate(splittable_candidate_filenames)}

                    for fn_iter in splittable_candidate_filenames:
                        row_idx_iter = filename_to_row_idx_iter[fn_iter]
                        detections_in_this_file_iter = df_candidate_detections_for_iter[df_candidate_detections_for_iter['filename'] == fn_iter]
                        for species_iter in detections_in_this_file_iter['primary_label'].unique():
                            if species_iter in species_to_col_idx_iter:
                                col_idx_iter = species_to_col_idx_iter[species_iter]
                                y_ml_iter[row_idx_iter, col_idx_iter] = 1
                    
                    if y_ml_iter.shape[0] == 0 or y_ml_iter.shape[1] == 0: raise ValueError("y_ml matrix for IterativeStrat is empty")
                    if np.sum(y_ml_iter) == 0: raise ValueError("y_ml for IterativeStrat contains no positive labels")

                    # For IterativeStrat fallback, use the original VALIDATION_SET_SIZE_FRACTION
                    n_splits_for_iterative = max(2, int(np.ceil(1.0 / VALIDATION_SET_SIZE_FRACTION)))
                    stratifier_iter = IterativeStratification(n_splits=n_splits_for_iterative, order=1)
                    train_indices_iter, val_indices_iter = next(stratifier_iter.split(X_iter_strat, y_ml_iter))
                    
                    val_filenames_list = [splittable_candidate_filenames[i] for i in val_indices_iter]
                    train_pool_from_split_list = [splittable_candidate_filenames[i] for i in train_indices_iter]
                    print(f"Successfully performed IterativeStratification (fallback). Val size: {len(val_filenames_list)}, Train pool from split: {len(train_pool_from_split_list)}")
                    iterative_stratification_success = True
                except Exception as e_is:
                    print(f"IterativeStratification (fallback) failed: {e_is}. Falling back to simpler stratification.")
            
            if not iterative_stratification_success:
                print("Using final fallback stratification method (locally rarest species or random).")
                # Re-calculate stratify_labels for the current splittable_candidate_filenames if not already available for this exact set
                # This part of fallback assumes stratify_labels_for_splittable_candidates was prepared before SGKF attempt for the full splittable set
                # If splittable_candidate_filenames changed, these would need to be recalculated for this precise set.
                # However, stratify_labels_for_splittable_candidates are from the heuristic pre-filter stage, so they are for the current splittable set.
                
                final_fallback_labels = []
                # Need to regenerate labels for the current splittable_candidate_filenames if they weren't passed down or were for a different set
                # This ensures the stratify labels used are for the exact set of files being split here.
                df_final_fallback_detections = df_soundscape[df_soundscape['filename'].isin(splittable_candidate_filenames)]
                final_fallback_species_counts = df_final_fallback_detections['primary_label'].value_counts()
                for fn_ff in splittable_candidate_filenames:
                    dets_ff = df_final_fallback_detections[df_final_fallback_detections['filename'] == fn_ff]
                    sps_ff = dets_ff['primary_label'].unique()
                    if not sps_ff.any(): final_fallback_labels.append("NO_LABEL_FALLBACK_"+fn_ff.replace(".ogg","")); continue
                    counts_ff = final_fallback_species_counts.reindex(sps_ff).fillna(float('inf'))
                    final_fallback_labels.append(counts_ff.idxmin())

                if len(set(final_fallback_labels)) < 2 :
                     print("Warning: Only one unique stratification label for final fallback. Performing non-stratified split.")
                     train_pool_from_split_list, val_filenames_list = train_test_split(
                        splittable_candidate_filenames, test_size=VALIDATION_SET_SIZE_FRACTION, random_state=config.seed)
                else:
                    try:
                        train_pool_from_split_list, val_filenames_list = train_test_split(
                            splittable_candidate_filenames, test_size=VALIDATION_SET_SIZE_FRACTION, 
                            stratify=final_fallback_labels, random_state=config.seed)
                        print("Successfully performed final fallback stratified split (locally rarest species).")
                    except ValueError as e_final_fallback:
                        print(f"Final fallback stratified split failed: {e_final_fallback}. Using non-stratified split.")
                        train_pool_from_split_list, val_filenames_list = train_test_split(
                            splittable_candidate_filenames, test_size=VALIDATION_SET_SIZE_FRACTION, random_state=config.seed)

    # Combine all parts for the final training pool
    # train_pool_from_split_list contains filenames from splittable_candidate_filenames that are for training
    # files_to_force_train_pool_heuristic contains filenames from heuristic pre-filter
    # files_containing_rare_species contains filenames with globally rare species
    soundscape_training_pool_filenames = sorted(list(set(train_pool_from_split_list + files_containing_rare_species + files_to_force_train_pool_heuristic)))

    print(f"\n--- Results ---")
    print(f"Total unique soundscape files: {len(all_soundscape_filenames)}")
    print(f"Files with rare species (<= {RARE_SPECIES_SOUNDSCAPE_DETECTION_THRESHOLD} detections - directly to training pool): {len(files_containing_rare_species)}")
    print(f"Original candidate files (after excluding rare species files): {len(candidate_filenames_initial)}")
    print(f"  - Files moved to training pool due to pre-filter heuristic unique label: {len(files_to_force_train_pool_heuristic)}")
    print(f"  - Final splittable candidate files for main process: {len(splittable_candidate_filenames)}")
    print(f"    - Selected for Fixed Validation Set: {len(val_filenames_list)} files")
    print(f"    - Added to Soundscape Training Pool from split process: {len(train_pool_from_split_list)} files") # This count is from the splittable set
    print(f"Total files in Soundscape Training Pool (rare + pre-filter_forced + split_train_part): {len(soundscape_training_pool_filenames)}")
    
    # 6. Save the list of validation filenames
    os.makedirs(os.path.dirname(OUTPUT_FILENAME_LIST_PATH), exist_ok=True)
    try:
        with open(OUTPUT_FILENAME_LIST_PATH, 'w') as f:
            for filename in val_filenames_list:
                f.write(f"{filename}\n")
        print(f"Fixed soundscape validation filenames saved to: {OUTPUT_FILENAME_LIST_PATH}")
    except IOError as e:
        print(f"Error saving validation filenames: {e}")

    os.makedirs(os.path.dirname(OUTPUT_TRAIN_POOL_FILENAME_LIST_PATH), exist_ok=True)
    try:
        with open(OUTPUT_TRAIN_POOL_FILENAME_LIST_PATH, 'w') as f:
            for filename in soundscape_training_pool_filenames:
                f.write(f"{filename}\n")
        print(f"Soundscape training pool filenames saved to: {OUTPUT_TRAIN_POOL_FILENAME_LIST_PATH}")
    except IOError as e:
        print(f"Error saving training pool filenames: {e}")

if __name__ == "__main__":
    # This assumes your config object is globally available or loaded correctly
    if 'config' not in globals():
        print("Error: Configuration object 'config' not found. Ensure config.py is accessible and loaded.")
    else:
        create_fixed_soundscape_validation_set() 