import os
import sys
import numpy as np
import pandas as pd # For potentially loading metadata if needed for deeper inspection

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming eda is directly under project_root
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config

def verify_soundscape_val_npz():
    print("--- EDA: Verifying preprocessed_soundscape_val_specs.npz ---")

    npz_path = config.SOUNDSCAPE_VAL_NPZ_PATH
    val_filenames_list_path = os.path.join(config.PROCESSED_DATA_DIR, "fixed_soundscape_validation_filenames.txt")
    expected_shape = tuple(config.PREPROCESS_TARGET_SHAPE) # Ensure it's a tuple for comparison

    # 1. Load expected validation filenames
    try:
        with open(val_filenames_list_path, 'r') as f:
            expected_val_filenames = {line.strip() for line in f if line.strip()}
        if not expected_val_filenames:
            print(f"Error: The validation filenames list at {val_filenames_list_path} is empty.")
            return
        print(f"Loaded {len(expected_val_filenames)} unique filenames from fixed_soundscape_validation_filenames.txt")
    except FileNotFoundError:
        print(f"Error: Validation filenames list not found at {val_filenames_list_path}")
        return
    except Exception as e:
        print(f"Error reading validation filenames list: {e}")
        return

    # 2. Load the NPZ file
    try:
        data = np.load(npz_path, allow_pickle=True)
        detection_ids_in_npz = list(data.keys())
        if not detection_ids_in_npz:
            print(f"Error: The NPZ file at {npz_path} is empty or contains no keys.")
            return
        print(f"Loaded {len(detection_ids_in_npz)} spectrogram entries from {npz_path}")
    except FileNotFoundError:
        print(f"Error: NPZ file not found at {npz_path}. Please run preprocess_val.py first.")
        return
    except Exception as e:
        print(f"Error loading NPZ file {npz_path}: {e}")
        return

    # 3. Iterate and Verify
    correct_shape_count = 0
    incorrect_shape_count = 0
    filename_in_val_list_count = 0
    filename_not_in_val_list_count = 0

    incorrect_shape_examples = []
    filename_not_in_val_examples = []

    print(f"Expected spectrogram shape: {expected_shape}")

    for detection_id in detection_ids_in_npz:
        spectrogram = data[detection_id]
        
        # Shape Check
        if spectrogram.shape == expected_shape:
            correct_shape_count += 1
        else:
            incorrect_shape_count += 1
            if len(incorrect_shape_examples) < 5:
                incorrect_shape_examples.append(f"{detection_id} (Shape: {spectrogram.shape})")
        
        # Filename Check
        # Extract original filename: everything before the last "_idx_"
        try:
            original_filename = "_idx_".join(detection_id.split("_idx_")[:-1])
            if not original_filename:
                print(f"Warning: Could not parse original filename from detection_id: {detection_id}")
                original_filename = detection_id # Fallback to full ID if parsing fails
        except Exception:
            print(f"Warning: Error parsing original filename from detection_id: {detection_id}")
            original_filename = detection_id # Fallback

        if original_filename in expected_val_filenames:
            filename_in_val_list_count += 1
        else:
            filename_not_in_val_list_count += 1
            if len(filename_not_in_val_examples) < 5:
                filename_not_in_val_examples.append(f"{detection_id} (Original: {original_filename})")

    # 4. Report Summary
    print("\n--- Verification Summary ---")
    print(f"Total spectrograms in NPZ: {len(detection_ids_in_npz)}")
    
    print(f"\nShape Verification:")
    print(f"  Correct shape ({expected_shape}): {correct_shape_count}")
    print(f"  Incorrect shape: {incorrect_shape_count}")
    if incorrect_shape_examples:
        print("    Examples of incorrect shapes:")
        for ex in incorrect_shape_examples:
            print(f"      - {ex}")

    print(f"\nFilename Source Verification (based on fixed_soundscape_validation_filenames.txt):")
    print(f"  Filename found in validation list: {filename_in_val_list_count}")
    print(f"  Filename NOT found in validation list: {filename_not_in_val_list_count}")
    if filename_not_in_val_examples:
        print("    Examples of filenames not in validation list:")
        for ex in filename_not_in_val_examples:
            print(f"      - {ex}")
    
    if incorrect_shape_count == 0 and filename_not_in_val_list_count == 0 and len(detection_ids_in_npz) > 0:
        print("\nSUCCESS: All loaded spectrograms have the correct shape and originate from the expected validation files.")
    else:
        print("\nWARNING: Some issues found. Please review the counts and examples above.")

if __name__ == '__main__':
    verify_soundscape_val_npz()
