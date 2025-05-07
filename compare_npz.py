# compare_npz.py
import numpy as np
import os
import sys

# --- Configuration ---
# Assuming this script is run from the project root directory
project_root = os.path.dirname(os.path.abspath(__file__)) 
npz_dir = os.path.join(project_root, "outputs", "preprocessed")
file1_name = "debug_spectrograms.npz"  # Baseline file
file2_name = "debug_spectrograms2.npz" # File after refactoring

file1_path = os.path.join(npz_dir, file1_name)
file2_path = os.path.join(npz_dir, file2_name)
# --- End Configuration ---

def compare_npz_files(f1_path, f2_path):
    """
    Compares two .npz files for identical keys and array contents.

    Args:
        f1_path (str): Path to the first .npz file.
        f2_path (str): Path to the second .npz file.

    Returns:
        bool: True if files are identical, False otherwise.
    """
    print(f"Comparing NPZ files:")
    print(f"  File 1: {f1_path}")
    print(f"  File 2: {f2_path}")

    try:
        # Use context managers to ensure files are closed
        with np.load(f1_path, allow_pickle=True) as data1, \
             np.load(f2_path, allow_pickle=True) as data2:

            keys1 = set(data1.keys())
            keys2 = set(data2.keys())

            # 1. Check if the sets of keys are identical
            if keys1 != keys2:
                print("\n[FAIL] NPZ files have different keys (samplenames)!")
                if keys1 - keys2:
                    print(f"  Keys only in {os.path.basename(f1_path)}: {sorted(list(keys1 - keys2))}")
                if keys2 - keys1:
                    print(f"  Keys only in {os.path.basename(f2_path)}: {sorted(list(keys2 - keys1))}")
                return False
            else:
                print(f"\n[PASS] Both files contain the same {len(keys1)} keys.")

            # 2. Check if the arrays for each key are identical
            all_arrays_match = True
            mismatched_keys = []
            print("Checking array contents for each key...")
            for key in sorted(list(keys1)): # Iterate in sorted order for consistency
                array1 = data1[key]
                array2 = data2[key]
                
                # Check shapes first (quick failure point)
                if array1.shape != array2.shape:
                     print(f"  [FAIL] Mismatch for key '{key}': Shapes differ - {array1.shape} vs {array2.shape}")
                     all_arrays_match = False
                     mismatched_keys.append(key)
                     continue # No need to check content if shapes differ
                     
                # Check data types
                if array1.dtype != array2.dtype:
                     print(f"  [WARN] Mismatch for key '{key}': Data types differ - {array1.dtype} vs {array2.dtype} (Continuing content check)")
                     # Might still be numerically equal, so continue check
                
                # Check content
                if not np.array_equal(array1, array2):
                    print(f"  [FAIL] Mismatch for key '{key}': Array contents are not equal.")
                    all_arrays_match = False
                    mismatched_keys.append(key)
                    # Optional: Add more detailed diff info here if needed
                    # diff = np.sum(np.abs(array1 - array2))
                    # print(f"     Sum of absolute differences: {diff}")

            if all_arrays_match:
                print("\n[PASS] All corresponding arrays are identical.")
                return True
            else:
                print(f"\n[FAIL] Found mismatches in arrays for keys: {sorted(list(set(mismatched_keys)))}")
                return False

    except FileNotFoundError:
        print(f"\n[ERROR] One or both NPZ files not found. Cannot compare.")
        print(f"  Checked for: {f1_path}")
        print(f"  Checked for: {f2_path}")
        return False
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("-" * 40)
    comparison_result = compare_npz_files(file1_path, file2_path)
    print("-" * 40)
    if comparison_result:
        print("Conclusion: The NPZ files are identical. Refactoring likely preserved preprocessing output.")
        # Optionally exit with success code
        # sys.exit(0) 
    else:
        print("Conclusion: The NPZ files differ. Review discrepancies.")
        # Optionally exit with failure code
        # sys.exit(1)