import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def verify_npz_file(npz_filepath):
    """Attempts to load and access all keys in an NPZ file to verify integrity."""
    print(f"--- Verifying NPZ file integrity: {npz_filepath} ---")
    data_archive = None
    error_count = 0
    success_count = 0
    try:
        if not os.path.exists(npz_filepath):
            print(f"  ERROR: NPZ file does not exist at path: {npz_filepath}")
            return False

        print("  Attempting to load NPZ archive...")
        # Use memory mapping cautiously if files are huge, but for 11GB direct load is ok
        data_archive = np.load(npz_filepath)
        keys = list(data_archive.keys())
        print(f"  Successfully loaded archive. Found {len(keys)} keys. Verifying access to each...")

        if not keys:
            print("  Warning: NPZ file contains no keys (is empty).")
            return True # An empty file isn't corrupt per se

        for key in tqdm(keys, desc="Verifying keys"):
            try:
                # The actual access is the verification step
                _ = data_archive[key]
                success_count += 1
            except Exception as e:
                print(f"\n  ERROR accessing key '{key}': {e}")
                error_count += 1
                # Optional: Stop after first error?
                # For verification, it's better to report all errors found.

        print("\n-- Verification Summary --")
        if error_count == 0:
            print(f"  Result: PASSED! Successfully accessed all {success_count} entries without errors.")
            return True
        else:
            print(f"  Result: FAILED! Encountered errors accessing {error_count} out of {len(keys)} entries.")
            print("  The NPZ file structure appears to be corrupted.")
            return False

    except Exception as e:
        print(f"\n  CRITICAL ERROR: Failed to load or process the NPZ archive itself: {e}")
        print("  This often indicates severe file corruption (e.g., incomplete write)." )
        return False
    finally:
        if data_archive is not None:
            try:
                data_archive.close()
            except Exception as e_close:
                 print(f"  Warning: Error closing NPZ archive during verification: {e_close}")

if __name__ == "__main__":
    # Hardcoded path
    npz_file_path = "outputs/preprocessed/spectrograms.npz"

    # Check if the relative path exists from the current working directory
    if not os.path.exists(npz_file_path):
        # If not, try constructing a path assuming the script is run from the project root
        # This depends on where the user runs the script from.
        # A more robust approach might involve finding the project root dynamically,
        # but for now, let's stick to the relative path.
        print(f"Error: Cannot find file at relative path: {npz_file_path}")
        print(f"Please ensure you are running this script from the project root directory,")
        print(f"or adjust the hardcoded path inside the script.")
        sys.exit(1)

    print(f"Starting verification for hardcoded path: {npz_file_path}")

    verification_passed = verify_npz_file(npz_file_path)

    if verification_passed:
        print("\nVerification finished: OK")
        sys.exit(0)
    else:
        print("\nVerification finished: FAILED")
        sys.exit(1) 