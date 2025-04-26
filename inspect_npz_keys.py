import numpy as np
import os
import sys

# Need to import the config to get the NPZ path
try:
    # Assuming config.py is in the same directory or accessible via PYTHONPATH
    from config import config 
except ImportError:
    print("Error: Could not import config. Ensure config.py is accessible.")
    sys.exit(1)

def inspect_keys(npz_path):
    """Loads an NPZ file and prints its keys."""
    print(f"Inspecting NPZ file: {npz_path}")

    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    try:
        # Load the archive. mmap_mode='r' is good practice for large files
        with np.load(npz_path, mmap_mode='r') as data_archive:
            keys = list(data_archive.keys())
            
            if not keys:
                print("NPZ file is empty (contains no keys).")
                return

            print(f"Found {len(keys)} keys in the NPZ file.")
            
            # --- Print Sample Keys ---
            num_to_show = min(20, len(keys))
            print(f"\n--- First {num_to_show} Keys --- ")
            for i in range(num_to_show):
                print(f"  {i+1}: '{keys[i]}'")
            if len(keys) > num_to_show:
                print("  ...")
            print("---------------------")

            # --- Optional: Check for the special key we discussed ---
            special_key = '_successful_samplenames'
            if special_key in keys:
                 print(f"\nFound the special key: '{special_key}'")
                 # You could optionally print len(data_archive[special_key]) here
            else:
                 print(f"\nDid NOT find the special key: '{special_key}'")


            # --- Optional: Generate a sample key for comparison ---
            print("\n--- Generating Sample Key using Preprocessing Logic --- ")
            # Example filename format from metadata
            sample_filename = "asbfly/XC134896.ogg" 
            # Logic from preprocessing.py:
            generated_samplename = os.path.splitext(sample_filename.replace('/', '-'))[0]
            print(f"  Sample Filename: '{sample_filename}'")
            print(f"  Generated Samplename: '{generated_samplename}'")
            print("-----------------------------------------------------")


    except Exception as e:
        print(f"Error loading or reading NPZ file: {e}")

if __name__ == "__main__":
    # Get the NPZ path from the config instance
    npz_filepath = config.PREPROCESSED_NPZ_PATH
    inspect_keys(npz_filepath) 