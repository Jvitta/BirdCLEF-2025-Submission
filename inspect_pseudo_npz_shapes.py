import numpy as np
import os
from config import config

# Define the path to the pseudo-label NPZ file
pseudo_npz_path = os.path.join(config._PREPROCESSED_OUTPUT_DIR, 'pseudo_spectrograms.npz')

print(f"Attempting to load NPZ file from: {pseudo_npz_path}")

if not os.path.exists(pseudo_npz_path):
    print(f"Error: NPZ file not found at {pseudo_npz_path}")
    print("Please ensure 'preprocess_pseudo.py' has been run successfully.")
else:
    try:
        print("Loading NPZ file (this might take a moment)...")
        with np.load(pseudo_npz_path) as data_archive:
            keys = list(data_archive.keys())
            num_keys = len(keys)
            print(f"Successfully loaded NPZ file containing {num_keys} keys.")

            if num_keys > 0:
                limit = min(20, num_keys) # Limit to 20 or total keys if fewer
                print(f"\n--- Shapes of first {limit} arrays ---")
                for i in range(limit):
                    key = keys[i]
                    array_data = data_archive[key]
                    print(f"{i+1}: Key='{key}', Shape={array_data.shape}, Type={array_data.dtype}")
            else:
                print("The NPZ file is empty (contains no keys).")

    except Exception as e:
        print(f"An error occurred while loading or reading the NPZ file: {e}") 