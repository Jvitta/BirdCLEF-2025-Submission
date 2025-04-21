import os
import pickle
from config import config

print("--- VAD Key Transformation Script ---")

# Define input and output paths using config
input_pkl_path = config.VOICE_DATA_PKL_PATH # Original file with Kaggle paths
output_pkl_path = os.path.join(config.VOICE_SEPARATION_DIR, "transformed_train_voice_data.pkl") # New file

kaggle_prefix = "/kaggle/input/birdclef-2025/"
gcp_prefix = config.DATA_ROOT # Should be /home/jupyter/gcs_mount/raw_data

print(f"Input VAD file: {input_pkl_path}")
print(f"Output VAD file: {output_pkl_path}")
print(f"Transforming prefix: '{kaggle_prefix}' -> '{gcp_prefix}'")

try:
    # Load the original pickle file
    print("Loading original VAD data...")
    with open(input_pkl_path, 'rb') as f:
        raw_vad_data = pickle.load(f)
    print(f"Loaded {len(raw_vad_data)} entries.")

    # Transform keys
    print("Transforming keys...")
    transformed_vad_intervals = {}
    keys_updated = 0
    keys_skipped = 0
    for kaggle_path, intervals in raw_vad_data.items():
        if isinstance(kaggle_path, str) and kaggle_path.startswith(kaggle_prefix):
            suffix = kaggle_path[len(kaggle_prefix):]
            gcp_path = os.path.join(gcp_prefix, suffix)
            transformed_vad_intervals[gcp_path] = intervals
            keys_updated += 1
        else:
            print(f"Warning: Key '{kaggle_path}' does not start with expected prefix '{kaggle_prefix}'. Skipping.")
            keys_skipped += 1

    print(f"Transformation complete: {keys_updated} updated, {keys_skipped} skipped.")

    # Save the transformed dictionary to the new file
    print(f"Saving transformed data to {output_pkl_path}...")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(transformed_vad_intervals, f)
    print("Transformed data saved successfully.")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_pkl_path}. Cannot perform transformation.")
except Exception as e:
    print(f"An error occurred during transformation: {e}")

print("--- Script Finished ---") 