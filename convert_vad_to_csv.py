import os
import pickle
import pandas as pd
from config import config

def pkl_to_csv(input_pkl_path, output_csv_path):
    """Loads a VAD pickle file and converts it to a CSV.

    The pickle file is expected to be a dictionary where keys are filepaths
    and values are lists of dictionaries like [{'start': s, 'end': e}, ...].

    The output CSV will have columns: filepath, start_time, end_time.
    """
    print(f"--- Converting {input_pkl_path} to {output_csv_path} ---")
    try:
        # Load the pickle file
        print(f"Loading pickle data from {input_pkl_path}...")
        with open(input_pkl_path, 'rb') as f:
            vad_data = pickle.load(f)
        print(f"Loaded {len(vad_data)} file entries.")

        # Prepare data for DataFrame
        csv_data = []
        total_intervals = 0
        for filepath, intervals in vad_data.items():
            if isinstance(intervals, list):
                for interval in intervals:
                    if isinstance(interval, dict) and 'start' in interval and 'end' in interval:
                        csv_data.append({
                            'filepath': filepath,
                            'start_time': interval['start'],
                            'end_time': interval['end']
                        })
                        total_intervals += 1
                    else:
                        print(f"Warning: Invalid interval format found for {filepath}: {interval}. Skipping interval.")
            else:
                 print(f"Warning: Invalid intervals value found for {filepath}: {intervals}. Skipping entry.")
        
        if not csv_data:
             print("No valid intervals found to convert to CSV.")
             return

        # Create DataFrame and save to CSV
        print(f"Creating DataFrame with {len(csv_data)} rows ({total_intervals} intervals)...")
        df = pd.DataFrame(csv_data)
        print(f"Saving CSV data to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False)
        print("CSV saved successfully.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_pkl_path}. Cannot convert.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
    print("---")

if __name__ == "__main__":
    print("--- VAD Pickle to CSV Conversion Script ---")

    # Define paths using config
    original_pkl = config.VOICE_DATA_PKL_PATH
    transformed_pkl = config.TRANSFORMED_VOICE_DATA_PKL_PATH

    original_csv = os.path.join(config.VOICE_SEPARATION_DIR, "original_vad_intervals.csv")
    transformed_csv = os.path.join(config.VOICE_SEPARATION_DIR, "transformed_vad_intervals.csv")

    # Convert original file
    pkl_to_csv(original_pkl, original_csv)

    # Convert transformed file
    pkl_to_csv(transformed_pkl, transformed_csv)

    print("--- Script Finished ---") 