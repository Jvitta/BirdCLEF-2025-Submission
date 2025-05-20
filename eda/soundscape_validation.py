import pandas as pd
import os
import sys

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming eda is directly under project_root
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config

def analyze_species_representation(config_obj):
    """
    Compares species representation between original training data (file counts)
    and calibrated soundscape pseudo-labels (detection counts).
    """
    train_csv_path = config_obj.train_csv_path
    soundscape_pseudo_labels_path = config_obj.soundscape_pseudo_calibrated_csv_path

    print(f"--- EDA: Species Representation Comparison ---")
    print(f"Loading training metadata from: {train_csv_path}")
    try:
        train_df = pd.read_csv(train_csv_path)
        if not all(col in train_df.columns for col in ['primary_label', 'filename']):
            print("Error: train.csv must contain 'primary_label' and 'filename' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Training CSV file not found at {train_csv_path}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading training CSV: {e}. Exiting.")
        return

    print(f"Loading calibrated soundscape pseudo-labels from: {soundscape_pseudo_labels_path}")
    try:
        soundscape_df = pd.read_csv(soundscape_pseudo_labels_path)
        if not all(col in soundscape_df.columns for col in ['primary_label', 'confidence']):
            print("Error: Soundscape pseudo-label CSV must contain 'primary_label' and 'confidence' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Calibrated soundscape pseudo-label file not found at {soundscape_pseudo_labels_path}.")
        print("Please ensure preprocess/calibrate_confidence.py has been run successfully.")
        return
    except Exception as e:
        print(f"Error loading soundscape pseudo-labels: {e}. Exiting.")
        return

    # 1. Calculate Training Data File Counts per Species
    train_species_file_counts = train_df.groupby('primary_label')['filename'].nunique().reset_index()
    train_species_file_counts.rename(columns={'filename': 'train_file_count'}, inplace=True)
    print(f"\nFound {len(train_species_file_counts)} species in training data.")

    # 2. Calculate Soundscape Pseudo-Label Detection Counts per Species
    if not soundscape_df.empty:
        soundscape_species_detection_counts = soundscape_df.groupby('primary_label').size().reset_index(name='soundscape_detection_count')
        print(f"Found {len(soundscape_species_detection_counts)} species in soundscape pseudo-labels.")
    else:
        print("Soundscape pseudo-label data is empty. No detection counts to calculate.")
        soundscape_species_detection_counts = pd.DataFrame(columns=['primary_label', 'soundscape_detection_count'])

    # 3. Combine Information
    # Start with training data counts as the base, then merge soundscape counts
    combined_df = pd.merge(train_species_file_counts, soundscape_species_detection_counts, on='primary_label', how='left')
    combined_df['soundscape_detection_count'] = combined_df['soundscape_detection_count'].fillna(0).astype(int)

    # In case some species are ONLY in soundscape data (unlikely for this comparison but good practice for general merge)
    # all_species = pd.DataFrame({'primary_label': pd.unique(list(train_species_file_counts['primary_label']) + list(soundscape_species_detection_counts['primary_label']))})
    # combined_df = pd.merge(all_species, train_species_file_counts, on='primary_label', how='left')
    # combined_df = pd.merge(combined_df, soundscape_species_detection_counts, on='primary_label', how='left')
    # combined_df.fillna(0, inplace=True)
    # combined_df['train_file_count'] = combined_df['train_file_count'].astype(int)
    # combined_df['soundscape_detection_count'] = combined_df['soundscape_detection_count'].astype(int)

    # 4. Sort and Display
    combined_df = combined_df.sort_values(by='soundscape_detection_count', ascending=False).reset_index(drop=True)

    print("\n--- Species Representation Summary ---")
    print("(Sorted by number of detections in soundscape pseudo-labels, descending)")
    pd.set_option('display.max_rows', None)
    print(combined_df)
    pd.reset_option('display.max_rows')
    
    # Save to CSV for easier review
    output_eda_dir = os.path.join(project_root, "eda", "outputs") # New subdirectory for EDA outputs
    os.makedirs(output_eda_dir, exist_ok=True)
    output_csv_path = os.path.join(output_eda_dir, "species_representation_summary.csv")
    try:
        combined_df.to_csv(output_csv_path, index=False)
        print(f"\nSummary saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving summary CSV: {e}")

if __name__ == "__main__":
    analyze_species_representation(config)
