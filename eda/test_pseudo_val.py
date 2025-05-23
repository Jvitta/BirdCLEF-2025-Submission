import pandas as pd
import os
import sys

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming eda is directly under project_root
if project_root not in sys.path:
    sys.path.append(project_root)

from config import config # For accessing configured paths, if needed, though direct paths are used below

def analyze_split_species_distribution():
    """
    Analyzes and reports the species distribution (detection counts)
    between the soundscape training pool and the fixed soundscape validation set.
    """
    print("--- EDA: Soundscape Split Species Distribution ---")

    # Define paths (adjust if they are managed differently, e.g., via config)
    soundscape_data_path = os.path.join(project_root, "data", "raw", "soundscape_pseudo_calibrated.csv")
    train_filenames_path = os.path.join(project_root, "data", "processed", "soundscape_train_pool_filenames.txt")
    val_filenames_path = os.path.join(project_root, "data", "processed", "fixed_soundscape_validation_filenames.txt")

    # 1. Load main soundscape data
    try:
        df_soundscape = pd.read_csv(soundscape_data_path)
        if not all(col in df_soundscape.columns for col in ['primary_label', 'filename']):
            print(f"Error: Soundscape CSV {soundscape_data_path} must contain 'primary_label' and 'filename'.")
            return
    except FileNotFoundError:
        print(f"Error: Soundscape data file not found at {soundscape_data_path}")
        return
    except Exception as e:
        print(f"Error loading soundscape data: {e}")
        return
    print(f"Loaded {len(df_soundscape)} total detections from soundscape data.")

    # 2. Load training filenames
    try:
        with open(train_filenames_path, 'r') as f:
            train_filenames = {line.strip() for line in f if line.strip()}
        if not train_filenames:
            print(f"Warning: Training filenames list at {train_filenames_path} is empty.")
    except FileNotFoundError:
        print(f"Error: Training filenames file not found at {train_filenames_path}")
        return
    except Exception as e:
        print(f"Error loading training filenames: {e}")
        return
    print(f"Loaded {len(train_filenames)} unique filenames for the training pool.")

    # 3. Load validation filenames
    try:
        with open(val_filenames_path, 'r') as f:
            val_filenames = {line.strip() for line in f if line.strip()}
        if not val_filenames:
            print(f"Warning: Validation filenames list at {val_filenames_path} is empty.")
    except FileNotFoundError:
        print(f"Error: Validation filenames file not found at {val_filenames_path}")
        return
    except Exception as e:
        print(f"Error loading validation filenames: {e}")
        return
    print(f"Loaded {len(val_filenames)} unique filenames for the validation set.")
    
    # Check for overlap in filenames between train and val (should ideally be 0)
    overlap_filenames = train_filenames.intersection(val_filenames)
    if overlap_filenames:
        print(f"Warning: {len(overlap_filenames)} filenames are present in BOTH training and validation sets. This indicates an issue in the split.")
        # print(f"Overlapping files: {list(overlap_filenames)[:5]}...") # Optionally print some

    # 4. Filter DataFrames
    df_train_detections = df_soundscape[df_soundscape['filename'].isin(train_filenames)]
    df_val_detections = df_soundscape[df_soundscape['filename'].isin(val_filenames)]

    print(f"Total detections in training pool files: {len(df_train_detections)}")
    print(f"Total detections in validation set files: {len(df_val_detections)}")

    # 5. Calculate detection counts per species for each split
    train_species_counts = df_train_detections['primary_label'].value_counts().reset_index()
    train_species_counts.columns = ['primary_label', 'train_detections']

    val_species_counts = df_val_detections['primary_label'].value_counts().reset_index()
    val_species_counts.columns = ['primary_label', 'val_detections']

    # 6. Combine counts
    # Start with all species present in the original soundscape data to not miss any
    all_species_in_soundscape = pd.DataFrame({'primary_label': df_soundscape['primary_label'].unique()})
    
    merged_counts = pd.merge(all_species_in_soundscape, train_species_counts, on='primary_label', how='left')
    merged_counts = pd.merge(merged_counts, val_species_counts, on='primary_label', how='left')

    merged_counts['train_detections'] = merged_counts['train_detections'].fillna(0).astype(int)
    merged_counts['val_detections'] = merged_counts['val_detections'].fillna(0).astype(int)

    # 7. Calculate total and percentage
    merged_counts['total_detections'] = merged_counts['train_detections'] + merged_counts['val_detections']
    # Calculate validation percentage, handle division by zero for species with 0 total detections (should not happen if based on all_species_in_soundscape)
    merged_counts['val_percentage'] = merged_counts.apply(
        lambda row: (row['val_detections'] / row['total_detections'] * 100) if row['total_detections'] > 0 else 0,
        axis=1
    )
    
    # Filter out species that have 0 total detections after filtering by train/val filenames (if any somehow slip through)
    # This might happen if a species was in a file not included in either list, though this should be rare with current setup
    merged_counts = merged_counts[merged_counts['total_detections'] > 0]

    # Sort for better readability, e.g., by validation percentage or total detections
    merged_counts = merged_counts.sort_values(by=['val_percentage', 'total_detections'], ascending=[False, False]).reset_index(drop=True)

    print("\n--- Species Detection Counts and Validation Percentage ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000) 
    print(merged_counts)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')

    # Save to CSV for easier review
    output_eda_dir = os.path.join(project_root, "eda", "outputs")
    os.makedirs(output_eda_dir, exist_ok=True)
    output_csv_path = os.path.join(output_eda_dir, "soundscape_split_species_distribution.csv")
    try:
        merged_counts.to_csv(output_csv_path, index=False)
        print(f"\nSplit distribution summary saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving summary CSV: {e}")

if __name__ == "__main__":
    analyze_split_species_distribution()

