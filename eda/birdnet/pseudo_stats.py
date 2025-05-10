import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Adjust path to import config if your config.py is in the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Define the path to the pseudo-label CSV file
# Path is relative to the project root.
PSEUDO_LABEL_CSV_PATH = os.path.join(project_root, "data", "raw_data", "train_pseudo.csv")

def analyze_pseudo_labels(csv_path):
    """
    Analyzes the BirdNET pseudo-label CSV file.
    """
    print(f"Attempting to load pseudo-labels from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        print("Please ensure the path is correct and the pseudo-label generation script has been run.")
        return

    try:
        df = pd.read_csv(csv_path)
        df = df[df['confidence'] > 0.9]
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    print("\n--- Basic DataFrame Info ---")
    print(f"Shape: {df.shape}")
    print("\nHead:")
    print(df.head())
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe(include='all'))

    # Common columns to analyze - adjust if your CSV has different names
    label_col = 'primary_label' # Or 'label', 'species', etc.
    scientific_name_col = 'scientific_name'
    confidence_col = 'confidence'
    filename_col = 'filename' # Soundscape filename
    start_time_col = 'start_time'
    end_time_col = 'end_time'

    if label_col in df.columns:
        print(f"\n--- Value Counts for {label_col} ---")
        print(df[label_col].value_counts())
        print(f"Number of unique {label_col}: {df[label_col].nunique()}")

    if scientific_name_col in df.columns:
        print(f"\n--- Value Counts for {scientific_name_col} ---")
        print(df[scientific_name_col].value_counts())
        print(f"Number of unique {scientific_name_col}: {df[scientific_name_col].nunique()}")
    
    if filename_col in df.columns:
        print(f"\n--- Detections per Soundscape File ({filename_col}) ---")
        print(df[filename_col].value_counts())
        print(f"Number of unique soundscape files with detections: {df[filename_col].nunique()}")


    if confidence_col in df.columns:
        print(f"\n--- Confidence Score ({confidence_col}) Distribution ---")
        print(df[confidence_col].describe())
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df[confidence_col], bins=20, kde=True)
        plt.title(f'Distribution of {confidence_col}')
        plt.xlabel(confidence_col)
        plt.ylabel('Frequency')
        plt.grid(True)
        # Try to save the plot
        plot_path = os.path.join(os.path.dirname(csv_path), "pseudo_label_confidence_distribution.png")
        try:
            plt.savefig(plot_path)
            print(f"Saved confidence distribution plot to {plot_path}")
        except Exception as e:
            print(f"Could not save confidence plot: {e}")
        # plt.show() # Uncomment if running in an environment that can display plots

    if start_time_col in df.columns and end_time_col in df.columns:
        print(f"\n--- Detection Duration (end_time - start_time) ---")
        df['detection_duration'] = df[end_time_col] - df[start_time_col]
        print(df['detection_duration'].describe())

        plt.figure(figsize=(10, 6))
        sns.histplot(df['detection_duration'], bins=20, kde=False)
        plt.title('Distribution of Pseudo-Label Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_duration_path = os.path.join(os.path.dirname(csv_path), "pseudo_label_duration_distribution.png")
        try:
            plt.savefig(plot_duration_path)
            print(f"Saved duration distribution plot to {plot_duration_path}")
        except Exception as e:
            print(f"Could not save duration plot: {e}")


    # Add more analyses as needed:
    # - How many unique labels per soundscape file?
    # - Correlation between confidence and number of detections?
    # - Time of day analysis if timestamps are detailed enough.

    print("\nEDA for pseudo-labels complete.")

if __name__ == '__main__':
    print(f"Attempting to run EDA on: {PSEUDO_LABEL_CSV_PATH}")
    
    # PSEUDO_LABEL_CSV_PATH is constructed using project_root, 
    # so it should be an absolute path or a path that os.path.exists can resolve.
    if not os.path.exists(PSEUDO_LABEL_CSV_PATH):
        print(f"File not found: {PSEUDO_LABEL_CSV_PATH}")
        print("Please ensure the path is correct. It's currently set in the script to:")
        print(f"  {PSEUDO_LABEL_CSV_PATH}")
        print("Verify that this file exists and the pseudo-label generation script has been run successfully.")
    else:
        analyze_pseudo_labels(PSEUDO_LABEL_CSV_PATH)

