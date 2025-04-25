import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
from tqdm import tqdm
from config import config # <-- Add import

# --- Use Paths from Config --- #
METADATA_PATH = config.train_csv_path
AUDIO_DIR = config.train_audio_dir
VISUALIZATION_DIR = os.path.join(config.OUTPUT_DIR, 'visualizations')
# Create output directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
# --- End Path Configuration --- #

print(f"Loading metadata from: {METADATA_PATH}")
try:
    df = pd.read_csv(METADATA_PATH)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_PATH}. Cannot proceed with EDA.")
    exit()
except Exception as e:
     print(f"Error loading {METADATA_PATH}: {e}")
     exit()

# Construct full file paths
# Assuming filename is like 'species/file.ogg'
df['filepath'] = df['filename'].apply(lambda f: os.path.join(AUDIO_DIR, f))

# --- Calculate Audio Durations --- #
def get_audio_duration(filepath):
    """Safely get duration of an audio file."""
    try:
        if os.path.exists(filepath):
            return librosa.get_duration(filename=filepath)
        else:
            # print(f"Warning: File not found: {filepath}") # Optional warning
            return None
    except Exception as e:
        # print(f"Warning: Error processing {filepath}: {e}") # Optional warning
        return None

print("Calculating audio durations (this might take a while)...")
# Get unique filepaths to avoid redundant calculations
unique_files_df = df[['filepath']].drop_duplicates()
durations = []
for filepath in tqdm(unique_files_df['filepath'], desc="Getting Durations"):
    duration = get_audio_duration(filepath)
    if duration is not None:
        durations.append(duration)

if not durations:
    print("Error: Could not calculate durations for any files. Check paths and file integrity.")
    exit()

print(f"Calculated durations for {len(durations)} files.")
durations_series = pd.Series(durations)
# --- End Calculate Audio Durations --- #

# --- Plot Duration Distribution --- #
plt.figure(figsize=(12, 6))
plt.hist(durations_series, bins=50, edgecolor='black') 
plt.title('Distribution of Audio File Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()

duration_plot_path = os.path.join(VISUALIZATION_DIR, 'audio_duration_distribution.png')
print(f"Saving duration plot to: {duration_plot_path}")
try:
    plt.savefig(duration_plot_path)
except Exception as e:
    print(f"Error saving duration plot: {e}")
plt.close()
# --- End Plot Duration Distribution --- #

# --- Existing EDA Plots (Keep them) --- #

label_counts = df['primary_label'].value_counts().sort_values(ascending=False)

# Calculate the number of unique filenames per primary label
files_per_label = df.groupby('primary_label')['filename'].nunique().sort_values(ascending=False)

file_counts = df['filename'].value_counts().sort_values(ascending=False)

print(label_counts)

print(file_counts)

# Create bar plot of label counts
plt.figure(figsize=(15, 8))
sns.barplot(x=label_counts.values, y=label_counts.index)
plt.title('Distribution of Primary Labels')
plt.xlabel('Count')
plt.ylabel('Primary Label')

# Rotate y-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot
label_dist_path = os.path.join(VISUALIZATION_DIR, 'label_distribution.png')
print(f"Saving label distribution plot to: {label_dist_path}")
try:
    plt.savefig(label_dist_path)
except Exception as e:
    print(f"Error saving label distribution plot: {e}")
plt.close()

# Create histogram of file counts
plt.figure(figsize=(12, 6))
plt.hist(file_counts.values, bins=50, edgecolor='black')
plt.title('Distribution of Files per Audio Sample')
plt.xlabel('Number of Files')
plt.ylabel('Frequency')

plt.tight_layout()

# Save plot
file_count_path = os.path.join(VISUALIZATION_DIR, 'file_count_distribution.png')
print(f"Saving file count plot to: {file_count_path}")
try:
    plt.savefig(file_count_path)
except Exception as e:
    print(f"Error saving file count plot: {e}")
plt.close()

# Create histogram of unique filenames per label
plt.figure(figsize=(15, 8))
plt.hist(files_per_label.values, bins=50, edgecolor='black')
plt.title('Distribution of Unique Filenames per Primary Label')
plt.xlabel('Number of Unique Filenames')
plt.ylabel('Frequency')

plt.tight_layout()

files_per_label_path = os.path.join(VISUALIZATION_DIR, 'files_per_label_distribution.png')
print(f"Saving files per label plot to: {files_per_label_path}")
try:
    plt.savefig(files_per_label_path)
except Exception as e:
     print(f"Error saving files per label plot: {e}")
plt.close() 





