# eda/eda_lat_lon.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Assuming config.py is one level up
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- Latitude/Longitude EDA ---")

# --- Load Data ---
try:
    print(f"Loading training metadata from: {config.train_csv_path}")
    train_df = pd.read_csv(config.train_csv_path)
    print(f"Loaded {len(train_df)} rows.")
except FileNotFoundError:
    print(f"Error: train.csv not found at {config.train_csv_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading train.csv: {e}. Exiting.")
    sys.exit(1)

# --- Check for Columns ---
required_cols = ['latitude', 'longitude']
if not all(col in train_df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in train_df.columns]
    print(f"Error: Missing required columns: {missing}. Exiting.")
    sys.exit(1)

# --- Data Cleaning & Validation ---
print("Cleaning and validating coordinate data...")
initial_rows = len(train_df)

# Attempt conversion to numeric, coercing errors to NaN
train_df['latitude'] = pd.to_numeric(train_df['latitude'], errors='coerce')
train_df['longitude'] = pd.to_numeric(train_df['longitude'], errors='coerce')

# Drop rows with invalid (NaN) coordinates
train_df.dropna(subset=['latitude', 'longitude'], inplace=True)
rows_after_nan_drop = len(train_df)
if initial_rows > rows_after_nan_drop:
    print(f"Dropped {initial_rows - rows_after_nan_drop} rows with non-numeric or missing coordinates.")

# Basic range check (latitude should be -90 to 90, longitude -180 to 180)
# Note: Some datasets might use 0/null island for unknown locations
lat_outliers = train_df[(train_df['latitude'] < -90) | (train_df['latitude'] > 90)]
lon_outliers = train_df[(train_df['longitude'] < -180) | (train_df['longitude'] > 180)]

if not lat_outliers.empty:
    print(f"Warning: Found {len(lat_outliers)} rows with latitude values outside [-90, 90].")
    # print(lat_outliers[['latitude', 'longitude']].head()) # Optionally print examples
if not lon_outliers.empty:
    print(f"Warning: Found {len(lon_outliers)} rows with longitude values outside [-180, 180].")
    # print(lon_outliers[['latitude', 'longitude']].head()) # Optionally print examples

# Consider filtering based on expected region (e.g., Colombia/South America) if needed,
# but for now let's just plot valid numeric ranges.

# --- Basic Statistics ---
print("\n--- Coordinate Statistics (after cleaning) ---")
print(train_df[['latitude', 'longitude']].describe())

# --- Visualization ---
print("\nGenerating scatter plot...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='longitude', y='latitude', s=5, alpha=0.5) # Smaller points, slight transparency
plt.title('Geographic Distribution of Training Recordings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Create EDA output directory if it doesn't exist
eda_output_dir = os.path.join(config.OUTPUT_DIR, "eda_plots")
os.makedirs(eda_output_dir, exist_ok=True)
plot_save_path = os.path.join(eda_output_dir, "lat_lon_distribution.png")

try:
    plt.savefig(plot_save_path)
    print(f"Saved plot to: {plot_save_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
# plt.show() # Uncomment if running interactively and want to see the plot immediately

print("\nEDA script finished.")
