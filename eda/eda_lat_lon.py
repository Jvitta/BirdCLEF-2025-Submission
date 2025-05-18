# eda/eda_lat_lon.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math # Added for Haversine

# Assuming config.py is one level up
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- Latitude/Longitude EDA (Filtered for Colombia Bounding Box, with Distance Weighting) ---")

# --- Define Colombia Bounding Box ---
COLOMBIA_BBOX_MIN_LAT = -5.0
COLOMBIA_BBOX_MAX_LAT = 13.5
COLOMBIA_BBOX_MIN_LON = -80.0
COLOMBIA_BBOX_MAX_LON = -66.0

# --- Define El Silencio Reserve Coordinates and Weighting Parameters ---
RESERVE_LAT = 6.76
RESERVE_LON = -74.21
MIN_WEIGHT = 0.2
MAX_RELEVANT_DISTANCE_KM = 1000.0 # km
EARTH_RADIUS_KM = 6371.0

# --- Haversine Distance Function ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_RADIUS_KM * c
    return distance

# --- Weighting Function ---
def calculate_sample_weight(distance_km):
    if distance_km <= MAX_RELEVANT_DISTANCE_KM:
        slope = (1.0 - MIN_WEIGHT) / MAX_RELEVANT_DISTANCE_KM
        weight = 1.0 - (distance_km * slope)
    else:
        weight = MIN_WEIGHT
    return max(MIN_WEIGHT, weight) # Ensure weight doesn't go below MIN_WEIGHT due to precision

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

try:
    print(f"Loading taxonomy data from: {config.taxonomy_path}")
    taxonomy_df = pd.read_csv(config.taxonomy_path, dtype={'primary_label': str})
    print(f"Loaded {len(taxonomy_df)} taxonomy rows.")
    # Select and rename columns for easier merging if needed, e.g.
    # taxonomy_df = taxonomy_df[['primary_label', 'common_name', 'scientific_name']]
except FileNotFoundError:
    print(f"Error: taxonomy.csv not found at {config.taxonomy_path}. Proceeding without species names.")
    taxonomy_df = None
except AttributeError:
    print(f"Error: 'taxonomy_path' not found in config. Please check config.py. Proceeding without species names.")
    taxonomy_df = None
except Exception as e:
    print(f"Error loading taxonomy.csv: {e}. Proceeding without species names.")
    taxonomy_df = None

# --- Check for Columns ---
required_cols = ['latitude', 'longitude', 'primary_label']
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

# --- Calculate Distance and Weight for each sample ---
print("\nCalculating distance to reserve and sample weights...")
if 'latitude' in train_df.columns and 'longitude' in train_df.columns:
    train_df['distance_to_reserve_km'] = train_df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], RESERVE_LAT, RESERVE_LON), axis=1
    )
    train_df['sample_weight'] = train_df['distance_to_reserve_km'].apply(calculate_sample_weight)
    print("Added 'distance_to_reserve_km' and 'sample_weight' columns to train_df.")
    print(train_df[['distance_to_reserve_km', 'sample_weight']].describe())
    
    # Define unique_locations_df early as it's used for multiple calculations
    unique_locations_df = train_df.drop_duplicates(subset=['primary_label', 'latitude', 'longitude']).copy() # Ensure it's a copy
    print(f"Total unique location-species pairs identified: {len(unique_locations_df)}")
else:
    print("Skipping distance and weight calculation as latitude/longitude columns are missing.")
    train_df['distance_to_reserve_km'] = pd.NA # type: ignore
    train_df['sample_weight'] = pd.NA # type: ignore
    # Create an empty DataFrame with expected columns if unique_locations_df cannot be formed
    # This helps prevent downstream errors if subsequent merges expect these columns.
    unique_locations_df = pd.DataFrame(columns=['primary_label', 'latitude', 'longitude', 'distance_to_reserve_km', 'sample_weight'])

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

# --- Filter by Colombia Bounding Box ---
print(f"\nFiltering by Colombia bounding box: Lat ({COLOMBIA_BBOX_MIN_LAT} to {COLOMBIA_BBOX_MAX_LAT}), Lon ({COLOMBIA_BBOX_MIN_LON} to {COLOMBIA_BBOX_MAX_LON})")
train_df_colombia = train_df[
    (train_df['latitude'] >= COLOMBIA_BBOX_MIN_LAT) &
    (train_df['latitude'] <= COLOMBIA_BBOX_MAX_LAT) &
    (train_df['longitude'] >= COLOMBIA_BBOX_MIN_LON) &
    (train_df['longitude'] <= COLOMBIA_BBOX_MAX_LON)
].copy()
print(f"Found {len(train_df_colombia)} rows within the Colombia bounding box.")

if train_df_colombia.empty:
    print("Warning: No data points found within the specified Colombia bounding box. Scatter plot and species CSV will be empty or not generated.")

# --- Calculate Species Statistics within Bounding Box ---
print("\nCalculating species statistics based on unique locations...") # Updated print

# Total unique locations per species
if not unique_locations_df.empty:
    species_total_unique_locs = unique_locations_df['primary_label'].value_counts().reset_index()
    species_total_unique_locs.columns = ['primary_label', 'total_unique_locations']
    species_total_unique_locs['primary_label'] = species_total_unique_locs['primary_label'].astype(str)
    # This becomes the base for species_stats_df
    species_stats_df = species_total_unique_locs.copy()
else:
    # Fallback for empty unique_locations_df (e.g., if train_df had no valid lat/lon after cleaning)
    # Create a DataFrame from all primary_labels in train_df, with 0 counts
    print("Warning: unique_locations_df is empty. Initializing species_stats_df with 0 counts.")
    all_labels_in_train = train_df['primary_label'].astype(str).unique() if not train_df.empty else []
    species_stats_df = pd.DataFrame({
        'primary_label': pd.Series(all_labels_in_train, dtype=str),
        'total_unique_locations': 0
    })
    # Ensure other columns that will be merged/calculated are present if we start this way
    species_stats_df['median_dist_to_reserve_km'] = pd.NA
    species_stats_df['avg_weight'] = pd.NA
    species_stats_df['unique_locations_in_bbox'] = 0
    species_stats_df['percentage_unique_locations_in_bbox'] = 0.0


# --- Calculate Median Distance and Average Weight per Species (from unique locations) ---
# Median distance is based on unique lat/lon pairs per species
# Average weight is based on unique lat/lon pairs for species as well
if not unique_locations_df.empty and 'distance_to_reserve_km' in unique_locations_df.columns and 'sample_weight' in unique_locations_df.columns:
    # Check if unique_locations_df has any rows before groupby
    if not unique_locations_df.dropna(subset=['distance_to_reserve_km']).empty:
        species_median_dist = unique_locations_df.groupby('primary_label')['distance_to_reserve_km'].median().reset_index()
        species_median_dist.columns = ['primary_label', 'median_dist_to_reserve_km']
        species_median_dist['primary_label'] = species_median_dist['primary_label'].astype(str)
        species_stats_df = pd.merge(species_stats_df, species_median_dist, on='primary_label', how='left')
    else:
        species_stats_df['median_dist_to_reserve_km'] = pd.NA


    if not unique_locations_df.dropna(subset=['sample_weight']).empty:
        species_avg_weight = unique_locations_df.groupby('primary_label')['sample_weight'].mean().reset_index()
        species_avg_weight.columns = ['primary_label', 'avg_weight']
        species_avg_weight['primary_label'] = species_avg_weight['primary_label'].astype(str)
        species_stats_df = pd.merge(species_stats_df, species_avg_weight, on='primary_label', how='left')
    else:
        species_stats_df['avg_weight'] = pd.NA
# If unique_locations_df was initially empty, the NA columns were already added during its fallback initialization.


# Unique locations in bbox per species
if not train_df_colombia.empty:
    unique_locations_colombia_df = train_df_colombia.drop_duplicates(subset=['primary_label', 'latitude', 'longitude']).copy()
    if not unique_locations_colombia_df.empty:
        species_bbox_unique_locs = unique_locations_colombia_df['primary_label'].value_counts().reset_index()
        species_bbox_unique_locs.columns = ['primary_label', 'unique_locations_in_bbox']
        species_bbox_unique_locs['primary_label'] = species_bbox_unique_locs['primary_label'].astype(str)
        
        species_stats_df = pd.merge(species_stats_df, species_bbox_unique_locs, on='primary_label', how='left')
        # Ensure the column exists even if merge doesn't add it for all rows (though how='left' should handle this)
        if 'unique_locations_in_bbox' not in species_stats_df.columns:
             species_stats_df['unique_locations_in_bbox'] = 0
        species_stats_df['unique_locations_in_bbox'] = species_stats_df['unique_locations_in_bbox'].fillna(0).astype(int)
    else: # If train_df_colombia was not empty, but resulted in empty unique_locations_colombia_df
        if 'unique_locations_in_bbox' not in species_stats_df.columns: # Check if column needs to be added
             species_stats_df['unique_locations_in_bbox'] = 0
        else: # Column exists, just fill NaNs if any (though merge should handle primary_labels not in species_bbox_unique_locs)
            species_stats_df['unique_locations_in_bbox'] = species_stats_df['unique_locations_in_bbox'].fillna(0).astype(int)

else: # If train_df_colombia itself was empty
    if 'unique_locations_in_bbox' not in species_stats_df.columns:
        species_stats_df['unique_locations_in_bbox'] = 0
    else:
        species_stats_df['unique_locations_in_bbox'] = species_stats_df['unique_locations_in_bbox'].fillna(0).astype(int)


# Calculate percentage of unique locations in bbox
if 'total_unique_locations' in species_stats_df.columns and 'unique_locations_in_bbox' in species_stats_df.columns:
    # Ensure total_unique_locations is numeric for division and not zero
    species_stats_df['total_unique_locations'] = pd.to_numeric(species_stats_df['total_unique_locations'], errors='coerce').fillna(0)
    
    # Create a mask for rows where total_unique_locations is not 0
    valid_division_mask = species_stats_df['total_unique_locations'] != 0
    
    # Initialize column with 0.0
    species_stats_df['percentage_unique_locations_in_bbox'] = 0.0
    
    # Calculate percentage only for valid rows
    species_stats_df.loc[valid_division_mask, 'percentage_unique_locations_in_bbox'] = (
        species_stats_df.loc[valid_division_mask, 'unique_locations_in_bbox'] / 
        species_stats_df.loc[valid_division_mask, 'total_unique_locations']
    ) * 100
    
    species_stats_df['percentage_unique_locations_in_bbox'] = species_stats_df['percentage_unique_locations_in_bbox'].fillna(0).round(2)
else:
    species_stats_df['percentage_unique_locations_in_bbox'] = 0.0


# Merge with taxonomy data
if taxonomy_df is not None:
    # Ensure 'primary_label' exists in taxonomy_df and is the correct merge key
    if 'primary_label' in taxonomy_df.columns:
        # DEBUG: Check dtypes before merge
        # print("--- Debug: Dtypes before taxonomy merge ---")
        # print(f"species_stats_df['primary_label'].dtype: {species_stats_df['primary_label'].dtype}")
        # print(f"taxonomy_df['primary_label'].dtype: {taxonomy_df['primary_label'].dtype}")
        species_stats_df = pd.merge(species_stats_df, taxonomy_df[['primary_label', 'common_name', 'scientific_name']], on='primary_label', how='left')
        # DEBUG: Check for nulls after merge
        # print("--- Debug: Null common_names after taxonomy merge ---")
        # print(species_stats_df['common_name'].isnull().sum())

    else:
        print("Warning: 'primary_label' column not found in taxonomy data. Cannot merge species names.")
else:
    print("Skipping merge with taxonomy data as it was not loaded.")


# Sort by avg_weight descending
species_stats_df = species_stats_df.sort_values(by='avg_weight', ascending=False)

# Reorder columns for the CSV
if taxonomy_df is not None and 'common_name' in species_stats_df.columns and 'scientific_name' in species_stats_df.columns:
    csv_columns = ['primary_label', 'total_unique_locations', 'unique_locations_in_bbox', 'percentage_unique_locations_in_bbox', 
                   'median_dist_to_reserve_km', 'avg_weight',
                   'common_name', 'scientific_name']
    # Filter out any columns that might not exist if merge failed partially or names are missing
    csv_columns = [col for col in csv_columns if col in species_stats_df.columns]
    species_stats_df = species_stats_df[csv_columns]
elif taxonomy_df is None or ('common_name' not in species_stats_df.columns or 'scientific_name' not in species_stats_df.columns):
    # Fallback order if names are not available
    csv_columns = ['primary_label', 'total_unique_locations', 'unique_locations_in_bbox', 'percentage_unique_locations_in_bbox',
                   'median_dist_to_reserve_km', 'avg_weight']
    csv_columns = [col for col in csv_columns if col in species_stats_df.columns]
    species_stats_df = species_stats_df[csv_columns]

# Define EDA output directory relative to project root
eda_output_dir = os.path.join(project_root, "eda", "plots")
os.makedirs(eda_output_dir, exist_ok=True)

# --- Output Species Statistics CSV ---
stats_csv_save_path = os.path.join(eda_output_dir, "species_in_colombia_bbox.csv")
try:
    species_stats_df.to_csv(stats_csv_save_path, index=False)
    print(f"Saved species statistics to: {stats_csv_save_path}")
except Exception as e:
    print(f"Error saving species statistics CSV: {e}")


# --- Visualization ---
print("\nGenerating scatter plot for data within Colombia bounding box...")
plt.figure(figsize=(10, 6))
if not train_df_colombia.empty:
    sns.scatterplot(data=train_df_colombia, x='longitude', y='latitude', s=5, alpha=0.5) # Smaller points, slight transparency
    plt.title('Geographic Distribution of Training Recordings (Within Colombia Bounding Box)')
else:
    plt.title('Geographic Distribution - No Data in Colombia Bounding Box')
    plt.text(0.5, 0.5, "No data points in the specified bounding box",
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Save plot to the new eda_output_dir
plot_save_path = os.path.join(eda_output_dir, "lat_lon_distribution_colombia_bbox.png")

try:
    plt.savefig(plot_save_path)
    print(f"Saved plot to: {plot_save_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
# plt.show() # Uncomment if running interactively and want to see the plot immediately

print("\nEDA script finished.")
