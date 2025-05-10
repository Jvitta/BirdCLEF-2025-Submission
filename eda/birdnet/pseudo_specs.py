import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import random
from tqdm.auto import tqdm

# Add project root to sys.path to allow importing config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import config

print("--- EDA: Comparing Soundscape Pseudo-Label Specs with Train Audio BN Specs ---")

# --- Configuration & Paths ---
NUM_SPECIES_TO_COMPARE = 20 # Changed from 5
NUM_COMPARISON_PLOTS_PER_SPECIES = 10 # New: How many comparison plots per species
SOUNDSCAPE_PSEUDO_CONF_THRESHOLD = 0.90 # As used in preprocess_birdnet_pseudo.py
TRAIN_AUDIO_BN_CONF_THRESHOLD = 0.75 # Threshold for considering a train_audio BN detection as "high confidence"

PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "pseudo_vs_train_bn_comparison")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- Load Necessary Data ---
print("Loading data...")
try:
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    if 'primary_label' in taxonomy_df.columns:
        taxonomy_df['primary_label'] = taxonomy_df['primary_label'].astype(str)
    else:
        print(f"Warning: 'primary_label' not in {config.taxonomy_path}")
        taxonomy_df = pd.DataFrame() # Empty if problematic

    # 1. Soundscape Pseudo-Labels (Metadata & Spectrograms)
    soundscape_pseudo_df = pd.read_csv(config.train_pseudo_csv_path)
    soundscape_pseudo_df_filtered = soundscape_pseudo_df[soundscape_pseudo_df['confidence'] >= SOUNDSCAPE_PSEUDO_CONF_THRESHOLD].copy()
    print(f"Loaded {len(soundscape_pseudo_df_filtered)} high-confidence soundscape pseudo-labels (>= {SOUNDSCAPE_PSEUDO_CONF_THRESHOLD})")
    
    with np.load(os.path.join(config._PREPROCESSED_OUTPUT_DIR, 'pseudo_spectrograms.npz')) as data:
        soundscape_specs_data = {k: data[k] for k in data.files}
    print(f"Loaded {len(soundscape_specs_data)} soundscape spectrogram entries.")

    # 2. Train Audio Data (Metadata, BirdNET Detections, Spectrograms)
    train_metadata_df = pd.read_csv(config.train_csv_path)
    if 'primary_label' in train_metadata_df.columns:
        train_metadata_df['primary_label'] = train_metadata_df['primary_label'].astype(str)
    
    with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
        train_audio_bn_detections = {k: data[k] for k in data.files}
    print(f"Loaded {len(train_audio_bn_detections)} BirdNET detection entries for train_audio.")

    with np.load(config.PREPROCESSED_NPZ_PATH) as data:
        train_audio_specs_data = {k: data[k] for k in data.files}
    print(f"Loaded {len(train_audio_specs_data)} train_audio spectrogram entries (samplenames).")

except FileNotFoundError as fnf_e:
    print(f"ERROR: A required data file was not found: {fnf_e}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading data: {e}. Exiting.")
    sys.exit(1)

# --- Find Common Species with High-Confidence Detections in Both Sets ---
soundscape_species_set = set(soundscape_pseudo_df_filtered['primary_label'].unique())

# Find species in train_audio that have high-confidence BirdNET detections
train_audio_bn_species_set = set()
for filename, detections in train_audio_bn_detections.items():
    if detections is not None:
        # Ensure detections is a list of dicts for consistent processing
        det_list = []
        if isinstance(detections, np.ndarray):
            det_list = [item for item in detections if isinstance(item, dict)]
        elif isinstance(detections, list):
            det_list = [d for d in detections if isinstance(d, dict)]

        for det_dict in det_list:
            if det_dict.get('confidence', 0) >= TRAIN_AUDIO_BN_CONF_THRESHOLD:
                # Need to get primary_label for this filename from train_metadata_df
                meta_row = train_metadata_df[train_metadata_df['filename'] == filename]
                if not meta_row.empty:
                    train_audio_bn_species_set.add(meta_row['primary_label'].iloc[0])
                break # Found one high-conf detection for this file, move to next file

common_species = list(soundscape_species_set.intersection(train_audio_bn_species_set))
print(f"\\nFound {len(common_species)} species with high-confidence pseudo-labels from soundscapes AND high-confidence BN detections in train_audio.")

if not common_species:
    print("No common species found for comparison based on current criteria. Exiting.")
    sys.exit(0)

# --- Select Species and Plot ---
species_to_plot = random.sample(common_species, min(NUM_SPECIES_TO_COMPARE, len(common_species)))
print(f"Will attempt to plot examples for up to {len(species_to_plot)} species: {species_to_plot}")

plt.style.use('seaborn-v0_8-whitegrid')

for species_label in tqdm(species_to_plot, desc="Processing Species"):
    species_name_info = ""
    if not taxonomy_df.empty and 'primary_label' in taxonomy_df.columns:
        tax_row = taxonomy_df[taxonomy_df['primary_label'] == species_label]
        if not tax_row.empty:
            common = tax_row['common_name'].iloc[0]
            scientific = tax_row['scientific_name'].iloc[0]
            species_name_info = f" ({common} / {scientific})"

    species_plot_dir = os.path.join(PLOT_OUTPUT_DIR, species_label.replace('/', '_').replace(':', '_'))
    os.makedirs(species_plot_dir, exist_ok=True)

    # 1. Collect all valid Soundscape Pseudo-Label Spectrogram candidates for this species
    valid_ss_candidates = []
    soundscape_rows_for_species = soundscape_pseudo_df_filtered[soundscape_pseudo_df_filtered['primary_label'] == species_label].copy()
    shuffled_soundscape_rows = soundscape_rows_for_species.sample(frac=1).reset_index(drop=True)

    for _, ss_row in shuffled_soundscape_rows.iterrows():
        ss_filename = ss_row['filename']
        ss_start_time = ss_row['start_time']
        ss_end_time = ss_row['end_time']
        ss_segment_key = f"{ss_filename}_{int(ss_start_time)}_{int(ss_end_time)}"
        ss_spec_array = soundscape_specs_data.get(ss_segment_key)
        if ss_spec_array is not None and isinstance(ss_spec_array, np.ndarray) and ss_spec_array.ndim == 3 and ss_spec_array.shape[0] > 0:
            valid_ss_candidates.append({
                "segment_key": ss_segment_key,
                "spec": ss_spec_array[0] # Assuming (1, H, W) -> (H, W)
            })

    # 2. Collect all valid Train Audio BirdNET Spectrogram candidates for this species
    valid_ta_candidates = []
    train_audio_files_for_species = train_metadata_df[train_metadata_df['primary_label'] == species_label]['filename'].tolist()
    random.shuffle(train_audio_files_for_species) # Shuffle files

    for ta_filename in train_audio_files_for_species:
        is_high_conf_bn_file = False
        if ta_filename in train_audio_bn_detections:
            bn_dets_for_file = train_audio_bn_detections[ta_filename]
            det_list_for_file = []
            if isinstance(bn_dets_for_file, np.ndarray):
                det_list_for_file = [item for item in bn_dets_for_file if isinstance(item, dict)]
            elif isinstance(bn_dets_for_file, list):
                det_list_for_file = [d for d in bn_dets_for_file if isinstance(d, dict)]
            
            if any(d.get('confidence', 0) >= TRAIN_AUDIO_BN_CONF_THRESHOLD for d in det_list_for_file):
                is_high_conf_bn_file = True

        if is_high_conf_bn_file:
            # This file is confirmed to have high-confidence BirdNET detections.
            # The preprocessing script should have saved its BirdNET-guided chunks 
            # (as a stack) in train_audio_specs_data under the base samplename.
            filename_no_ext = os.path.splitext(ta_filename)[0]
            base_samplename_for_file = filename_no_ext.replace('/', '-') # e.g., 'species_code-XC123456'
            
            if base_samplename_for_file in train_audio_specs_data:
                spec_chunks_array = train_audio_specs_data[base_samplename_for_file] # Should be (num_chunks, H, W)
                
                # Ensure it's a valid array of chunks. These are the _bnchunks.
                if spec_chunks_array is not None and spec_chunks_array.ndim == 3 and spec_chunks_array.shape[0] > 0:
                    num_available_chunks_for_this_file = spec_chunks_array.shape[0]
                    for chunk_idx_in_stack in range(num_available_chunks_for_this_file):
                        spec_2d = spec_chunks_array[chunk_idx_in_stack] # Individual (H, W) spectrogram
                        
                        # Reconstruct the individual chunk samplename for metadata/display consistency
                        individual_chunk_samplename = f"{base_samplename_for_file}_bnchunk{chunk_idx_in_stack}"

                        if spec_2d is not None and spec_2d.ndim == 2: # Final check on the 2D spec
                            valid_ta_candidates.append({
                                "samplename": individual_chunk_samplename,
                                "spec": spec_2d,
                                "original_filename": ta_filename,
                                "chunk_idx": chunk_idx_in_stack # Index of the chunk within this file's bn_chunks
                            })
    random.shuffle(valid_ta_candidates) # Shuffle all collected valid chunks from all files for this species

    # 3. Generate comparison plots
    plots_made_for_species = 0
    num_plots_to_attempt = min(len(valid_ss_candidates), len(valid_ta_candidates), NUM_COMPARISON_PLOTS_PER_SPECIES)

    if num_plots_to_attempt == 0:
        print(f"Skipping {species_label} - not enough valid data for comparison plots. Soundscape candidates: {len(valid_ss_candidates)}, Train audio candidates: {len(valid_ta_candidates)}")
        continue

    for i in range(num_plots_to_attempt):
        ss_candidate = valid_ss_candidates[i]
        ta_candidate = valid_ta_candidates[i]

        ss_spec_to_plot = ss_candidate["spec"]
        ss_segment_key_to_display = ss_candidate["segment_key"]

        train_audio_spec_to_plot = ta_candidate["spec"]
        ta_original_filename_display = ta_candidate["original_filename"]
        ta_chunk_idx_display = ta_candidate["chunk_idx"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Always 2 subplots for a pair
        fig.suptitle(f"Species: {species_label}{species_name_info} - Comparison {plots_made_for_species + 1}", fontsize=16)

        # Plot Soundscape Spectrogram
        axes[0].imshow(ss_spec_to_plot, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Soundscape Pseudo-Label\n{ss_segment_key_to_display}\nShape: {ss_spec_to_plot.shape}")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Mel Bins")

        # Plot Train Audio Spectrogram
        axes[1].imshow(train_audio_spec_to_plot, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f"Train Audio (BirdNET Guided Chunk)\nFile: ...{ta_original_filename_display[-30:]}\nChunk Index: {ta_chunk_idx_display}\nShape: {train_audio_spec_to_plot.shape}")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins")

        plot_filename = f"comparison_{plots_made_for_species + 1}.png"
        plot_path = os.path.join(species_plot_dir, plot_filename)
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
            plt.savefig(plot_path)
        except Exception as e_save_fig:
            print(f"Error saving figure {plot_filename} for species {species_label}: {e_save_fig}")
        plt.close(fig)
        plots_made_for_species += 1
    
    if plots_made_for_species > 0:
        print(f"Generated {plots_made_for_species} comparison plots for species {species_label} in {species_plot_dir}")

print(f"\nComparison plots saved to subdirectories within: {PLOT_OUTPUT_DIR}")
print("--- EDA Finished ---")
