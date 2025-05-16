import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import random
from tqdm.auto import tqdm
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec

# Assuming config.py is two levels up (e.g., from eda/ -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import config

print("--- EDA: Comprehensive Analysis of Preprocessed Spectrogram Chunks ---")

# --- Configuration ---
spectrogram_path = config.PREPROCESSED_NPZ_PATH
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "chunk_analysis")
os.makedirs(plot_dir, exist_ok=True)

# --- Load Metadata ---
print("Loading metadata to correlate with chunks...")
try:
    # Load main train.csv for class information
    df_train = pd.read_csv(config.train_csv_path)
    df_train['filename'] = df_train['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
    df_train['samplename'] = df_train['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])
    
    # Load rare species data if available
    if config.USE_RARE_DATA and os.path.exists(config.train_rare_csv_path):
        df_rare = pd.read_csv(config.train_rare_csv_path)
        df_rare['filename'] = df_rare['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
        df_rare['samplename'] = df_rare['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])
        
        # Combine and handle duplicates
        df_metadata = pd.concat([df_train, df_rare], ignore_index=True)
        df_metadata = df_metadata.drop_duplicates(subset=['samplename'], keep='first')
    else:
        df_metadata = df_train
        
    # Load taxonomy for class information
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    df_metadata = pd.merge(
        df_metadata, 
        taxonomy_df[['primary_label', 'class_name', 'scientific_name']], 
        on='primary_label', 
        how='left'
    )
    
    print(f"Loaded metadata for {len(df_metadata)} samples")
    
    # Load manual annotations data if available
    has_manual_annotations = False
    if os.path.exists(config.ANNOTATED_SEGMENTS_CSV_PATH):
        try:
            manual_ann_df = pd.read_csv(config.ANNOTATED_SEGMENTS_CSV_PATH)
            if 'filename' in manual_ann_df.columns:
                manual_ann_df['filename'] = manual_ann_df['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
                manual_ann_df['samplename'] = manual_ann_df['filename'].map(lambda x: os.path.splitext(x.replace('/', '-'))[0])
                has_manual_annotations = True
                print(f"Loaded manual annotations for {manual_ann_df['samplename'].nunique()} unique samples")
        except Exception as e:
            print(f"Error loading manual annotations: {e}")
            
except Exception as e:
    print(f"Warning: Could not load metadata files: {e}")
    # Create a minimal metadata df if we couldn't load the real one
    df_metadata = pd.DataFrame(columns=['samplename', 'primary_label'])

# --- Load Spectrogram Data ---
print(f"\nLoading processed spectrograms from: {spectrogram_path}")

# Structures to track statistics
chunks_by_sample = {}  # Store counts by samplename
chunks_by_species = defaultdict(int)  # Count by primary_label
chunks_per_file = []  # List of chunk counts per file
all_specs_shape = []  # Store shapes of all spectrograms
manually_annotated_samples = set()  # Samples with manual annotations

if has_manual_annotations:
    manually_annotated_samples = set(manual_ann_df['samplename'].unique())

try:
    with np.load(spectrogram_path) as data:
        samplenames = list(data.files)
        if not samplenames:
            print("Loaded NPZ file is empty. Exiting.")
            sys.exit(0)
        print(f"Found {len(samplenames)} samples in the NPZ.")

        # First pass to collect basic statistics
        for name in tqdm(samplenames, desc="Analyzing chunks"):
            try:
                array = data[name]
                if isinstance(array, np.ndarray) and array.ndim == 3:
                    num_chunks = array.shape[0]
                    chunks_by_sample[name] = num_chunks
                    chunks_per_file.append(num_chunks)
                    
                    # Get primary_label for this sample if available
                    sample_meta = df_metadata[df_metadata['samplename'] == name]
                    if not sample_meta.empty:
                        species = sample_meta.iloc[0]['primary_label']
                        chunks_by_species[species] += num_chunks
                    
                    # Record spectrogram shape (from first chunk)
                    if array.shape[0] > 0:
                        all_specs_shape.append((array.shape[1], array.shape[2]))
                    
                else:
                    print(f"Warning: Unexpected data format for samplename '{name}'. Skipping.")
            except Exception as e_inner:
                print(f"Warning: Error accessing data for samplename '{name}': {e_inner}. Skipping.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Spectrogram file not found at {spectrogram_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR loading spectrogram NPZ: {e}. Exiting.")
    sys.exit(1)

print(f"Analyzed {len(chunks_by_sample)} samples with a total of {sum(chunks_per_file)} chunks")

# --- Analysis and Visualization ---
print("\n--- Generating Analysis Plots ---")

# ----- 1. Basic Distribution of Chunks per File -----
plt.figure(figsize=(12, 8))
sns.histplot(chunks_per_file, discrete=True, kde=False)
plt.title("Distribution of Number of Chunks per File")
plt.xlabel("Number of Chunks")
plt.ylabel("Count of Files")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "1_chunks_per_file_distribution.png"))
plt.close()

# ----- 2. Dynamic Chunking Analysis -----
if df_metadata is not None and not df_metadata.empty:
    # Calculate species file counts
    species_file_counts = df_metadata.groupby('primary_label')['samplename'].nunique().to_dict()
    
    # Map species to their expected chunk counts based on dynamic chunking rules
    species_expected_chunks = {}
    if config.DYNAMIC_CHUNK_COUNTING:
        for species, file_count in species_file_counts.items():
            if file_count >= config.COMMON_SPECIES_FILE_THRESHOLD:
                species_expected_chunks[species] = config.MIN_CHUNKS_COMMON
            elif file_count <= 1:
                species_expected_chunks[species] = config.MAX_CHUNKS_RARE
            else:
                # Linear interpolation
                x1 = 1
                y1 = float(config.MAX_CHUNKS_RARE)
                x2 = float(config.COMMON_SPECIES_FILE_THRESHOLD - 1)
                y2 = float(config.MIN_CHUNKS_COMMON + 1)
                
                if x2 <= x1:
                    calculated_chunks = config.MAX_CHUNKS_RARE if file_count <= (x1 + x2) / 2 else config.MIN_CHUNKS_COMMON + 1
                else:
                    calculated_chunks = y1 + (float(file_count) - x1) * (y2 - y1) / (x2 - x1)
                
                species_expected_chunks[species] = min(max(int(round(calculated_chunks)), config.MIN_CHUNKS_COMMON), config.MAX_CHUNKS_RARE)
    else:
        # Static chunking - all species get the same number
        for species in species_file_counts:
            species_expected_chunks[species] = config.PRECOMPUTE_VERSIONS
    
    # Calculate actual mean chunks per species
    species_actual_chunks = {}
    for species, count in chunks_by_species.items():
        files_for_species = df_metadata[df_metadata['primary_label'] == species]['samplename'].nunique()
        if files_for_species > 0:
            species_actual_chunks[species] = count / files_for_species

    # Plot: Expected vs Actual Mean Chunks per Species
    if species_expected_chunks and species_actual_chunks:
        common_species = set(species_expected_chunks.keys()) & set(species_actual_chunks.keys())
        if common_species:
            plt.figure(figsize=(14, 7))
            
            x = [species_file_counts.get(s, 0) for s in common_species]
            y_expected = [species_expected_chunks.get(s, 0) for s in common_species]
            y_actual = [species_actual_chunks.get(s, 0) for s in common_species]
            
            plt.scatter(x, y_expected, alpha=0.7, label="Expected Chunks (Theory)")
            plt.scatter(x, y_actual, alpha=0.7, label="Actual Mean Chunks")
            
            plt.xscale('log')
            plt.title("Dynamic Chunking: Expected vs. Actual Mean Chunks per Species")
            plt.xlabel("Number of Files for Species (log scale)")
            plt.ylabel("Number of Chunks per File")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add vertical lines for thresholds
            plt.axvline(x=1, color='r', linestyle='--', alpha=0.5, 
                        label=f"Rare Threshold (1 file)")
            plt.axvline(x=config.COMMON_SPECIES_FILE_THRESHOLD, color='g', linestyle='--', alpha=0.5, 
                        label=f"Common Threshold ({config.COMMON_SPECIES_FILE_THRESHOLD} files)")
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "2_dynamic_chunking_analysis.png"))
            plt.close()

# ----- 3. Species Representation Analysis -----
if df_metadata is not None and not df_metadata.empty:
    # Get top 30 species by chunk count
    top_species = sorted(chunks_by_species.items(), key=lambda x: x[1], reverse=True)[:30]
    
    plt.figure(figsize=(14, 10))
    species_names = [s[0] for s in top_species]
    chunk_counts = [s[1] for s in top_species]
    
    # Get file counts for reference
    file_counts = [species_file_counts.get(s, 0) for s in species_names]
    
    # Create a DataFrame for easier plotting
    top_data = pd.DataFrame({
        'species': species_names,
        'chunks': chunk_counts,
        'files': file_counts,
        'avg_chunks_per_file': [c/f if f > 0 else 0 for c, f in zip(chunk_counts, file_counts)]
    })
    
    # Sort by chunks
    top_data = top_data.sort_values('chunks', ascending=False)
    
    # Create subplot with 2 axes
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax2 = ax1.twinx()
    
    # Plot bars for chunk counts
    sns.barplot(x='species', y='chunks', data=top_data, ax=ax1, alpha=0.7)
    ax1.set_ylabel('Total Chunk Count', fontsize=12)
    
    # Plot line for average chunks per file
    ax2.plot(range(len(top_data)), top_data['avg_chunks_per_file'].values, 'ro-', alpha=0.7)
    ax2.set_ylabel('Avg Chunks per File', fontsize=12, color='r')
    
    plt.title("Top 30 Species by Total Chunk Count")
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3_top_species_chunk_distribution.png"))
    plt.close()
    
    # Also create a table with this data
    top_data.to_csv(os.path.join(plot_dir, "top_species_chunk_stats.csv"), index=False)

# ----- 4. Rare vs Common Species Analysis -----
if df_metadata is not None and not df_metadata.empty:
    # Define rare species threshold (< 10 files)
    rare_threshold = 10
    
    # Identify rare and common species
    rare_species = {s: count for s, count in species_file_counts.items() if count < rare_threshold}
    common_species = {s: count for s, count in species_file_counts.items() if count >= rare_threshold}
    
    # Calculate chunk statistics
    rare_chunks = sum(chunks_by_species.get(s, 0) for s in rare_species)
    common_chunks = sum(chunks_by_species.get(s, 0) for s in common_species)
    
    # Pie chart of file counts vs chunk counts
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie([len(rare_species), len(common_species)], 
            labels=['Rare Species', 'Common Species'],
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'],
            startangle=90)
    plt.title(f"Species Count Distribution\n(Rare < {rare_threshold} files)")
    
    plt.subplot(1, 2, 2)
    plt.pie([rare_chunks, common_chunks], 
            labels=['Chunks from Rare Species', 'Chunks from Common Species'],
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'],
            startangle=90)
    plt.title("Chunk Count Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "4_rare_vs_common_distribution.png"))
    plt.close()
    
    # Create a summary stats table
    rare_common_stats = pd.DataFrame([
        {"Category": "Rare Species", "Count": len(rare_species), "Total Files": sum(rare_species.values()), 
         "Total Chunks": rare_chunks, "Avg Files/Species": sum(rare_species.values())/len(rare_species) if rare_species else 0,
         "Avg Chunks/Species": rare_chunks/len(rare_species) if rare_species else 0},
        {"Category": "Common Species", "Count": len(common_species), "Total Files": sum(common_species.values()), 
         "Total Chunks": common_chunks, "Avg Files/Species": sum(common_species.values())/len(common_species) if common_species else 0,
         "Avg Chunks/Species": common_chunks/len(common_species) if common_species else 0},
    ])
    rare_common_stats.to_csv(os.path.join(plot_dir, "rare_common_stats.csv"), index=False)

# ----- 5. Manual Annotations Impact (if available) -----
if has_manual_annotations:
    # Compare manually annotated samples vs non-annotated
    manual_samples_in_npz = manually_annotated_samples.intersection(set(chunks_by_sample.keys()))
    
    if manual_samples_in_npz:
        manual_chunks_per_file = [chunks_by_sample.get(s, 0) for s in manual_samples_in_npz]
        non_manual_samples = set(chunks_by_sample.keys()) - manually_annotated_samples
        non_manual_chunks_per_file = [chunks_by_sample.get(s, 0) for s in non_manual_samples]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(manual_chunks_per_file, discrete=True, kde=False, color='green', alpha=0.7)
        plt.title(f"Chunks per File - Manually Annotated\n({len(manual_samples_in_npz)} samples)")
        plt.xlabel("Number of Chunks")
        plt.ylabel("Count of Files")
        
        plt.subplot(1, 2, 2)
        sns.histplot(non_manual_chunks_per_file, discrete=True, kde=False, color='blue', alpha=0.7)
        plt.title(f"Chunks per File - No Manual Annotations\n({len(non_manual_samples)} samples)")
        plt.xlabel("Number of Chunks")
        plt.ylabel("Count of Files")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "5_manual_annotation_impact.png"))
        plt.close()
        
        # Summary statistics
        manual_annotation_stats = pd.DataFrame([
            {"Category": "Files with Manual Annotations", "Count": len(manual_samples_in_npz), 
             "Total Chunks": sum(manual_chunks_per_file), "Mean Chunks/File": np.mean(manual_chunks_per_file),
             "Median Chunks/File": np.median(manual_chunks_per_file)},
            {"Category": "Files without Manual Annotations", "Count": len(non_manual_samples), 
             "Total Chunks": sum(non_manual_chunks_per_file), "Mean Chunks/File": np.mean(non_manual_chunks_per_file),
             "Median Chunks/File": np.median(non_manual_chunks_per_file)}
        ])
        manual_annotation_stats.to_csv(os.path.join(plot_dir, "manual_annotation_stats.csv"), index=False)

# ----- 6. Class Analysis (Aves vs Non-Aves) -----
if 'class_name' in df_metadata.columns:
    # Create a mapping from samplename to class_name
    sample_to_class = dict(zip(df_metadata['samplename'], df_metadata['class_name']))
    
    # Count chunks by class
    chunks_by_class = defaultdict(int)
    for sample, chunk_count in chunks_by_sample.items():
        class_name = sample_to_class.get(sample, "Unknown")
        chunks_by_class[class_name] += chunk_count
    
    # Visualization for Aves vs Non-Aves
    if 'Aves' in chunks_by_class:
        aves_chunks = chunks_by_class.get('Aves', 0)
        non_aves_chunks = sum(v for k, v in chunks_by_class.items() if k != 'Aves')
        
        plt.figure(figsize=(10, 6))
        plt.pie([aves_chunks, non_aves_chunks], 
                labels=['Aves', 'Non-Aves'],
                autopct='%1.1f%%',
                colors=['#ff9999','#66b3ff'],
                startangle=90)
        plt.title("Chunk Distribution: Aves vs Non-Aves")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "6_aves_vs_nonaves_distribution.png"))
        plt.close()

# ----- 7. Summary Statistics -----
# Create a comprehensive summary dataframe
summary_stats = {
    "Total Samples in NPZ": len(chunks_by_sample),
    "Total Chunks": sum(chunks_per_file),
    "Mean Chunks per File": np.mean(chunks_per_file),
    "Median Chunks per File": np.median(chunks_per_file),
    "Min Chunks per File": min(chunks_per_file) if chunks_per_file else 0,
    "Max Chunks per File": max(chunks_per_file) if chunks_per_file else 0,
    "Total Species": len(chunks_by_species),
    "Spectrogram Shape": Counter(all_specs_shape).most_common(1)[0][0] if all_specs_shape else "N/A",
}

# Write summary to file
with open(os.path.join(plot_dir, "summary_statistics.txt"), "w") as f:
    f.write("=== Preprocessed NPZ Analysis Summary ===\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")
    
    # Add config information
    f.write("\n=== Preprocessing Configuration ===\n")
    f.write(f"Dynamic Chunking: {config.DYNAMIC_CHUNK_COUNTING}\n")
    if config.DYNAMIC_CHUNK_COUNTING:
        f.write(f"MAX_CHUNKS_RARE: {config.MAX_CHUNKS_RARE}\n")
        f.write(f"MIN_CHUNKS_COMMON: {config.MIN_CHUNKS_COMMON}\n")
        f.write(f"COMMON_SPECIES_FILE_THRESHOLD: {config.COMMON_SPECIES_FILE_THRESHOLD}\n")
    else:
        f.write(f"PRECOMPUTE_VERSIONS: {config.PRECOMPUTE_VERSIONS}\n")
    f.write(f"USE_RARE_DATA: {config.USE_RARE_DATA}\n")
    f.write(f"REMOVE_SPEECH_INTERVALS: {config.REMOVE_SPEECH_INTERVALS}\n")

print(f"\nEDA complete! Results saved to: {plot_dir}")
print("Summary of findings:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")
