import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This should be 'BirdCLEF-2025-Submission'
sys.path.append(project_root)

from config import config

def plot_confidence_distribution(config_obj):
    """
    Loads pseudo-labels and plots the distribution of their confidences.
    Saves the plot to eda/plots/pseudo_label_confidence_distribution.png.
    """
    print(f"Loading pseudo-labels from: {config_obj.soundscape_pseudo_csv_path}")
    
    try:
        df = pd.read_csv(config_obj.soundscape_pseudo_csv_path)
    except FileNotFoundError:
        print(f"Error: Pseudo-label file not found at {config_obj.soundscape_pseudo_csv_path}")
        print("Please ensure you have run 'preprocess/pseudo/birdnet_labels.py' first.")
        return
    except Exception as e:
        print(f"Error loading pseudo-labels: {e}")
        return

    if 'confidence' not in df.columns:
        print("Error: 'confidence' column not found in the pseudo-label CSV.")
        return
        
    if df.empty:
        print("Pseudo-label data is empty. No distribution to plot.")
        return

    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(project_root, "eda", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_dir, "pseudo_label_confidence_distribution.png")

    plt.figure(figsize=(12, 7))
    sns.histplot(df['confidence'], kde=True, bins=50, color='skyblue')
    plt.title('Distribution of BirdNET Detection Confidences in Soundscape Pseudo-Labels', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    
    # Add some descriptive statistics
    mean_conf = df['confidence'].mean()
    median_conf = df['confidence'].median()
    plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_conf:.2f}')
    plt.axvline(median_conf, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_conf:.2f}')
    plt.legend()

    try:
        plt.savefig(output_plot_path)
        print(f"Confidence distribution plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close() # Close the plot to free memory

def plot_confidence_distribution_per_species(config_obj):
    """
    Loads pseudo-labels and plots the confidence distribution for each species using facet plots.
    Saves the plot to eda/plots/pseudo_label_confidence_per_species.png.
    """
    print(f"Loading pseudo-labels for per-species analysis from: {config_obj.soundscape_pseudo_csv_path}")

    try:
        df = pd.read_csv(config_obj.soundscape_pseudo_csv_path)
    except FileNotFoundError:
        print(f"Error: Pseudo-label file not found at {config_obj.soundscape_pseudo_csv_path}")
        return
    except Exception as e:
        print(f"Error loading pseudo-labels: {e}")
        return

    if 'confidence' not in df.columns or 'primary_label' not in df.columns:
        print("Error: 'confidence' or 'primary_label' column not found in the pseudo-label CSV.")
        return
        
    if df.empty:
        print("Pseudo-label data is empty. No per-species distributions to plot.")
        return

    plot_dir = os.path.join(project_root, "eda", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_dir, "pseudo_label_confidence_per_species.png")

    unique_species = df['primary_label'].unique()
    n_species = len(unique_species)
    print(f"Found {n_species} unique species in pseudo-labels.")

    if n_species == 0:
        print("No species found to plot.")
        return

    # Determine grid layout
    ncols = 5  # Number of columns in the facet grid
    nrows = (n_species + ncols - 1) // ncols # Calculate rows needed

    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better facet plot aesthetics

    g = sns.FacetGrid(df, col='primary_label', col_wrap=ncols, height=2.5, aspect=1.5, sharex=True, sharey=False)
    g.map(sns.histplot, 'confidence', bins=25, kde=False, color='teal') # Using histplot, kde can be too slow/messy for many facets
    
    g.set_titles("{col_name}", size=10)
    g.set_axis_labels("Confidence", "Frequency")
    
    # Add a main title
    plt.subplots_adjust(top=0.92) # Adjust top to make space for suptitle
    g.fig.suptitle(f'BirdNET Detection Confidence Distribution per Species (Soundscape Pseudo-Labels)\\n{n_species} species shown', fontsize=16, y=0.98)


    try:
        plt.savefig(output_plot_path, dpi=150) # Increased DPI for better readability if there are many facets
        print(f"Per-species confidence distribution plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving per-species plot: {e}")
    
    plt.close(g.fig) # Close the plot

if __name__ == "__main__":
    # plot_confidence_distribution(config) # You can run the overall distribution if needed
    plot_confidence_distribution_per_species(config)
