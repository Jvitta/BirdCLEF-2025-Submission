import os
import pandas as pd
import sys

# Add project root to sys.path to allow importing config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import config
except ImportError:
    print("Error: Could not import config. Make sure this script is runnable from the project root or utils/ is in PYTHONPATH.")
    sys.exit(1)

def generate_rare_csv():
    """
    Scans the config.train_audio_rare_dir, extracts primary_label from
    folder names, constructs filenames, and saves to config.train_rare_csv_path.
    The output CSV will have columns: primary_label, filename, url, license.
    URL and license will be empty strings.
    """
    rare_audio_dir = config.train_audio_rare_dir
    output_csv_path = config.train_rare_csv_path

    if not os.path.isdir(rare_audio_dir):
        print(f"Error: Rare audio directory not found: {rare_audio_dir}")
        return

    print(f"Scanning for rare audio files in: {rare_audio_dir}")
    data = []
    found_files = 0

    for species_folder in os.listdir(rare_audio_dir):
        species_path = os.path.join(rare_audio_dir, species_folder)
        if os.path.isdir(species_path):
            primary_label = species_folder # Assuming folder name is the primary label
            for audio_file in os.listdir(species_path):
                # Add checks for common audio extensions if needed
                if audio_file.lower().endswith(('.ogg', '.wav', '.mp3', '.flac', '.m4a')): # Add more if you use others
                    # Construct filename as primary_label/audio_file_name
                    filename_col_value = f"{primary_label}/{audio_file}"
                    data.append({
                        'primary_label': primary_label,
                        'filename': filename_col_value,
                        'url': '',  # Placeholder
                        'license': ''  # Placeholder
                    })
                    found_files += 1
    
    if not data:
        print("No audio files found in the rare species directory.")
        # Create an empty CSV with correct headers if no files are found
        df = pd.DataFrame(columns=['primary_label', 'filename', 'url', 'license'])
    else:
        df = pd.DataFrame(data)

    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully generated {output_csv_path} with {len(df)} entries ({found_files} audio files found).")
    except Exception as e:
        print(f"Error writing CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    generate_rare_csv() 