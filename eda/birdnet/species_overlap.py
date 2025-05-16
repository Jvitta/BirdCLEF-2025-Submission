import pandas as pd
from config import config
import os
import difflib # Import the library

birdnet_species_list_path = os.path.join(config.RAW_DATA_DIR, 'birdnet_metadata.txt')

birdnet_df = pd.read_csv(
    birdnet_species_list_path,
    header=None,      
    names=['birdnet_species'] 
)

birdnet_df[['birdnet_scientific_name', 'birdnet_common_name']] = birdnet_df['birdnet_species'].str.split('_', expand=True)

taxonomy_df_full = pd.read_csv(config.taxonomy_path) # Load full taxonomy first
taxonomy_df = taxonomy_df_full[taxonomy_df_full['class_name'] == 'Aves'].copy() # Filter for Aves

print("--- BirdNet Species List (Processed Head) ---")
print(birdnet_df.head())

print("--- Taxonomy (Head) ---")
print(taxonomy_df.head())

is_in_birdnet = taxonomy_df['scientific_name'].isin(birdnet_df['birdnet_scientific_name'])

count = is_in_birdnet.sum()

print("Number of Aves in taxonomy: ", len(taxonomy_df))
print(f"Number of taxonomy species in BirdNET: {count}")

missing_species_df = taxonomy_df[~is_in_birdnet]

if not missing_species_df.empty:
    print("\n--- Species NOT Found in BirdNET List ---")
    print(missing_species_df[['primary_label', 'scientific_name', 'common_name']])

    # --- Find closest matches --- 
    print("\n--- Finding Closest Matches in BirdNET List ---")
    birdnet_scientific_names = birdnet_df['birdnet_scientific_name'].tolist() # Get list of names to search within

    for index, missing_row in missing_species_df.iterrows():
        missing_name = missing_row['scientific_name']
        # Get the top N closest matches (e.g., top 3)
        # cutoff=0.6 means only consider matches with a similarity ratio >= 0.6
        close_matches = difflib.get_close_matches(
            missing_name,
            birdnet_scientific_names,
            n=3, 
            cutoff=0.6
        )
        print(f"\nMissing: '{missing_name}' ({missing_row['common_name']})")
        if close_matches:
            print(f"  Closest matches found in BirdNET:")
            for match in close_matches:
                # Optional: Show common name from birdnet_df as well
                match_common_name = birdnet_df.loc[birdnet_df['birdnet_scientific_name'] == match, 'birdnet_common_name'].iloc[0]
                print(f"    - {match} ({match_common_name})")
        else:
            print(f"  No close matches found (similarity >= 0.6).")
    # --- End find closest matches --- 

else:
    print("\nAll Aves species from taxonomy were found in the BirdNET list.")


