from config import config
import pandas as pd

df = pd.read_csv(config.train_csv_path)

# Group by latitude and longitude and count occurrences
location_counts = df.groupby(['latitude', 'longitude', 'author', 'primary_label']).size().reset_index(name='count')

# Count how many times each count value appears
count_distribution = location_counts['count'].value_counts().sort_index()

print("\nDistribution of location pair frequencies:")
print("----------------------------------------")
print("Times a lat/lon/author/primary_label pair appears | Number of such pairs")
print("----------------------------------------")
for count_value, num_pairs in count_distribution.items():
    print(f"{count_value:>24} | {num_pairs:>19}")
