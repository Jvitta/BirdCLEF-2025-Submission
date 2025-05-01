import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/train_metadata_chunked_full2.csv')

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
plt.savefig('visualizations/label_distribution.png')
plt.close()

# Create histogram of file counts
plt.figure(figsize=(12, 6))
plt.hist(file_counts.values, bins=50, edgecolor='black')
plt.title('Distribution of Files per Audio Sample')
plt.xlabel('Number of Files')
plt.ylabel('Frequency')

plt.tight_layout()

# Save plot
plt.savefig('visualizations/file_count_distribution.png')
plt.close()

# Create histogram of unique filenames per label
plt.figure(figsize=(15, 8))
plt.hist(files_per_label.values, bins=50, edgecolor='black')
plt.title('Distribution of Unique Filenames per Primary Label')
plt.xlabel('Number of Unique Filenames')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig('visualizations/files_per_label_distribution.png')
plt.close() 





