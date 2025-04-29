import pandas as pd

pseudo_labels_df = pd.read_csv('outputs/pseudo_labels.csv')

primary_label_counts = pseudo_labels_df['primary_label'].value_counts()

unique_labels = pseudo_labels_df['primary_label'].unique()

print(primary_label_counts)

print(len(unique_labels))


